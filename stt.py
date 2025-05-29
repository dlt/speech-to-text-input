import argparse
import pyaudio
import numpy as np
import whisper
import webrtcvad
import collections
import json
import os
import time
import subprocess
import sys
import urllib.request
import zipfile
from vosk import Model, KaldiRecognizer

# Model configurations
MODELS = {
    'en': {
        'small': {
            'name': 'vosk-model-small-en-us-0.15',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
            'size': '40 MB'
        },
        'large': {
            'name': 'vosk-model-en-us-0.42-gigaspeech', 
            'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip',
            'size': '2.3 GB'
        }
    },
    'pt': {
        'small': {
            'name': 'vosk-model-small-pt-0.3',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip',
            'size': '31 MB'
        },
        'large': {
            'name': 'vosk-model-pt-fb-v0.1.1-20220516_2113',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip',
            'size': '1.6 GB'
        }
    }
}

# Debug mode
DEBUG = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')

# Platform-specific keyboard typing
def type_text(text):
    """Type text at current cursor position"""
    if sys.platform == "darwin":  # macOS
        # Use osascript to type text
        script = f'tell application "System Events" to keystroke "{text}"'
        subprocess.run(["osascript", "-e", script])
    else:
        # For other platforms, we'd need pyautogui
        try:
            import pyautogui
            pyautogui.typewrite(text)
        except ImportError:
            print("❌ pyautogui not installed. Run: pip install pyautogui")
            print(f"📋 Text copied to clipboard: {text}")
            return

def download_model(model_info, model_name):
    """Download and extract a Vosk model if not present"""
    if os.path.exists(model_name):
        return True
    
    zip_file = f"{model_name}.zip"
    
    # Check if zip already exists
    if not os.path.exists(zip_file):
        print(f"📥 Downloading {model_name} ({model_info['size']})...")
        print(f"   This may take a while for large models...")
        
        try:
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
            
            urllib.request.urlretrieve(model_info['url'], zip_file, download_progress)
            print()  # New line after progress
        except Exception as e:
            print(f"\n❌ Failed to download model: {e}")
            return False
    
    # Extract zip
    print(f"📦 Extracting {model_name}...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ Model {model_name} ready!")
        return True
    except Exception as e:
        print(f"❌ Failed to extract model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Speech-to-text with wake word activation')
    parser.add_argument('--model-en', choices=['small', 'large'], default='small',
                        help='English model size (default: small)')
    parser.add_argument('--model-pt', choices=['small', 'large'], default='small',
                        help='Portuguese model size (default: small)')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                        default='base', help='Whisper model size (default: base)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    # Audio constants
    RATE = 16000
    CHUNK_MS = 30
    CHUNK = int(RATE * CHUNK_MS / 1000)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    AUDIO_GAIN = 10.0  # Amplification factor
    
    # VAD config
    vad = webrtcvad.Vad(1)  # Lower sensitivity (1 instead of 2)
    MAX_SILENT = int(1000 / CHUNK_MS)  # 2 seconds instead of 5
    
    # Setup audio
    pa = pyaudio.PyAudio()
    devices = []
    
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            devices.append((i, dev['name']))
    
    if args.list_devices:
        print("🔍 Available input devices:")
        for i, name in devices:
            print(f"  [{i}] {name}")
        return
    
    if not devices:
        print("❌ No microphone devices found.")
        exit(1)
    
    # Device selection
    if DEBUG or len(devices) > 1:
        print("🔍 Available input devices:")
        for i, name in devices:
            print(f"  [{i}] {name}")
        
        if len(devices) > 1:
            print("\n🎯 Enter device number (or press Enter for default): ", end="")
            try:
                choice = input().strip()
                if choice:
                    input_device_index = int(choice)
                else:
                    input_device_index = devices[0][0]
            except:
                input_device_index = devices[0][0]
        else:
            input_device_index = devices[0][0]
    else:
        input_device_index = devices[0][0]
    
    print(f"\n🎙️ Using device index [{input_device_index}]")
    if DEBUG:
        print(f"🔊 Audio gain set to {AUDIO_GAIN}x")
    
    # Download models if needed
    en_model_info = MODELS['en'][args.model_en]
    pt_model_info = MODELS['pt'][args.model_pt]
    
    print("\n📦 Checking models...")
    if not download_model(en_model_info, en_model_info['name']):
        exit(1)
    if not download_model(pt_model_info, pt_model_info['name']):
        exit(1)
    
    # Load models
    print("\n📦 Loading models...")
    print(f"  English: {en_model_info['name']} ({args.model_en})")
    print(f"  Portuguese: {pt_model_info['name']} ({args.model_pt})") 
    print(f"  Whisper: {args.whisper_model}")
    
    vosk_en = Model(en_model_info['name'])
    vosk_pt = Model(pt_model_info['name'])
    whisper_model = whisper.load_model(args.whisper_model)
    print("✅ Models loaded.")
    
    # Create recognizers
    rec_en = KaldiRecognizer(vosk_en, RATE)
    rec_pt = KaldiRecognizer(vosk_pt, RATE)
    rec_en.SetWords(True)  # Enable words for better recognition
    rec_pt.SetWords(True)  # Enable words for better recognition
    
    # Start audio stream
    try:
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                         input=True, input_device_index=input_device_index,
                         frames_per_buffer=CHUNK)
    except Exception as e:
        print("❌ Failed to open audio stream:", e)
        exit(1)
    
    print("\n🎤 Say 'transcribe' (EN) or 'transcreva' (PT) to begin...")
    print("📌 The transcribed text will be typed at your cursor position")
    
    if DEBUG:
        # Test audio stream
        print("\n🔊 Testing audio capture for 2 seconds...")
        test_frames = []
        for i in range(int(2 * RATE / CHUNK)):  # 2 seconds
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio_array).mean()
                max_level = np.abs(audio_array).max()
                test_frames.append(level)
                if i % 10 == 0:  # Print every ~300ms
                    print(f"  Audio level: avg={level:.0f}, max={max_level:.0f}")
            except Exception as e:
                print(f"❌ Error reading audio: {e}")
                break
        
        if test_frames:
            avg_level = np.mean(test_frames)
            print(f"\n📊 Average audio level over 2s: {avg_level:.1f}")
            if avg_level < 10:
                print("⚠️  Very low audio levels detected. Audio will be amplified.")
                print(f"   Raw levels will be boosted {AUDIO_GAIN}x")
            else:
                print("✅ Audio capture appears to be working")
        else:
            print("❌ No audio data captured!")
    
    print("\n🎯 Listening for wake words...")
    print("💡 Tip: Click where you want the text to appear before speaking")
    
    
    def detect_wake_word():
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Amplify audio
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array * AUDIO_GAIN
            audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
            amplified_data = audio_array.tobytes()
            
            # Debug: Check audio level
            if DEBUG:
                audio_level = np.abs(audio_array).mean()
                if audio_level > 50:  # Lowered threshold
                    print(f"🔊 Audio level: {audio_level:.0f}")
    
            # Check English model with amplified audio
            if rec_en.AcceptWaveform(amplified_data):
                result = json.loads(rec_en.Result())
                text = result.get("text", "").lower()
                if text:  # Only print if there's text
                    if DEBUG:
                        print(f"[EN MODEL] → {text}")
                    if "transcribe" in text:
                        print("🟢 Wake word detected: 'transcribe'")
                        return "en"
            else:
                # Get partial results
                if DEBUG:
                    partial = json.loads(rec_en.PartialResult())
                    if partial.get("partial"):
                        print(f"[EN PARTIAL] → {partial['partial']}")
    
            # Check Portuguese model with amplified audio
            if rec_pt.AcceptWaveform(amplified_data):
                result = json.loads(rec_pt.Result())
                text = result.get("text", "").lower()
                if text:  # Only print if there's text
                    if DEBUG:
                        print(f"[PT MODEL] → {text}")
                    if "transcreva" in text:
                        print("🟢 Wake word detected: 'transcreva'")
                        return "pt"
            else:
                # Get partial results
                if DEBUG:
                    partial = json.loads(rec_pt.PartialResult())
                    if partial.get("partial"):
                        print(f"[PT PARTIAL] → {partial['partial']}")
    
    
    def record_until_silence():
        print("🎙️ Recording... Speak now.")
        frames = []
        ring_buffer = collections.deque(maxlen=MAX_SILENT)
        
        # Pre-fill buffer with True to avoid immediate stop
        for _ in range(MAX_SILENT // 2):
            ring_buffer.append(True)
    
        while True:
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(chunk, dtype=np.int16))
            
            # Debug VAD
            is_speech = vad.is_speech(chunk, RATE)
            ring_buffer.append(is_speech)
            
            if len(frames) > 10 and not any(ring_buffer):
                if DEBUG:
                    print("🔕 Silence detected. Stopping recording.")
                break
            elif DEBUG and len(frames) % 33 == 0:  # Every ~1 second
                print(f"⏱️  Recording... {len(frames) * CHUNK_MS / 1000:.1f}s")
    
        audio_data = np.hstack(frames)
        if DEBUG:
            print(f"📊 Recorded {len(audio_data) / RATE:.1f} seconds of audio")
        return audio_data.astype(np.float32) / 32768.0
    
    
    # Main loop
    try:
        while True:
            lang = detect_wake_word()
            audio = record_until_silence()
            if DEBUG:
                print("🧠 Transcribing with Whisper...")
            result = whisper_model.transcribe(audio, language=lang)
            text = result["text"].strip()
            
            if text:
                print(f"📝 Transcribed: {text}")
                
                # Small delay to ensure user is ready
                print("✍️  Typing in 0.5 seconds...")
                time.sleep(0.5)
                
                # Type the text where the cursor is
                type_text(text)
                
                # Audio feedback - system sound
                if sys.platform == "darwin":
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], capture_output=True)
                
                print("✅ Text typed at cursor position")
            else:
                print("⚠️  No text was transcribed")
    
    except KeyboardInterrupt:
        print("🛑 Exiting.")
    except Exception as e:
        print(f"❌ Error: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()