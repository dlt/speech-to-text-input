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
from typing import Dict, Optional, Tuple, List
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


class TextTyper:
    """Handles typing text at cursor position across different platforms."""

    @staticmethod
    def type_text(text: str) -> None:
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


class ModelManager:
    """Manages downloading and loading of speech recognition models."""

    def __init__(self, en_size: str, pt_size: str, whisper_size: str):
        self.en_model_info = MODELS['en'][en_size]
        self.pt_model_info = MODELS['pt'][pt_size]
        self.whisper_size = whisper_size
        self.vosk_en: Optional[Model] = None
        self.vosk_pt: Optional[Model] = None
        self.whisper_model = None


    def download_model(self, model_info: Dict[str, str], model_name: str) -> bool:
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


    def load_models(self) -> bool:
        """Download and load all required models"""
        print("\n📦 Checking models...")
        if not self.download_model(self.en_model_info, self.en_model_info['name']):
            return False
        if not self.download_model(self.pt_model_info, self.pt_model_info['name']):
            return False

        print("\n📦 Loading models...")
        print(f"  English: {self.en_model_info['name']}")
        print(f"  Portuguese: {self.pt_model_info['name']}")
        print(f"  Whisper: {self.whisper_size}")

        try:
            self.vosk_en = Model(self.en_model_info['name'])
            self.vosk_pt = Model(self.pt_model_info['name'])
            self.whisper_model = whisper.load_model(self.whisper_size)
            print("✅ Models loaded.")
            return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False


class AudioManager:
    """Manages audio input stream and device selection."""

    def __init__(self, rate: int = 16000, chunk_ms: int = 30, channels: int = 1,
                 audio_gain: float = 10.0, debug: bool = False):
        self.rate = rate
        self.chunk_ms = chunk_ms
        self.chunk = int(rate * chunk_ms / 1000)
        self.format = pyaudio.paInt16
        self.channels = channels
        self.audio_gain = audio_gain
        self.debug = debug
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.input_device_index = None


    def get_available_devices(self) -> List[Tuple[int, str]]:
        """Get list of available input devices"""
        devices = []
        for i in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                devices.append((i, dev['name']))
        return devices


    def select_device(self, list_only: bool = False) -> bool:
        """Select audio input device"""
        devices = self.get_available_devices()

        if list_only:
            print("🔍 Available input devices:")
            for i, name in devices:
                print(f"  [{i}] {name}")
            return False

        if not devices:
            print("❌ No microphone devices found.")
            return False

        # Device selection
        if self.debug or len(devices) > 1:
            print("🔍 Available input devices:")
            for i, name in devices:
                print(f"  [{i}] {name}")

            if len(devices) > 1:
                print("\n🎯 Enter device number (or press Enter for default): ", end="")
                try:
                    choice = input().strip()
                    if choice:
                        self.input_device_index = int(choice)
                    else:
                        self.input_device_index = devices[0][0]
                except:
                    self.input_device_index = devices[0][0]
            else:
                self.input_device_index = devices[0][0]
        else:
            self.input_device_index = devices[0][0]

        print(f"\n🎙️ Using device index [{self.input_device_index}]")
        if self.debug:
            print(f"🔊 Audio gain set to {self.audio_gain}x")
        return True


    def start_stream(self) -> bool:
        """Start audio input stream"""
        try:
            self.stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk
            )
            return True
        except Exception as e:
            print("❌ Failed to open audio stream:", e)
            return False


    def test_audio(self) -> None:
        """Test audio capture for debugging"""
        if not self.debug or not self.stream:
            return

        print("\n🔊 Testing audio capture for 2 seconds...")
        test_frames = []
        for i in range(int(2 * self.rate / self.chunk)):  # 2 seconds
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
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
                print(f"   Raw levels will be boosted {self.audio_gain}x")
            else:
                print("✅ Audio capture appears to be working")
        else:
            print("❌ No audio data captured!")


    def read_chunk(self) -> np.ndarray:
        """Read and amplify audio chunk"""
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        # Amplify audio
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        audio_array = audio_array * self.audio_gain
        audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
        return audio_array


    def cleanup(self) -> None:
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


class WakeWordDetector:
    """Handles wake word detection using Vosk models."""

    def __init__(self, vosk_en: Model, vosk_pt: Model, rate: int,
                 en_model_name: str, pt_model_name: str, debug: bool = False):
        self.rec_en = KaldiRecognizer(vosk_en, rate)
        self.rec_pt = KaldiRecognizer(vosk_pt, rate)
        self.rec_en.SetWords(True)
        self.rec_pt.SetWords(True)
        self.en_model_name = en_model_name
        self.pt_model_name = pt_model_name
        self.debug = debug


    def detect(self, audio_data: bytes) -> Optional[str]:
        """Check audio for wake words, return language if detected"""
        # Check English model
        if self.rec_en.AcceptWaveform(audio_data):
            result = json.loads(self.rec_en.Result())
            text = result.get("text", "").lower()
            if text:
                if self.debug:
                    print(f"[EN MODEL - {self.en_model_name}] → {text}")
                if "transcribe" in text:
                    print(f"🟢 Wake word detected: 'transcribe' (using {self.en_model_name})")
                    return "en"
        else:
            # Get partial results
            if self.debug:
                partial = json.loads(self.rec_en.PartialResult())
                if partial.get("partial"):
                    print(f"[EN PARTIAL] → {partial['partial']}")

        # Check Portuguese model
        if self.rec_pt.AcceptWaveform(audio_data):
            result = json.loads(self.rec_pt.Result())
            text = result.get("text", "").lower()
            if text:
                if self.debug:
                    print(f"[PT MODEL - {self.pt_model_name}] → {text}")
                if "transcreva" in text:
                    print(f"🟢 Wake word detected: 'transcreva' (using {self.pt_model_name})")
                    return "pt"
        else:
            # Get partial results
            if self.debug:
                partial = json.loads(self.rec_pt.PartialResult())
                if partial.get("partial"):
                    print(f"[PT PARTIAL] → {partial['partial']}")

        return None


    def reset(self) -> None:
        """Reset recognizers for next detection"""
        self.rec_en.Reset()
        self.rec_pt.Reset()


class Transcriber:
    """Handles audio recording and transcription using Whisper."""

    def __init__(self, whisper_model, whisper_size: str, audio_manager: AudioManager,
                 vad_sensitivity: int = 1, max_silence_seconds: float = 1.0, debug: bool = False):
        self.whisper_model = whisper_model
        self.whisper_size = whisper_size
        self.audio_manager = audio_manager
        self.vad = webrtcvad.Vad(vad_sensitivity)
        self.max_silent = int(max_silence_seconds * 1000 / audio_manager.chunk_ms)
        self.debug = debug


    def record_until_silence(self) -> np.ndarray:
        """Record audio until silence is detected"""
        print("🎙️ Recording... Speak now.")
        frames = []
        ring_buffer = collections.deque(maxlen=self.max_silent)

        # Pre-fill buffer with True to avoid immediate stop
        for _ in range(self.max_silent // 2):
            ring_buffer.append(True)

        while True:
            audio_array = self.audio_manager.read_chunk()
            frames.append(audio_array)

            # Debug VAD
            chunk_bytes = audio_array.tobytes()
            is_speech = self.vad.is_speech(chunk_bytes, self.audio_manager.rate)
            ring_buffer.append(is_speech)

            if len(frames) > 10 and not any(ring_buffer):
                if self.debug:
                    print("🔕 Silence detected. Stopping recording.")
                break
            elif self.debug and len(frames) % 33 == 0:  # Every ~1 second
                print(f"⏱️  Recording... {len(frames) * self.audio_manager.chunk_ms / 1000:.1f}s")

        audio_data = np.hstack(frames)
        if self.debug:
            print(f"📊 Recorded {len(audio_data) / self.audio_manager.rate:.1f} seconds of audio")
        return audio_data.astype(np.float32) / 32768.0


    def transcribe(self, audio: np.ndarray, language: str) -> str:
        """Transcribe audio using Whisper"""
        if self.debug:
            print(f"🧠 Transcribing with Whisper ({self.whisper_size} model) in {language.upper()}...")
        else:
            print(f"🧠 Using Whisper {self.whisper_size} model for {language.upper()} transcription...")
        result = self.whisper_model.transcribe(audio, language=language)
        return result["text"].strip()


class SpeechToText:
    """Main speech-to-text application class."""

    def __init__(self, args):
        self.args = args
        self.debug = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')

        # Initialize components
        self.model_manager = ModelManager(args.model_en, args.model_pt, args.whisper_model)
        self.audio_manager = AudioManager(debug=self.debug)
        self.text_typer = TextTyper()
        self.wake_word_detector = None
        self.transcriber = None


    def setup(self) -> bool:
        """Initialize all components"""
        # Select audio device
        if self.args.list_devices:
            self.audio_manager.select_device(list_only=True)
            return False

        if not self.audio_manager.select_device():
            return False

        # Load models
        if not self.model_manager.load_models():
            return False

        # Start audio stream
        if not self.audio_manager.start_stream():
            return False

        # Initialize detectors
        self.wake_word_detector = WakeWordDetector(
            self.model_manager.vosk_en,
            self.model_manager.vosk_pt,
            self.audio_manager.rate,
            self.model_manager.en_model_info['name'],
            self.model_manager.pt_model_info['name'],
            self.debug
        )

        self.transcriber = Transcriber(
            self.model_manager.whisper_model,
            self.model_manager.whisper_size,
            self.audio_manager,
            debug=self.debug
        )

        return True


    def run(self) -> None:
        """Main application loop"""
        print("\n🎤 Say 'transcribe' (EN) or 'transcreva' (PT) to begin...")
        print("📌 The transcribed text will be typed at your cursor position")
        print(f"\n🔧 Active Models:")
        print(f"   • Vosk EN: {self.model_manager.en_model_info['name']} ({self.args.model_en})")
        print(f"   • Vosk PT: {self.model_manager.pt_model_info['name']} ({self.args.model_pt})")
        print(f"   • Whisper: {self.args.whisper_model}")

        # Test audio if in debug mode
        self.audio_manager.test_audio()

        print("\n🎯 Listening for wake words...")
        print("💡 Tip: Click where you want the text to appear before speaking")

        try:
            while True:
                # Detect wake word
                lang = None
                while lang is None:
                    audio_chunk = self.audio_manager.read_chunk()

                    # Debug audio level
                    if self.debug:
                        audio_level = np.abs(audio_chunk).mean()
                        if audio_level > 50:
                            print(f"🔊 Audio level: {audio_level:.0f}")

                    lang = self.wake_word_detector.detect(audio_chunk.tobytes())

                # Record and transcribe
                audio = self.transcriber.record_until_silence()
                text = self.transcriber.transcribe(audio, lang)

                if text:
                    print(f"📝 Transcribed: {text}")

                    # Small delay to ensure user is ready
                    print("✍️  Typing in 0.5 seconds...")
                    time.sleep(0.2)

                    # Type the text where the cursor is
                    self.text_typer.type_text(text)

                    # Audio feedback - system sound
                    if sys.platform == "darwin":
                        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], capture_output=True)

                    print("✅ Text typed at cursor position")
                else:
                    print("⚠️  No text was transcribed")

                # Reset for next detection
                self.wake_word_detector.reset()
                print("\n🎤 Say 'transcribe' (EN) or 'transcreva' (PT) to begin...")

        except KeyboardInterrupt:
            print("🛑 Exiting.")
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
        finally:
            self.audio_manager.cleanup()


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

    app = SpeechToText(args)
    if app.setup():
        app.run()


if __name__ == "__main__":
    main()
