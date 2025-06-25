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
import signal
import threading
import queue
from typing import Dict, Optional, Tuple, List
from vosk import Model, KaldiRecognizer
from pathlib import Path
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


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


class SettingsManager:
    """Manages application settings and preferences."""

    def __init__(self):
        self.settings_dir = Path.home() / '.stt_config'
        self.settings_file = self.settings_dir / 'settings.json'
        self.settings = self._load_settings()


    def _load_settings(self) -> Dict:
        """Load settings from file or return defaults"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading settings: {e}")
        return {}


    def _save_settings(self) -> None:
        """Save current settings to file"""
        try:
            self.settings_dir.mkdir(exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving settings: {e}")


    def get_audio_device(self) -> Optional[Tuple[int, str]]:
        """Get saved audio device preference"""
        if 'audio_device' in self.settings:
            return (self.settings['audio_device']['index'],
                    self.settings['audio_device']['name'])
        return None


    def save_audio_device(self, index: int, name: str) -> None:
        """Save audio device preference"""
        self.settings['audio_device'] = {
            'index': index,
            'name': name
        }
        self._save_settings()


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
                 audio_gain: float = 10.0, debug: bool = False,
                 settings_manager: Optional[SettingsManager] = None):
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
        self.settings_manager = settings_manager
        self._stream_lock = threading.Lock()
        self._last_read_time = time.time()


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

        # Check for saved device preference
        saved_device = None
        if self.settings_manager:
            saved_device = self.settings_manager.get_audio_device()
            if saved_device:
                # Verify saved device still exists
                if any(d[0] == saved_device[0] and d[1] == saved_device[1] for d in devices):
                    self.input_device_index = saved_device[0]
                    print(f"🎙️ Using saved audio device: {saved_device[1]}")
                    print(f"   Device index: [{self.input_device_index}]")
                    if self.debug:
                        print(f"🔊 Audio gain set to {self.audio_gain}x")
                    return True
                else:
                    print(f"⚠️  Saved device '{saved_device[1]}' no longer available")

        # Manual device selection
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

        # Find device name and save preference
        device_name = next((name for idx, name in devices if idx == self.input_device_index), "Unknown")
        if self.settings_manager:
            self.settings_manager.save_audio_device(self.input_device_index, device_name)
            print(f"💾 Saved audio device preference")

        print(f"\n🎙️ Using device: {device_name}")
        print(f"   Device index: [{self.input_device_index}]")
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


    def read_chunk(self) -> Optional[np.ndarray]:
        """Read and amplify audio chunk with timeout protection"""

        # Use a thread to read audio with timeout
        result = [None]
        exception = [None]

        def _read_audio():
            try:
                with self._stream_lock:
                    # First check if stream is still valid
                    if not self.stream:
                        return

                    # Try to check if stream is active (this might fail if device disconnected)
                    try:
                        if not self.stream.is_active():
                            return
                    except:
                        # Stream object is corrupted
                        return

                    # Read with multiple safety measures
                    try:
                        data = self.stream.read(self.chunk, exception_on_overflow=False)
                        result[0] = data
                    except OSError as e:
                        # This catches the PaMacCore error
                        if "err='-50'" in str(e) or "Unknown Error" in str(e):
                            exception[0] = e
                            return
                        raise
            except Exception as e:
                exception[0] = e

        # Run the read in a thread with timeout
        read_thread = threading.Thread(target=_read_audio)
        read_thread.daemon = True
        read_thread.start()

        # Wait for read to complete with timeout
        read_thread.join(timeout=0.5)  # 500ms timeout

        if read_thread.is_alive():
            # Read is stuck, likely due to device disconnection
            print("⚠️  Audio read timeout - device may be disconnected")
            return None

        if exception[0]:
            if self.debug or "PaMacCore" in str(exception[0]):
                print(f"❌ Audio read error: {exception[0]}")
            return None

        if result[0] is None:
            return None

        try:
            # Amplify audio
            audio_array = np.frombuffer(result[0], dtype=np.int16).astype(np.float32)
            audio_array = audio_array * self.audio_gain
            audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
            self._last_read_time = time.time()
            return audio_array
        except Exception as e:
            if self.debug:
                print(f"❌ Audio processing error: {e}")
            return None


    def is_stream_active(self) -> bool:
        """Check if the audio stream is still active"""
        try:
            return self.stream and self.stream.is_active()
        except:
            return False


    def restart_stream(self) -> bool:
        """Restart the audio stream (useful after device disconnection)"""
        try:
            with self._stream_lock:
                # Close existing stream if any
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except:
                        pass  # Ignore errors when closing a dead stream
                    finally:
                        self.stream = None

            # Wait a moment for device to stabilize
            time.sleep(0.5)

            # Re-initialize PyAudio to refresh device list
            self.pa.terminate()
            self.pa = pyaudio.PyAudio()

            # Check if the previously selected device is still available
            devices = self.get_available_devices()
            device_available = any(d[0] == self.input_device_index for d in devices)

            if not device_available:
                print(f"⚠️  Previous audio device (index {self.input_device_index}) no longer available")
                print("🔍 Attempting to select a new device...")

                # Try to find a device with similar name from saved preference
                saved_device = self.settings_manager.get_audio_device() if self.settings_manager else None
                if saved_device:
                    for idx, name in devices:
                        if saved_device[1] in name or name in saved_device[1]:
                            self.input_device_index = idx
                            print(f"🎙️ Found similar device: {name}")
                            break
                    else:
                        # No similar device found, use default
                        if devices:
                            self.input_device_index = devices[0][0]
                            print(f"🎙️ Using default device: {devices[0][1]}")
                        else:
                            print("❌ No audio devices available")
                            return False
                else:
                    # No saved preference, use default
                    if devices:
                        self.input_device_index = devices[0][0]
                        print(f"🎙️ Using default device: {devices[0][1]}")
                    else:
                        print("❌ No audio devices available")
                        return False

            # Start new stream
            return self.start_stream()

        except Exception as e:
            print(f"❌ Failed to restart audio stream: {e}")
            return False


    def cleanup(self) -> None:
        """Clean up audio resources"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass  # Ignore errors if stream is already dead
        self.pa.terminate()


class WakeWordDetector:
    """Handles wake word detection using Vosk models."""

    def __init__(self, vosk_en: Model, vosk_pt: Model, rate: int,
                 en_model_name: str, pt_model_name: str, 
                 wake_word_en: str = "transcribe", wake_word_pt: str = "transcreva",
                 debug: bool = False):
        self.rec_en = KaldiRecognizer(vosk_en, rate)
        self.rec_pt = KaldiRecognizer(vosk_pt, rate)
        self.rec_en.SetWords(True)
        self.rec_pt.SetWords(True)
        self.en_model_name = en_model_name
        self.pt_model_name = pt_model_name
        self.wake_word_en = wake_word_en
        self.wake_word_pt = wake_word_pt
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
                # More flexible wake word matching
                if self.wake_word_en in text:
                    print(f"🟢 Wake word detected: '{self.wake_word_en}' in '{text}' (using {self.en_model_name})")
                    return "en"
                elif self.debug and text.strip():
                    print(f"[EN NO MATCH] Looking for '{self.wake_word_en}' in '{text}'")
        else:
            # Get partial results
            if self.debug:
                partial = json.loads(self.rec_en.PartialResult())
                if partial.get("partial"):
                    partial_text = partial['partial'].lower()
                    print(f"[EN PARTIAL] → {partial_text}")
                    # Check if wake word is in partial result
                    if self.wake_word_en in partial_text:
                        print(f"[EN PARTIAL MATCH] Found '{self.wake_word_en}' in partial")

        # Check Portuguese model
        if self.rec_pt.AcceptWaveform(audio_data):
            result = json.loads(self.rec_pt.Result())
            text = result.get("text", "").lower()
            if text:
                if self.debug:
                    print(f"[PT MODEL - {self.pt_model_name}] → {text}")
                if self.wake_word_pt in text:
                    print(f"🟢 Wake word detected: '{self.wake_word_pt}' in '{text}' (using {self.pt_model_name})")
                    return "pt"
                elif self.debug and text.strip():
                    print(f"[PT NO MATCH] Looking for '{self.wake_word_pt}' in '{text}'")
        else:
            # Get partial results
            if self.debug:
                partial = json.loads(self.rec_pt.PartialResult())
                if partial.get("partial"):
                    partial_text = partial['partial'].lower()
                    print(f"[PT PARTIAL] → {partial_text}")
                    # Check if wake word is in partial result
                    if self.wake_word_pt in partial_text:
                        print(f"[PT PARTIAL MATCH] Found '{self.wake_word_pt}' in partial")

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


    def record_until_silence(self) -> Optional[np.ndarray]:
        """Record audio until silence is detected"""
        print("🎙️ Recording... Speak now.")
        frames = []
        ring_buffer = collections.deque(maxlen=self.max_silent)

        # Pre-fill buffer with True to avoid immediate stop
        for _ in range(self.max_silent // 2):
            ring_buffer.append(True)

        while True:
            audio_array = self.audio_manager.read_chunk()

            # Handle audio disconnection during recording
            if audio_array is None:
                print("\n⚠️  Audio device disconnected during recording")
                if frames:
                    print("🔄 Attempting to save partial recording...")
                    try:
                        audio_data = np.hstack(frames)
                        return audio_data.astype(np.float32) / 32768.0
                    except:
                        pass
                return None

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


class KeyboardManager:
    """Manages global keyboard shortcuts for triggering transcription."""
    
    def __init__(self, shortcut: str, trigger_queue: queue.Queue, debug: bool = False):
        self.shortcut = shortcut
        self.trigger_queue = trigger_queue
        self.debug = debug
        self.listener = None
        self.current_keys = set()
        self.shortcut_keys = self._parse_shortcut(shortcut)
        
    def _parse_shortcut(self, shortcut: str) -> set:
        """Parse shortcut string into a set of keys"""
        keys = set()
        parts = shortcut.lower().split('+')
        
        for part in parts:
            part = part.strip()
            if part in ['cmd', 'command']:
                keys.add('cmd')
            elif part in ['ctrl', 'control']:
                keys.add('ctrl')
            elif part in ['alt', 'option']:
                keys.add('alt')
            elif part in ['shift']:
                keys.add('shift')
            else:
                keys.add(part)
        
        return keys
    
    def _on_press(self, key):
        """Handle key press events"""
        try:
            # Get the key name
            if hasattr(key, 'char') and key.char:
                key_name = key.char.lower()
            elif hasattr(key, 'name'):
                key_name = key.name.lower()
            else:
                return
            
            # Map special keys
            if key == keyboard.Key.cmd:
                key_name = 'cmd'
            elif key == keyboard.Key.ctrl:
                key_name = 'ctrl'
            elif key == keyboard.Key.alt:
                key_name = 'alt'
            elif key == keyboard.Key.shift:
                key_name = 'shift'
            
            self.current_keys.add(key_name)
            
            # Check if shortcut is pressed
            if self.shortcut_keys.issubset(self.current_keys):
                if self.debug:
                    print(f"🎹 Keyboard shortcut triggered: {self.shortcut}")
                self.trigger_queue.put('keyboard')
                
        except Exception as e:
            if self.debug:
                print(f"❌ Keyboard error: {e}")
    
    def _on_release(self, key):
        """Handle key release events"""
        try:
            # Get the key name
            if hasattr(key, 'char') and key.char:
                key_name = key.char.lower()
            elif hasattr(key, 'name'):
                key_name = key.name.lower()
            else:
                return
            
            # Map special keys
            if key == keyboard.Key.cmd:
                key_name = 'cmd'
            elif key == keyboard.Key.ctrl:
                key_name = 'ctrl'
            elif key == keyboard.Key.alt:
                key_name = 'alt'
            elif key == keyboard.Key.shift:
                key_name = 'shift'
            
            self.current_keys.discard(key_name)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Keyboard error: {e}")
    
    def start(self):
        """Start listening for keyboard events"""
        if not PYNPUT_AVAILABLE:
            print("⚠️  pynput not installed. Keyboard shortcuts disabled.")
            print("   Install with: pip install pynput")
            return False
        
        try:
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.listener.start()
            print(f"⌨️  Keyboard shortcut enabled: {self.shortcut}")
            return True
        except Exception as e:
            print(f"❌ Failed to start keyboard listener: {e}")
            print("💡 On macOS, grant accessibility permissions in System Preferences > Security & Privacy > Privacy > Accessibility")
            return False
    
    def stop(self):
        """Stop the keyboard listener"""
        if self.listener:
            self.listener.stop()


class SpeechToText:
    """Main speech-to-text application class."""

    def __init__(self, args):
        self.args = args
        self.debug = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')
        self.wake_word_en = args.wake_word_en.lower()
        self.wake_word_pt = args.wake_word_pt.lower()

        # Initialize components
        self.settings_manager = SettingsManager()
        self.model_manager = ModelManager(args.model_en, args.model_pt, args.whisper_model)
        self.audio_manager = AudioManager(debug=self.debug, settings_manager=self.settings_manager)
        self.text_typer = TextTyper()
        self.wake_word_detector = None
        self.transcriber = None
        self.keyboard_manager = None
        self.trigger_queue = queue.Queue()


    def setup(self) -> bool:
        """Initialize all components"""
        # Handle reset audio device option
        if self.args.reset_audio_device:
            if 'audio_device' in self.settings_manager.settings:
                del self.settings_manager.settings['audio_device']
                self.settings_manager._save_settings()
                print("✅ Audio device preference reset")
            else:
                print("ℹ️  No saved audio device preference to reset")
            return False

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
        print(f"🔧 Initializing wake word detector with debug={self.debug}")
        self.wake_word_detector = WakeWordDetector(
            self.model_manager.vosk_en,
            self.model_manager.vosk_pt,
            self.audio_manager.rate,
            self.model_manager.en_model_info['name'],
            self.model_manager.pt_model_info['name'],
            self.wake_word_en,
            self.wake_word_pt,
            self.debug
        )
        print(f"✅ Wake word detector initialized")

        self.transcriber = Transcriber(
            self.model_manager.whisper_model,
            self.model_manager.whisper_size,
            self.audio_manager,
            debug=self.debug
        )
        
        # Initialize keyboard shortcut if specified
        if self.args.keyboard_shortcut:
            self.keyboard_manager = KeyboardManager(
                self.args.keyboard_shortcut,
                self.trigger_queue,
                self.debug
            )
            if not self.keyboard_manager.start():
                print("⚠️  Continuing without keyboard shortcuts")
                self.keyboard_manager = None

        return True


    def run(self) -> None:
        """Main application loop"""
        print(f"\n🎤 Say '{self.wake_word_en}' (EN) or '{self.wake_word_pt}' (PT) to begin...")
        if self.keyboard_manager:
            print(f"⌨️  Or press {self.args.keyboard_shortcut} to start transcription")
        print("📌 The transcribed text will be typed at your cursor position")
        print(f"\n🔧 Active Models:")
        print(f"   • Vosk EN: {self.model_manager.en_model_info['name']} ({self.args.model_en})")
        print(f"   • Vosk PT: {self.model_manager.pt_model_info['name']} ({self.args.model_pt})")
        print(f"   • Whisper: {self.args.whisper_model}")
        print(f"\n🎯 Wake Words:")
        print(f"   • English: '{self.wake_word_en}'")
        print(f"   • Portuguese: '{self.wake_word_pt}'")

        # Test audio if in debug mode
        self.audio_manager.test_audio()

        print("\n🎯 Listening for wake words...")
        print("💡 Tip: Click where you want the text to appear before speaking")

        try:
            while True:
                # Detect wake word or keyboard trigger
                lang = None
                triggered_by_keyboard = False
                
                while lang is None and not triggered_by_keyboard:
                    # Check for keyboard trigger (non-blocking)
                    try:
                        trigger = self.trigger_queue.get_nowait()
                        if trigger == 'keyboard':
                            triggered_by_keyboard = True
                            # Default to English for keyboard trigger
                            lang = 'en'
                            break
                    except queue.Empty:
                        pass
                    audio_chunk = self.audio_manager.read_chunk()

                    # Handle audio device disconnection
                    if audio_chunk is None:
                        print("⚠️  Audio device disconnected or error occurred")
                        print("🔄 Attempting to reconnect...")

                        # Attempt to restart the audio stream
                        reconnect_attempts = 0
                        while reconnect_attempts < 5:
                            if self.audio_manager.restart_stream():
                                print("✅ Audio stream reconnected successfully")
                                print(f"🎤 Say '{self.wake_word_en}' (EN) or '{self.wake_word_pt}' (PT) to begin...")
                                break
                            else:
                                reconnect_attempts += 1
                                if reconnect_attempts < 5:
                                    print(f"🔄 Retry {reconnect_attempts}/5 in 2 seconds...")
                                    time.sleep(2)
                        else:
                            print("❌ Failed to reconnect after 5 attempts")
                            print("💡 Please check your audio devices and restart the application")
                            return

                        # Continue to next iteration after reconnection
                        continue

                    # Debug audio level
                    if self.debug:
                        audio_level = np.abs(audio_chunk).mean()
                        if audio_level > 50:
                            print(f"🔊 Audio level: {audio_level:.0f}")

                    if not triggered_by_keyboard:
                        lang = self.wake_word_detector.detect(audio_chunk.tobytes())

                # Play sound alert when transcription starts
                if sys.platform == "darwin":
                    subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], capture_output=True)
                
                # Record and transcribe
                audio = self.transcriber.record_until_silence()

                # Handle recording failure
                if audio is None:
                    print("⚠️  Recording failed due to audio disconnection")
                    # The main loop will handle reconnection on next iteration
                    continue

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
                print(f"\n🎤 Say '{self.wake_word_en}' (EN) or '{self.wake_word_pt}' (PT) to begin...")
                if self.keyboard_manager:
                    print(f"⌨️  Or press {self.args.keyboard_shortcut} to start transcription")

        except KeyboardInterrupt:
            print("🛑 Exiting.")
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
        finally:
            if self.keyboard_manager:
                self.keyboard_manager.stop()
            self.audio_manager.cleanup()


def main():
    # Set up signal handler to prevent terminal freeze
    def signal_handler(sig, frame):
        print("\n🛑 Interrupted. Cleaning up...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Ignore SIGPIPE to prevent crashes on broken audio pipes
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description='Speech-to-text with wake word activation')
    parser.add_argument('--model-en', choices=['small', 'large'], default='small',
                        help='English model size (default: small)')
    parser.add_argument('--model-pt', choices=['small', 'large'], default='small',
                        help='Portuguese model size (default: small)')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        default='base', help='Whisper model size (default: base)')
    parser.add_argument('--wake-word-en', default='transcribe',
                        help='English wake word (default: transcribe)')
    parser.add_argument('--wake-word-pt', default='transcreva',
                        help='Portuguese wake word (default: transcreva)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--reset-audio-device', action='store_true',
                        help='Reset saved audio device preference')
    parser.add_argument('--keyboard-shortcut', default=None,
                        help='Global keyboard shortcut to trigger transcription (e.g., "cmd+shift+t")')

    args = parser.parse_args()

    app = SpeechToText(args)
    if app.setup():
        app.run()


if __name__ == "__main__":
    main()
