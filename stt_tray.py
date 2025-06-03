#!/usr/bin/env python3
"""
Speech-to-Text System Tray Application
"""

import pystray
from PIL import Image, ImageDraw
import threading
import queue
import sys
import os
import time
import subprocess
import argparse
from pathlib import Path
import numpy as np

# Import the existing STT components
from stt import (
    SpeechToText, SettingsManager, ModelManager, AudioManager, 
    WakeWordDetector, Transcriber, TextTyper, MODELS
)


class STTTrayApp:
    """System tray wrapper for Speech-to-Text application"""

    def __init__(self, args):
        self.args = args
        self.command_queue = queue.Queue()
        self.status_queue = queue.Queue()

        # State management
        self.is_listening = True
        self.is_recording = False
        self.is_processing = False
        self.current_status = "Ready"

        # Initialize settings manager to get saved preferences
        self.settings_manager = SettingsManager()

        # Speech recognition thread
        self.stt_thread = None
        self.stt_app = None
        self.running = False

        # Debug mode
        self.debug = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')
        if self.debug:
            print("🐛 Debug mode enabled")

        # Create system tray icon
        self.icon = None
        self.create_icon()

    def create_icon_image(self, color='black'):
        """Create a simple microphone icon"""
        # Create an image with a microphone shape
        width = 64
        height = 64
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Microphone colors based on state
        colors = {
            'black': (0, 0, 0, 255),
            'gray': (128, 128, 128, 255),
            'red': (255, 0, 0, 255),
            'green': (0, 200, 0, 255),
            'orange': (255, 165, 0, 255)
        }

        mic_color = colors.get(color, colors['black'])

        # Draw microphone body
        draw.ellipse([20, 10, 44, 40], fill=mic_color, outline=mic_color)
        draw.rectangle([20, 25, 44, 35], fill=mic_color, outline=mic_color)

        # Draw microphone stand
        draw.rectangle([30, 35, 34, 50], fill=mic_color, outline=mic_color)
        draw.rectangle([24, 50, 40, 54], fill=mic_color, outline=mic_color)

        # Draw arc for microphone holder
        draw.arc([15, 30, 49, 50], start=30, end=150, fill=mic_color, width=3)

        return image

    def create_icon(self):
        """Create the system tray icon"""
        self.icon = pystray.Icon(
            "speech-to-text",
            self.create_icon_image('black'),
            "Speech to Text - Ready",
            menu=self.create_menu()
        )

    def create_menu(self):
        """Create the system tray menu"""
        return pystray.Menu(
            pystray.MenuItem(
                "Listening",
                self.toggle_listening,
                checked=lambda item: self.is_listening
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Wake Words",
                pystray.Menu(
                    pystray.MenuItem(
                        lambda text: f"English: '{self.get_wake_word('en')}'",
                        self.change_wake_word_en
                    ),
                    pystray.MenuItem(
                        lambda text: f"Portuguese: '{self.get_wake_word('pt')}'",
                        self.change_wake_word_pt
                    )
                )
            ),
            pystray.MenuItem(
                "Audio Device",
                pystray.Menu(lambda: self.create_audio_device_menu())
            ),
            pystray.MenuItem(
                "Models",
                pystray.Menu(
                    pystray.MenuItem(
                        "English",
                        pystray.Menu(
                            pystray.MenuItem(
                                "Small",
                                lambda: self.change_model('en', 'small'),
                                checked=lambda item: self.args.model_en == 'small'
                            ),
                            pystray.MenuItem(
                                "Large",
                                lambda: self.change_model('en', 'large'),
                                checked=lambda item: self.args.model_en == 'large'
                            )
                        )
                    ),
                    pystray.MenuItem(
                        "Portuguese",
                        pystray.Menu(
                            pystray.MenuItem(
                                "Small",
                                lambda: self.change_model('pt', 'small'),
                                checked=lambda item: self.args.model_pt == 'small'
                            ),
                            pystray.MenuItem(
                                "Large",
                                lambda: self.change_model('pt', 'large'),
                                checked=lambda item: self.args.model_pt == 'large'
                            )
                        )
                    ),
                    pystray.MenuItem(
                        "Whisper",
                        pystray.Menu(
                            pystray.MenuItem(
                                "Tiny",
                                lambda: self.change_whisper_model('tiny'),
                                checked=lambda item: self.args.whisper_model == 'tiny'
                            ),
                            pystray.MenuItem(
                                "Base",
                                lambda: self.change_whisper_model('base'),
                                checked=lambda item: self.args.whisper_model == 'base'
                            ),
                            pystray.MenuItem(
                                "Small",
                                lambda: self.change_whisper_model('small'),
                                checked=lambda item: self.args.whisper_model == 'small'
                            ),
                            pystray.MenuItem(
                                "Medium",
                                lambda: self.change_whisper_model('medium'),
                                checked=lambda item: self.args.whisper_model == 'medium'
                            ),
                            pystray.MenuItem(
                                "Large",
                                lambda: self.change_whisper_model('large'),
                                checked=lambda item: self.args.whisper_model == 'large'
                            )
                        )
                    )
                )
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Logs", self.show_logs),
            pystray.MenuItem("Quit", self.quit_app)
        )

    def create_audio_device_menu(self):
        """Dynamically create audio device menu items"""
        menu_items = []

        # Get current audio devices
        if self.stt_app and self.stt_app.audio_manager:
            devices = self.stt_app.audio_manager.get_available_devices()
            current_device = self.stt_app.audio_manager.input_device_index

            for idx, name in devices:
                menu_items.append(
                    pystray.MenuItem(
                        name,
                        lambda _, device_idx=idx: self.change_audio_device(device_idx),
                        checked=lambda item, device_idx=idx: current_device == device_idx
                    )
                )
        else:
            menu_items.append(pystray.MenuItem("Loading...", lambda: None))

        return menu_items

    def get_wake_word(self, lang):
        """Get current wake word for language"""
        if lang == 'en':
            return self.args.wake_word_en
        else:
            return self.args.wake_word_pt

    def toggle_listening(self, icon, item):
        """Toggle listening on/off"""
        self.is_listening = not self.is_listening
        self.command_queue.put({'action': 'toggle_listening', 'value': self.is_listening})
        self.update_icon_state()

    def change_wake_word_en(self, icon, item):
        """Change English wake word"""
        self.change_wake_word('en')

    def change_wake_word_pt(self, icon, item):
        """Change Portuguese wake word"""
        self.change_wake_word('pt')

    def change_wake_word(self, lang):
        """Change wake word for specified language"""
        current_word = self.get_wake_word(lang)
        lang_name = "English" if lang == 'en' else "Portuguese"

        # For macOS, use osascript to get input
        if sys.platform == "darwin":
            try:
                script = f'''
                display dialog "Enter new {lang_name} wake word (current: '{current_word}'):" ¬
                    default answer "{current_word}" ¬
                    with title "Change Wake Word" ¬
                    buttons {{"Cancel", "OK"}} default button "OK"
                '''
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    # Extract the entered text
                    output = result.stdout.strip()
                    if "text returned:" in output:
                        new_word = output.split("text returned:")[-1].strip()
                        if new_word:
                            new_word = new_word.lower()
                            if lang == 'en':
                                self.args.wake_word_en = new_word
                            else:
                                self.args.wake_word_pt = new_word

                            self.command_queue.put({
                                'action': 'change_wake_word',
                                'lang': lang,
                                'word': new_word
                            })

                            self.show_notification(
                                "Wake Word Changed",
                                f"{lang_name} wake word changed to: '{new_word}'"
                            )
            except Exception as e:
                print(f"Error showing dialog: {e}")
                self.show_notification(
                    "Error",
                    "Could not show input dialog. Check terminal for instructions."
                )
                print(f"\nTo change {lang_name} wake word, restart with: --wake-word-{lang[:2]} 'newword'")
        else:
            # For other platforms, just show notification
            self.show_notification(
                "Wake Word Change",
                f"To change {lang_name} wake word, restart with: --wake-word-{lang[:2]} 'newword'"
            )

    def change_audio_device(self, device_idx):
        """Change audio input device"""
        self.command_queue.put({
            'action': 'change_device',
            'device_index': device_idx
        })

    def change_model(self, lang, size):
        """Change Vosk model size"""
        if lang == 'en':
            old_size = self.args.model_en
            self.args.model_en = size
        else:
            old_size = self.args.model_pt
            self.args.model_pt = size

        if old_size != size:
            self.command_queue.put({
                'action': 'change_model',
                'type': f'vosk_{lang}',
                'size': size
            })

            self.show_notification(
                f"Model Change",
                f"Changing {lang.upper()} model to {size}. This may take a moment..."
            )

    def change_whisper_model(self, size):
        """Change Whisper model size"""
        old_size = self.args.whisper_model
        self.args.whisper_model = size

        if old_size != size:
            self.command_queue.put({
                'action': 'change_model',
                'type': 'whisper',
                'size': size
            })

            self.show_notification(
                "Model Change",
                f"Changing Whisper model to {size}. This may take a moment..."
            )

    def show_logs(self, icon, item):
        """Show application logs"""
        status_info = (
            f"Status: {self.current_status}\n"
                f"Listening: {'Yes' if self.is_listening else 'No'}\n"
                f"Wake Words: EN:'{self.args.wake_word_en}', PT:'{self.args.wake_word_pt}'\n"
                f"Models: EN:{self.args.model_en}, PT:{self.args.model_pt}, Whisper:{self.args.whisper_model}"
        )

        if sys.platform == "darwin":
            # On macOS, use osascript to show a dialog
            script = f'''
            display dialog "{status_info}" ¬
                with title "Speech to Text Status" ¬
                buttons {{"OK"}} default button "OK"
            '''
            subprocess.run(["osascript", "-e", script], capture_output=True)
        else:
            # For other platforms, just print to console and show notification
            print(f"\n{status_info}\n")
            self.show_notification("Speech to Text Status", "Check terminal for details")

    def quit_app(self, icon, item):
        """Quit the application"""
        self.running = False
        self.command_queue.put({'action': 'quit'})
        self.icon.stop()

    def update_icon_state(self):
        """Update icon based on current state"""
        if not self.is_listening:
            color = 'gray'
            status = "Paused"
        elif self.is_recording:
            color = 'red'
            status = "Recording"
        elif self.is_processing:
            color = 'orange'
            status = "Processing"
        else:
            color = 'green'
            status = "Listening"

        self.current_status = status
        self.icon.icon = self.create_icon_image(color)
        self.icon.title = f"Speech to Text - {status}"

    def show_notification(self, title, message):
        """Show system notification"""
        if sys.platform == "darwin":
            subprocess.run([
                "osascript", "-e",
                f'display notification "{message}" with title "{title}"'
            ])
        else:
            # For other platforms, pystray can show notifications
            self.icon.notify(message, title)

    def stt_worker(self):
        """Worker thread for speech-to-text processing"""
        try:
            # Initialize the STT app
            self.stt_app = SpeechToText(self.args)

            if not self.stt_app.setup():
                self.status_queue.put({'error': 'Failed to initialize STT'})
                return

            # Custom run loop that checks command queue
            self.run_stt_loop()

        except Exception as e:
            self.status_queue.put({'error': str(e)})

    def run_stt_loop(self):
        """Modified STT run loop that integrates with system tray"""
        print("\n" + "="*60)
        print("🎤 SPEECH TO TEXT - SYSTEM TRAY MODE")
        print("="*60)
        print(f"📝 Wake words: EN:'{self.args.wake_word_en}', PT:'{self.args.wake_word_pt}'")
        print(f"🎯 Models: EN:{self.args.model_en}, PT:{self.args.model_pt}, Whisper:{self.args.whisper_model}")
        print("💡 Right-click the tray icon to access controls")
        print("🐛 Debug mode:", "ON" if self.debug else "OFF (set DEBUG=1 to enable)")
        print("="*60 + "\n")
        
        self.show_notification(
            "Speech to Text Ready",
            f"Say '{self.args.wake_word_en}' or '{self.args.wake_word_pt}' to begin"
        )
        
        # Main processing loop
        print("🎯 Listening for wake words...")
        while self.running:
            # Check for commands
            try:
                command = self.command_queue.get_nowait()
                self.handle_command(command)
            except queue.Empty:
                pass
            
            # Only process audio if listening is enabled
            if not self.is_listening:
                time.sleep(0.1)
                continue
            
            try:
                # Detect wake word
                audio_chunk = self.stt_app.audio_manager.read_chunk()
                
                if audio_chunk is None:
                    print("⚠️  Audio read returned None - attempting to reconnect...")
                    # Handle disconnection
                    if self.stt_app.audio_manager.restart_stream():
                        print("✅ Audio stream reconnected")
                        self.show_notification(
                            "Audio Reconnected",
                            "Audio device reconnected successfully"
                        )
                    else:
                        print("❌ Failed to reconnect audio stream")
                        time.sleep(2)
                    continue
                
                # Debug audio level
                if self.debug:
                    audio_level = np.abs(audio_chunk).mean()
                    if audio_level > 50:
                        print(f"🔊 Audio level: {audio_level:.0f}")
                
                # Check for wake word
                lang = self.stt_app.wake_word_detector.detect(audio_chunk.tobytes())
                
                if lang:
                    # Wake word detected
                    print(f"\n🟢 Wake word detected! Language: {lang}")
                    self.is_recording = True
                    self.update_icon_state()
                    
                    # Play start sound
                    if sys.platform == "darwin":
                        print("🔔 Playing start sound...")
                        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], capture_output=True)
                    
                    # Record audio
                    print("🎙️ Recording... Speak now.")
                    audio = self.stt_app.transcriber.record_until_silence()
                    
                    if audio is None:
                        print("⚠️  Recording failed")
                        self.is_recording = False
                        self.update_icon_state()
                        continue
                    
                    # Process transcription
                    self.is_recording = False
                    self.is_processing = True
                    self.update_icon_state()
                    
                    print(f"🧠 Transcribing with Whisper ({self.args.whisper_model} model)...")
                    text = self.stt_app.transcriber.transcribe(audio, lang)
                    
                    if text:
                        print(f"📝 Transcribed: {text}")
                        
                        # Type the text
                        print("✍️  Typing text...")
                        self.stt_app.text_typer.type_text(text)
                        
                        # Play completion sound
                        if sys.platform == "darwin":
                            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], capture_output=True)
                        
                        print("✅ Text typed at cursor position")
                        
                        self.show_notification(
                            "Transcription Complete",
                            text[:100] + "..." if len(text) > 100 else text
                        )
                    else:
                        print("⚠️  No text was transcribed")
                    
                    # Reset state
                    self.is_processing = False
                    self.update_icon_state()
                    self.stt_app.wake_word_detector.reset()
                    
                    print("\n🎯 Listening for wake words...")
                    
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)
        
        print("\n🛑 Shutting down...")
        # Cleanup
        self.stt_app.audio_manager.cleanup()

    def handle_command(self, command):
        """Handle commands from the UI thread"""
        action = command.get('action')

        if action == 'quit':
            self.running = False
        elif action == 'toggle_listening':
            self.is_listening = command.get('value', False)
            status = "Resumed" if self.is_listening else "Paused"
            self.show_notification("Speech to Text", f"Listening {status}")
        elif action == 'change_wake_word':
            lang = command.get('lang')
            word = command.get('word')
            if self.stt_app and self.stt_app.wake_word_detector:
                if lang == 'en':
                    self.stt_app.wake_word_en = word
                    self.stt_app.wake_word_detector.wake_word_en = word
                else:
                    self.stt_app.wake_word_pt = word
                    self.stt_app.wake_word_detector.wake_word_pt = word
        elif action == 'change_device':
            device_idx = command.get('device_index')
            if self.stt_app and self.stt_app.audio_manager:
                self.stt_app.audio_manager.input_device_index = device_idx
                if self.stt_app.audio_manager.restart_stream():
                    self.show_notification(
                        "Audio Device Changed",
                        f"Switched to device index {device_idx}"
                    )
        elif action == 'change_model':
            # Model changes require restart - for now just notify
            self.show_notification(
                "Model Change",
                "Please restart the application for model changes to take effect"
            )

    def run(self):
        """Run the system tray application"""
        self.running = True

        # Start STT worker thread
        self.stt_thread = threading.Thread(target=self.stt_worker, daemon=True)
        self.stt_thread.start()

        # Wait a moment for initialization
        time.sleep(1)

        # Update icon to show we're ready
        self.update_icon_state()

        # Run the system tray (this blocks until quit)
        self.icon.run()


def main():
    # Set up signal handler
    import signal
    def signal_handler(sig, frame):
        print("\n🛑 Interrupted. Cleaning up...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Speech-to-text with system tray')
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
    parser.add_argument('--no-tray', action='store_true',
                        help='Run without system tray (classic mode)')

    args = parser.parse_args()

    # Handle special arguments
    if args.list_devices:
        print("\n🔍 Listing available audio devices...")
        from stt import AudioManager
        audio_manager = AudioManager()
        devices = audio_manager.get_available_devices()
        print("\nAvailable input devices:")
        for idx, name in devices:
            print(f"  [{idx}] {name}")
        audio_manager.cleanup()
        return
    
    if args.reset_audio_device:
        print("\n🔄 Resetting audio device preference...")
        from stt import SettingsManager
        settings_manager = SettingsManager()
        if 'audio_device' in settings_manager.settings:
            del settings_manager.settings['audio_device']
            settings_manager._save_settings()
            print("✅ Audio device preference reset")
        else:
            print("ℹ️  No saved audio device preference to reset")
        return

    if args.no_tray:
        # Run classic mode
        from stt import main as stt_main
        stt_main()
    else:
        # Run with system tray
        app = STTTrayApp(args)
        app.run()


if __name__ == "__main__":
    main()
