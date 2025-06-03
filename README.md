# Speech-to-Text Auto-Typer

A real-time speech-to-text application that listens for wake words and automatically types transcribed text at your cursor position. Supports both English and Portuguese languages.

## Features

- **System Tray Interface**: Control everything from the system tray - no terminal needed
- **Wake Word Activation**: Say "transcribe" (English) or "transcreva" (Portuguese) to start recording
- **Customizable Wake Words**: Set your own wake words via command-line arguments or system tray
- **Auto-Typing**: Transcribed text is automatically typed at your current cursor position
- **Multi-Language Support**: Works with both English and Portuguese
- **Real-Time Processing**: Uses VAD (Voice Activity Detection) to detect when you stop speaking
- **Dual Engine**: Combines Vosk for wake word detection and Whisper for accurate transcription
- **Audio Amplification**: Automatically amplifies low audio levels for better recognition
- **Audio Device Recovery**: Automatically reconnects when audio devices (like headphones) disconnect
- **Sound Alerts**: Plays sounds when transcription starts and completes
- **Automatic Model Download**: Models are downloaded automatically on first use
- **Flexible Model Selection**: Choose between small (fast) and large (accurate) models
- **Persistent Settings**: Remembers your preferred audio device
- **Visual Status**: System tray icon changes color to show current state
- **Comprehensive Debugging**: Extensive logging and diagnostic tools for troubleshooting

## Requirements

- Python 3.7+
- macOS (for auto-typing functionality) or any OS with pyautogui installed
- Microphone access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speech-to-text
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pyaudio numpy whisper webrtcvad vosk pystray pillow
```

For non-macOS systems, also install:
```bash
pip install pyautogui
```

**Note for macOS users**: The system tray uses native macOS dialogs for input. No additional dependencies needed.

The models will be downloaded automatically when you first run the application.

## Usage

### Basic Usage

Run with system tray interface (recommended):
```bash
python stt_tray.py
```

Or run in classic terminal mode:
```bash
python stt.py
# or
python stt_tray.py --no-tray
```

### System Tray Mode

When running with `stt_tray.py`, you get:
- **System tray icon** that shows current status (green=listening, gray=paused, red=recording, orange=processing)
- **Right-click menu** with all controls:
  - Toggle listening on/off
  - Change wake words on the fly
  - Switch audio devices without restarting
  - Change models (requires restart)
  - View current settings
- **Notifications** for important events
- **No terminal window** needed after startup

### Advanced Usage

```bash
# Use large models for better accuracy
python stt_tray.py --model-en large --model-pt large --whisper-model medium

# List available audio devices (terminal mode)
python stt.py --list-devices

# Use specific models with custom wake words
python stt_tray.py --model-en small --model-pt large --wake-word-en "record"

# Classic terminal mode with all options
python stt.py --wake-word-en "start" --wake-word-pt "começar"

# Reset audio device preference
python stt.py --reset-audio-device
```

**Command-line Options:**
- `--model-en {small,large}`: English Vosk model size (default: small)
- `--model-pt {small,large}`: Portuguese Vosk model size (default: small)
- `--whisper-model {tiny,base,small,medium,large}`: Whisper model size (default: base)
- `--wake-word-en WORD`: Custom English wake word (default: transcribe)
- `--wake-word-pt WORD`: Custom Portuguese wake word (default: transcreva)
- `--list-devices`: List available audio devices and exit
- `--reset-audio-device`: Reset saved audio device preference

### Usage Steps

1. Run the application with your desired model configuration

2. Select your microphone device (if multiple are available)

3. Click where you want the text to appear (text editor, browser, etc.)

4. Say "transcribe" (English) or "transcreva" (Portuguese)

5. Start speaking - the app will record until you pause

6. The transcribed text will be automatically typed at your cursor position

### Debug Mode

For troubleshooting audio issues:
```bash
DEBUG=1 python stt.py
```

This will show:
- Audio levels
- Partial recognition results
- Detailed processing information

### Testing & Diagnostic Tools

The project includes several diagnostic tools to help troubleshoot issues:

```bash
# Test basic audio capture
python test_audio.py

# Test audio with visual level meter
python test_tray.py

# Test Vosk wake word detection
python test_vosk.py

# Test wake word detection with detailed output
DEBUG=1 python test_wake_word.py

# Test model loading
python test_models.py

# Test core STT functionality without UI
python test_stt_core.py

# Show step-by-step debugging guide
python debug_guide.py
```

## Model Information

### Available Models

The application automatically downloads the models you select. Here are the available options:

**Vosk Models (Wake Word Detection):**
- English:
  - `small`: 40 MB - Fast, good for most use cases
  - `large`: 2.3 GB (Gigaspeech) - Best accuracy
- Portuguese:
  - `small`: 31 MB - Fast, basic accuracy
  - `large`: 1.6 GB - Much better accuracy

**Whisper Models (Transcription):**
- `tiny`: 40 MB - Fastest, lowest accuracy
- `base`: 150 MB - Good balance (default)
- `small`: 500 MB - Better accuracy
- `medium`: 1.5 GB - High accuracy
- `large`: 3 GB - Best accuracy

### Model Selection Guide

- **For speed**: Use small Vosk models + tiny/base Whisper
- **For accuracy**: Use large Vosk models + medium/large Whisper
- **Balanced**: Use small Vosk models + small/medium Whisper

## Troubleshooting

### System Tray Issues

If the system tray version isn't working properly:

1. **Enable debug mode** to see what's happening:
   ```bash
   DEBUG=1 python stt_tray_simple.py
   ```

2. **Run diagnostic tests**:
   ```bash
   # Test basic audio capture
   python test_audio.py
   
   # Test audio with visual meter
   python test_tray.py
   
   # Test model loading
   python test_models.py
   
   # Test wake word detection specifically
   python test_wake_word.py
   
   # Test core STT functionality
   python test_stt_core.py
   
   # Show debugging guide
   python debug_guide.py
   ```

3. **Check console output** - The tray version shows:
   - Device selection and name
   - Model loading status
   - Audio levels (in debug mode)
   - Wake word detections
   - All errors and status updates

4. **Try the simple version** if dialogs aren't working:
   ```bash
   python stt_tray_simple.py
   ```

5. **Common issues**:
   - No logs after "Listening for wake words..." - Audio may not be working
   - "Still listening..." messages but no detection - Wake word not being recognized
   - Icon always green - Normal when listening, should change to red when recording
   - AttributeError 'reset_audio_device' - Update to latest version of all files

### Low Audio Levels
- The app automatically amplifies audio 10x
- Run `DEBUG=1 python stt_tray.py` to see audio levels
- Check microphone permissions in System Settings

### Microphone Not Working
1. Run `python test_audio.py` to test your setup
2. Check microphone permissions
3. Ensure microphone is not muted
4. Try a different device index

### Wake Word Not Detected
1. Run `python test_vosk.py` to test wake word detection
2. Speak clearly and pause after the wake word
3. Try speaking louder or closer to the microphone
4. Consider using a simpler/shorter wake word

### Audio Device Disconnection
- The app automatically detects when audio devices disconnect
- It will attempt to reconnect up to 5 times
- If the original device is unavailable, it will try to find a similar device
- To reset device preference: `python stt.py --reset-audio-device`

### macOS Permissions
Grant Terminal/Python permissions in:
- System Preferences → Security & Privacy → Microphone
- System Preferences → Security & Privacy → Accessibility (for auto-typing)

## How It Works

1. **Wake Word Detection**: Continuously listens using Vosk models for wake words
2. **Sound Alert**: Plays a sound when wake word is detected
3. **Recording**: Records audio until silence is detected
4. **Transcription**: Uses OpenAI Whisper for accurate speech-to-text
5. **Auto-Typing**: Types the transcribed text at your cursor position
6. **Feedback**: Plays a sound (macOS) to confirm completion

## Audio Feedback

The application provides audio feedback on macOS:
- **Tink sound**: When wake word is detected (transcription starts)
- **Glass sound**: When transcription is complete and text is typed

## File Structure

- `stt.py` - Core speech-to-text functionality
- `stt_tray.py` - System tray interface with native dialogs
- `stt_tray_simple.py` - System tray interface without dialogs (fallback)
- `run.sh` - Startup script that handles icon generation and fallbacks
- `generate_icons.py` - Creates system tray icons
- `test_*.py` - Various diagnostic and testing tools
- `debug_guide.py` - Interactive debugging guide

## License

This project uses open-source models and libraries. Please check individual model licenses:
- Vosk: Apache 2.0
- Whisper: MIT
- Model weights may have different licenses