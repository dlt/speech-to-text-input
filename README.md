# Speech-to-Text Auto-Typer

A real-time speech-to-text application that listens for wake words and automatically types transcribed text at your cursor position. Supports both English and Portuguese languages with global keyboard shortcuts for maximum convenience.

## Features

- **System Tray Interface**: Control everything from the system tray - no terminal needed
- **Wake Word Activation**: Say "transcribe" (English) or "transcreva" (Portuguese) to start recording
- **🆕 Global Keyboard Shortcuts**: Press customizable keyboard shortcuts (e.g., Cmd+Shift+T) from any app to trigger transcription
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

- **Python 3.7+**
- **macOS** (for full functionality including auto-typing and keyboard shortcuts) or **Linux/Windows** with additional setup
- **Microphone access**
- **For keyboard shortcuts**: Accessibility permissions on macOS

## Installation

### Quick Start (macOS)

1. **Install Python 3.7+** (if not already installed):
   ```bash
   # Using Homebrew (recommended)
   brew install python
   
   # Or download from python.org
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd speech-to-text
   ```

3. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Grant permissions** (will be prompted during first run):
   - **Microphone access**: System Preferences → Security & Privacy → Privacy → Microphone
   - **Accessibility access** (for auto-typing): System Preferences → Security & Privacy → Privacy → Accessibility
   - **For keyboard shortcuts**: Add Terminal or Python to Accessibility permissions

6. **Run the application**:
   ```bash
   # With system tray interface (recommended)
   python stt_tray.py
   
   # Or terminal mode with keyboard shortcuts
   python stt.py --keyboard-shortcut "cmd+shift+t"
   ```

### Detailed Installation Guide

#### Step 1: Python Installation

**macOS:**
```bash
# Method 1: Using Homebrew (recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python

# Method 2: Download from python.org
# Visit https://www.python.org/downloads/ and download Python 3.7+
```

**Windows:**
```bash
# Download from python.org or use winget
winget install Python.Python.3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-dev
```

#### Step 2: Clone Repository

```bash
git clone <repository-url>
cd speech-to-text
```

#### Step 3: Virtual Environment Setup

**All Platforms:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### Step 4: Install Dependencies

**macOS/Linux:**
```bash
# Install main dependencies
pip install -r requirements.txt

# If you get audio errors, you may need system dependencies:
# macOS:
brew install portaudio
pip install pyaudio

# Linux (Ubuntu/Debian):
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Windows:**
```bash
# Install main dependencies
pip install -r requirements.txt

# For Windows, you might need to install pyaudio separately:
pip install pipwin
pipwin install pyaudio
```

#### Step 5: System Permissions (macOS)

The application requires specific permissions on macOS:

1. **Microphone Access**:
   - System Preferences → Security & Privacy → Privacy → Microphone
   - Add Terminal (or your Python executable) to the list
   - Check the box to enable microphone access

2. **Accessibility Access** (for auto-typing):
   - System Preferences → Security & Privacy → Privacy → Accessibility
   - Click the lock to make changes
   - Add Terminal (or your Python executable) to the list
   - Check the box to enable accessibility access

3. **Input Monitoring** (for keyboard shortcuts):
   - System Preferences → Security & Privacy → Privacy → Input Monitoring
   - Add Terminal (or your Python executable) to the list
   - Check the box to enable input monitoring

#### Step 6: Verify Installation

Test your installation:

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Test basic functionality
python test_audio.py

# Test models loading
python test_models.py

# Run with debug mode to see detailed output
DEBUG=1 python stt.py --list-devices
```

### Alternative Installation Methods

#### Using pip directly (not recommended for beginners):
```bash
pip install pyaudio numpy openai-whisper webrtcvad vosk pynput pystray pillow pyautogui
```

#### For developers:
```bash
# Clone with development dependencies
git clone <repository-url>
cd speech-to-text
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Troubleshooting Installation

**Common Issues:**

1. **PyAudio installation fails**:
   ```bash
   # macOS:
   brew install portaudio
   pip install pyaudio
   
   # Linux:
   sudo apt install portaudio19-dev
   pip install pyaudio
   
   # Windows:
   pip install pipwin
   pipwin install pyaudio
   ```

2. **Permission denied errors**:
   ```bash
   # Make sure you're in virtual environment
   source venv/bin/activate
   
   # Or use --user flag
   pip install --user -r requirements.txt
   ```

3. **Models don't download**:
   ```bash
   # Check internet connection and try:
   python test_models.py
   
   # Or manually create models directory:
   mkdir models
   ```

4. **Microphone not detected**:
   ```bash
   # List available devices
   python stt.py --list-devices
   
   # Test audio capture
   python test_audio.py
   ```

**Getting Help:**
If you encounter issues during installation:
1. Run the debug guide: `python debug_guide.py`
2. Check the detailed logs with: `DEBUG=1 python stt.py`
3. Test individual components with the test scripts

The models will be downloaded automatically when you first run the application (this may take a few minutes depending on your internet connection).

## Usage

### Basic Usage

**Option 1: System Tray Interface (Recommended)**
```bash
python stt_tray.py
```

**Option 2: Terminal Mode with Keyboard Shortcuts**
```bash
# Use Cmd+Shift+T as global shortcut
python stt.py --keyboard-shortcut "cmd+shift+t"

# Use Ctrl+Alt+S as shortcut
python stt.py --keyboard-shortcut "ctrl+alt+s"

# Multiple modifiers
python stt.py --keyboard-shortcut "cmd+shift+ctrl+r"
```

**Option 3: Classic Terminal Mode (Wake Words Only)**
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

### Keyboard Shortcuts Guide

The new keyboard shortcut feature allows you to trigger transcription from anywhere on your system:

**Supported Key Combinations:**
- **Modifiers**: `cmd`, `ctrl`, `alt`/`option`, `shift`
- **Regular keys**: Any letter (a-z), number (0-9)
- **Format**: Join keys with `+` (e.g., `"cmd+shift+t"`)

**Popular Shortcut Examples:**
```bash
# Recommended for macOS
python stt.py --keyboard-shortcut "cmd+shift+t"

# Alternative for macOS
python stt.py --keyboard-shortcut "cmd+option+s"

# For users who prefer Ctrl
python stt.py --keyboard-shortcut "ctrl+alt+r"

# Single modifier shortcuts
python stt.py --keyboard-shortcut "cmd+t"
```

**How Keyboard Shortcuts Work:**
1. Press your configured shortcut from **any application**
2. System plays a sound to indicate recording started
3. Speak your text (recording stops automatically when you pause)
4. Text is typed at your current cursor position
5. System plays completion sound

**Important Notes:**
- Keyboard shortcuts default to **English transcription**
- Works system-wide - no need to focus on terminal
- Requires **Input Monitoring** permissions on macOS
- Both wake words and keyboard shortcuts work simultaneously

### Advanced Usage

```bash
# Use large models with keyboard shortcuts for best accuracy
python stt.py --keyboard-shortcut "cmd+shift+t" --model-en large --model-pt large --whisper-model medium

# System tray with large models for better accuracy
python stt_tray.py --model-en large --model-pt large --whisper-model medium

# List available audio devices (terminal mode)
python stt.py --list-devices

# Use specific models with custom wake words and shortcuts
python stt.py --keyboard-shortcut "cmd+r" --model-en small --model-pt large --wake-word-en "record"

# Classic terminal mode with all options
python stt.py --wake-word-en "start" --wake-word-pt "começar" --keyboard-shortcut "ctrl+alt+t"

# Reset audio device preference
python stt.py --reset-audio-device
```

**Command-line Options:**
- `--model-en {small,large}`: English Vosk model size (default: small)
- `--model-pt {small,large}`: Portuguese Vosk model size (default: small)
- `--whisper-model {tiny,base,small,medium,large}`: Whisper model size (default: base)
- `--wake-word-en WORD`: Custom English wake word (default: transcribe)
- `--wake-word-pt WORD`: Custom Portuguese wake word (default: transcreva)
- `--keyboard-shortcut SHORTCUT`: Global keyboard shortcut (e.g., "cmd+shift+t")
- `--list-devices`: List available audio devices and exit
- `--reset-audio-device`: Reset saved audio device preference

### Usage Steps

1. Run the application with your desired model configuration

2. Select your microphone device (if multiple are available)

3. Click where you want the text to appear (text editor, browser, etc.)

4. **Trigger transcription using either method:**
   - **Wake words**: Say "transcribe" (English) or "transcreva" (Portuguese)
   - **Keyboard shortcut**: Press your configured shortcut (e.g., Cmd+Shift+T)

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
- System Preferences → Security & Privacy → Privacy → Microphone
- System Preferences → Security & Privacy → Privacy → Accessibility (for auto-typing)
- System Preferences → Security & Privacy → Privacy → Input Monitoring (for keyboard shortcuts)

## How It Works

1. **Dual Trigger System**: 
   - **Wake Word Detection**: Continuously listens using Vosk models for wake words
   - **Keyboard Shortcuts**: Global hotkey listener using pynput library
2. **Sound Alert**: Plays a sound when wake word is detected or shortcut is pressed
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