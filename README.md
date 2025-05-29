# Speech-to-Text Auto-Typer

A real-time speech-to-text application that listens for wake words and automatically types transcribed text at your cursor position. Supports both English and Portuguese languages.

## Features

- **Wake Word Activation**: Say "transcribe" (English) or "transcreva" (Portuguese) to start recording
- **Auto-Typing**: Transcribed text is automatically typed at your current cursor position
- **Multi-Language Support**: Works with both English and Portuguese
- **Real-Time Processing**: Uses VAD (Voice Activity Detection) to detect when you stop speaking
- **Dual Engine**: Combines Vosk for wake word detection and Whisper for accurate transcription
- **Audio Amplification**: Automatically amplifies low audio levels for better recognition

## Requirements

- Python 3.7+
- macOS (for auto-typing functionality) or any OS with pyautogui installed
- Microphone access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speach-to-text
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pyaudio numpy whisper webrtcvad vosk
```

For non-macOS systems, also install:
```bash
pip install pyautogui
```

4. Download Vosk models:
```bash
# English model (included in repo)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Portuguese model (included in repo)
wget https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip
unzip vosk-model-small-pt-0.3.zip
```

## Usage

### Basic Usage

1. Run the main application:
```bash
python stt.py
```

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

### Testing Tools

Test your audio setup:
```bash
python test_audio.py
```

Test Vosk wake word detection:
```bash
python test_vosk.py
```

## Better Accuracy with Larger Models

The repository includes small models for quick setup. For better accuracy, download larger models:

### Vosk Models

Browse available models at: https://alphacephei.com/vosk/models

For English, try:
- `vosk-model-en-us-0.22` (1.8 GB) - Much better accuracy
- `vosk-model-en-us-0.42-gigaspeech` (2.3 GB) - Best accuracy

For Portuguese:
- `vosk-model-pt-fb-v0.1.1-20220516_2113` (1.6 GB) - Better accuracy

### Whisper Models

Change the model size in `stt.py` line 87:
```python
whisper_model = whisper.load_model("base")  # Change to "small", "medium", "large"
```

Model sizes:
- `tiny`: 40 MB
- `base`: 150 MB (default)
- `small`: 500 MB
- `medium`: 1.5 GB
- `large`: 3 GB

## Troubleshooting

### Low Audio Levels
- The app automatically amplifies audio 10x
- Run `DEBUG=1 python stt.py` to see audio levels
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

### macOS Permissions
Grant Terminal/Python permissions in:
- System Preferences → Security & Privacy → Microphone
- System Preferences → Security & Privacy → Accessibility (for auto-typing)

## How It Works

1. **Wake Word Detection**: Continuously listens using Vosk models for wake words
2. **Recording**: Once activated, records audio until silence is detected
3. **Transcription**: Uses OpenAI Whisper for accurate speech-to-text
4. **Auto-Typing**: Types the transcribed text at your cursor position
5. **Feedback**: Plays a sound (macOS) to confirm completion

## License

This project uses open-source models and libraries. Please check individual model licenses:
- Vosk: Apache 2.0
- Whisper: MIT
- Model weights may have different licenses