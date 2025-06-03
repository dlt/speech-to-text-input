#!/bin/bash
# Simple startup script for Speech-to-Text

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Generate icons if they don't exist
if [ ! -d "icons" ]; then
    echo "Generating icons..."
    python generate_icons.py
fi

# Try to run with system tray
echo "Starting Speech-to-Text with system tray..."
python stt_tray.py "$@"

# If it fails, try the simple version
if [ $? -ne 0 ]; then
    echo "Failed to start with dialogs. Trying simple version..."
    python stt_tray_simple.py "$@"
fi