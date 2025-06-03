#!/usr/bin/env python3
"""
Test the core STT functionality without system tray
"""

import argparse
import time
import os
import sys
import numpy as np

# Set debug mode
os.environ['DEBUG'] = '1'

from stt import (
    SpeechToText, SettingsManager, ModelManager, AudioManager, 
    WakeWordDetector, Transcriber, TextTyper, MODELS
)

def test_stt():
    """Test core STT functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING CORE STT FUNCTIONALITY")
    print("="*60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test STT core')
    parser.add_argument('--model-en', choices=['small', 'large'], default='small')
    parser.add_argument('--model-pt', choices=['small', 'large'], default='small')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'], default='base')
    parser.add_argument('--wake-word-en', default='transcribe')
    parser.add_argument('--wake-word-pt', default='transcreva')
    args = parser.parse_args()
    
    print(f"\n📝 Configuration:")
    print(f"   Wake words: EN:'{args.wake_word_en}', PT:'{args.wake_word_pt}'")
    print(f"   Models: EN:{args.model_en}, PT:{args.model_pt}, Whisper:{args.whisper_model}")
    
    # Initialize STT
    print("\n🔧 Initializing STT app...")
    stt = SpeechToText(args)
    
    print("🔧 Running setup...")
    if not stt.setup():
        print("❌ Failed to initialize STT!")
        return
    
    print("✅ STT initialized successfully")
    
    # Show audio device info
    if stt.audio_manager:
        print(f"\n🎙️ Audio Device Information:")
        print(f"   Device Index: {stt.audio_manager.input_device_index}")
        print(f"   Sample Rate: {stt.audio_manager.rate} Hz")
        print(f"   Chunk Size: {stt.audio_manager.chunk} samples")
        print(f"   Audio Gain: {stt.audio_manager.audio_gain}x")
        
        # Get device name
        try:
            devices = stt.audio_manager.get_available_devices()
            for idx, name in devices:
                if idx == stt.audio_manager.input_device_index:
                    print(f"   Device Name: {name}")
                    break
        except:
            pass
    
    # Test audio capture
    print("\n🧪 Testing audio capture for 3 seconds...")
    print("   Speak to see if audio is being captured...\n")
    
    start_time = time.time()
    max_level = 0
    chunk_count = 0
    
    while time.time() - start_time < 3:
        audio_chunk = stt.audio_manager.read_chunk()
        if audio_chunk is not None:
            chunk_count += 1
            level = np.abs(audio_chunk).mean()
            max_level = max(max_level, level)
            
            # Show level bar
            bar_length = int(level / 100)
            bar = '█' * min(bar_length, 50)
            print(f"\r   Level: {level:6.1f} |{bar:<50}|", end='', flush=True)
        else:
            print("\n   ⚠️  Audio read failed!")
            break
    
    print(f"\n\n📊 Audio test results:")
    print(f"   Chunks read: {chunk_count}")
    print(f"   Max level: {max_level:.1f}")
    print(f"   Status: {'✅ Working' if chunk_count > 0 else '❌ Not working'}")
    
    if chunk_count == 0:
        print("\n❌ No audio data captured! Check your microphone.")
        return
    
    # Test wake word detection
    print(f"\n🧪 Testing wake word detection...")
    print(f"   Say '{args.wake_word_en}' or '{args.wake_word_pt}' to test...")
    print("   (Testing for 10 seconds)\n")
    
    start_time = time.time()
    detected = False
    
    while time.time() - start_time < 10 and not detected:
        audio_chunk = stt.audio_manager.read_chunk()
        if audio_chunk is not None:
            # Show we're listening
            if int(time.time() - start_time) % 2 == 0:
                print(".", end='', flush=True)
            
            lang = stt.wake_word_detector.detect(audio_chunk.tobytes())
            if lang:
                print(f"\n\n🟢 Wake word detected! Language: {lang}")
                detected = True
                break
    
    if not detected:
        print(f"\n\n⚠️  No wake word detected in 10 seconds")
        print("   Possible issues:")
        print("   - Wake word not spoken clearly")
        print("   - Audio levels too low")
        print("   - Model not recognizing the word")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    stt.audio_manager.cleanup()
    print("✅ Done!")

if __name__ == "__main__":
    test_stt()