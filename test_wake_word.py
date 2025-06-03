#!/usr/bin/env python3
"""
Test wake word detection specifically
"""

import pyaudio
import numpy as np
import time
import sys
import os
import json
from vosk import Model, KaldiRecognizer

# Force debug mode
os.environ['DEBUG'] = '1'

def test_wake_word_detection():
    """Test wake word detection with Vosk"""
    print("\n" + "="*60)
    print("🧪 TESTING WAKE WORD DETECTION")
    print("="*60)
    
    # Configuration
    wake_word_en = "transcribe"
    wake_word_pt = "transcreva"
    
    print(f"\n📝 Wake words:")
    print(f"   English: '{wake_word_en}'")
    print(f"   Portuguese: '{wake_word_pt}'")
    
    # Load models
    print("\n🔧 Loading models...")
    
    try:
        # English model
        en_model_path = "vosk-model-small-en-us-0.15"
        print(f"   Loading EN model from: {en_model_path}")
        if not os.path.exists(en_model_path):
            print(f"   ❌ English model not found!")
            return
        model_en = Model(en_model_path)
        rec_en = KaldiRecognizer(model_en, 16000)
        rec_en.SetWords(True)
        print(f"   ✅ English model loaded")
        
        # Portuguese model
        pt_model_path = "vosk-model-small-pt-0.3"
        print(f"   Loading PT model from: {pt_model_path}")
        if not os.path.exists(pt_model_path):
            print(f"   ❌ Portuguese model not found!")
            return
        model_pt = Model(pt_model_path)
        rec_pt = KaldiRecognizer(model_pt, 16000)
        rec_pt.SetWords(True)
        print(f"   ✅ Portuguese model loaded")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Setup audio
    print("\n🎙️ Setting up audio...")
    p = pyaudio.PyAudio()
    
    # List devices
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
    
    # Open stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=480  # 30ms chunks
    )
    
    print(f"\n🎤 Listening for wake words...")
    print(f"   Say '{wake_word_en}' or '{wake_word_pt}'")
    print(f"   Press Ctrl+C to stop\n")
    
    frame_count = 0
    
    try:
        while True:
            # Read audio
            data = stream.read(480, exception_on_overflow=False)
            frame_count += 1
            
            # Show activity every second
            if frame_count % 33 == 0:  # ~1 second
                audio_array = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio_array).mean()
                print(f"📊 Audio level: {level:.1f}", end='\r', flush=True)
            
            # Check English
            if rec_en.AcceptWaveform(data):
                result = json.loads(rec_en.Result())
                text = result.get("text", "").lower()
                if text:
                    print(f"\n[EN] Recognized: '{text}'")
                    if wake_word_en in text:
                        print(f"🟢 WAKE WORD DETECTED! (English)")
                    else:
                        print(f"   (Looking for '{wake_word_en}')")
            else:
                # Check partial
                partial = json.loads(rec_en.PartialResult())
                if partial.get("partial"):
                    partial_text = partial['partial'].lower()
                    if partial_text and frame_count % 10 == 0:  # Don't spam
                        print(f"\n[EN Partial] {partial_text}", end='', flush=True)
            
            # Check Portuguese  
            if rec_pt.AcceptWaveform(data):
                result = json.loads(rec_pt.Result())
                text = result.get("text", "").lower()
                if text:
                    print(f"\n[PT] Recognized: '{text}'")
                    if wake_word_pt in text:
                        print(f"🟢 WAKE WORD DETECTED! (Portuguese)")
                    else:
                        print(f"   (Looking for '{wake_word_pt}')")
            else:
                # Check partial
                partial = json.loads(rec_pt.PartialResult())
                if partial.get("partial"):
                    partial_text = partial['partial'].lower()
                    if partial_text and frame_count % 10 == 0:  # Don't spam
                        print(f"\n[PT Partial] {partial_text}", end='', flush=True)
                        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_wake_word_detection()