#!/usr/bin/env python3
"""
Test if models are loading correctly
"""

import os
import sys
from vosk import Model, KaldiRecognizer
import whisper
import json

def test_models():
    """Test model loading"""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL LOADING")
    print("="*60)
    
    # Test Vosk English model
    print("\n1. Testing Vosk English model...")
    try:
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print(f"   ❌ Model not found at: {model_path}")
        else:
            print(f"   📁 Found model at: {model_path}")
            model = Model(model_path)
            rec = KaldiRecognizer(model, 16000)
            rec.SetWords(True)
            print("   ✅ Model loaded successfully")
            
            # Test with empty audio
            result = rec.FinalResult()
            print(f"   📝 Test result: {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test Vosk Portuguese model
    print("\n2. Testing Vosk Portuguese model...")
    try:
        model_path = "vosk-model-small-pt-0.3"
        if not os.path.exists(model_path):
            print(f"   ❌ Model not found at: {model_path}")
        else:
            print(f"   📁 Found model at: {model_path}")
            model = Model(model_path)
            rec = KaldiRecognizer(model, 16000)
            rec.SetWords(True)
            print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test Whisper model
    print("\n3. Testing Whisper model...")
    try:
        print("   Loading Whisper 'base' model...")
        model = whisper.load_model("base")
        print("   ✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # List available models
    print("\n4. Available model directories:")
    for item in os.listdir('.'):
        if item.startswith('vosk-model') and os.path.isdir(item):
            print(f"   📁 {item}")
    
    print("\n✅ Model test complete!")

if __name__ == "__main__":
    test_models()