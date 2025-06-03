#!/usr/bin/env python3
"""
Debug guide for wake word detection issues
"""

def show_debug_guide():
    print("\n" + "="*70)
    print("🔍 DEBUG GUIDE - Wake Word Detection Issues")
    print("="*70)
    
    print("\n📋 STEP-BY-STEP DEBUGGING:")
    
    print("\n1. 🧪 Test basic audio capture:")
    print("   Run: python test_tray.py")
    print("   Expected: Should show audio levels when you speak")
    print("   If fails: Check microphone permissions")
    
    print("\n2. 🤖 Test model loading:")
    print("   Run: python test_models.py")
    print("   Expected: All models should load without errors")
    print("   If fails: Models not downloaded or corrupted")
    
    print("\n3. 👂 Test wake word detection:")
    print("   Run: DEBUG=1 python test_wake_word.py")
    print("   Expected: Should show recognition results when you speak")
    print("   Say 'transcribe' or 'transcreva' clearly")
    
    print("\n4. 🔧 Test system tray with full debug:")
    print("   Run: DEBUG=1 python stt_tray_simple.py")
    print("   Expected output to look for:")
    print("   - ✅ STT app initialized successfully")
    print("   - 🎙️ Audio Device Information: [shows your device]")
    print("   - 🤖 Loaded Models: [shows 3 models]")
    print("   - 👂 Wake Word Detector: [shows wake words and debug=True]")
    print("   - ✅ English/Portuguese recognizer initialized")
    print("   - 🎯 Listening for wake words...")
    print("   - 💓 Still listening... (appears every 5 seconds)")
    print("   - When you speak: [EN PARTIAL] or [PT PARTIAL] should appear")
    
    print("\n🚨 COMMON ISSUES AND FIXES:")
    
    print("\n• No audio levels in test_tray.py:")
    print("  → Check microphone permissions in System Preferences")
    print("  → Try a different audio device")
    
    print("\n• Models fail to load:")
    print("  → Run the app once to download models")
    print("  → Check internet connection")
    
    print("\n• No [EN PARTIAL] or [PT PARTIAL] messages:")
    print("  → Speak louder and clearer")
    print("  → Check audio device is correct")
    print("  → Increase audio gain (currently 10x)")
    
    print("\n• [EN PARTIAL] shows but no wake word detection:")
    print("  → Try saying 'transcribe' more clearly")
    print("  → Try variations: 'transcript', 'transcription'")
    print("  → Check if wake word appears in partial results")
    
    print("\n• Wake word detector not initialized:")
    print("  → Models failed to load")
    print("  → Check model files exist in current directory")
    
    print("\n💡 QUICK FIXES TO TRY:")
    print("  1. Restart with: python stt.py --no-tray")
    print("  2. Use simpler wake word: --wake-word-en 'start'")
    print("  3. Try large models: --model-en large --model-pt large")
    print("  4. Reset audio device: python stt.py --reset-audio-device")
    
    print("\n📞 WHAT TO REPORT:")
    print("  Include output from: DEBUG=1 python stt_tray_simple.py")
    print("  Specifically:")
    print("  - Audio device information")
    print("  - Model loading status") 
    print("  - Wake word detector initialization")
    print("  - Any error messages")
    print("  - Whether you see [EN PARTIAL] messages when speaking")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    show_debug_guide()