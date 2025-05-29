import pyaudio
import numpy as np
import json
from vosk import Model, KaldiRecognizer

print("🧪 Testing Vosk with amplified audio\n")

# Audio settings
RATE = 16000
CHUNK = 480
GAIN = 10.0

# Load model
print("Loading Vosk model...")
model = Model("vosk-model-small-en-us-0.15")
rec = KaldiRecognizer(model, RATE)
rec.SetWords(True)

# Setup audio
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, 
                 input=True, frames_per_buffer=CHUNK)

print(f"✅ Model loaded. Audio gain: {GAIN}x")
print("\n🎤 Speak clearly and say 'transcribe' or any other words...")
print("Press Ctrl+C to stop\n")

try:
    while True:
        # Read and amplify
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        audio_array = audio_array * GAIN
        audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
        amplified_data = audio_array.tobytes()
        
        # Check level
        level = np.abs(audio_array).mean()
        
        # Process with Vosk
        if rec.AcceptWaveform(amplified_data):
            result = json.loads(rec.Result())
            if result.get("text"):
                print(f"✅ FINAL: {result['text']}")
        else:
            partial = json.loads(rec.PartialResult())
            if partial.get("partial"):
                print(f"⏳ PARTIAL: {partial['partial']}", end='\r')
        
        # Show audio level bar
        if level > 50:
            bar = "█" * int(min(level/200, 20))
            print(f"🔊 Level: {level:.0f} {bar}", end='\r')

except KeyboardInterrupt:
    print("\n\n👋 Stopped")

stream.stop_stream()
stream.close()
pa.terminate()