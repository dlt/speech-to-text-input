import pyaudio
import numpy as np
import sys

# Test PyAudio and microphone
print("🎤 Testing PyAudio and Microphone Setup\n")

pa = pyaudio.PyAudio()

# List all devices
print("📋 Available Audio Devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"  [{i}] {info['name']} - {info['maxInputChannels']} channels, {info['defaultSampleRate']}Hz")

# Try to open default input
print("\n🔧 Testing default input device...")
try:
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=16000,
                     input=True,
                     frames_per_buffer=480)
    
    print("✅ Successfully opened audio stream")
    print("\n📊 Recording 3 seconds of audio... Make some noise!")
    
    levels = []
    for i in range(100):  # 3 seconds
        data = stream.read(480, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        level = np.abs(audio_data).mean()
        max_val = np.abs(audio_data).max()
        levels.append(level)
        
        # Visual meter
        bar_length = int(level / 100)
        bar = "█" * min(bar_length, 50)
        print(f"\r  Level: {level:6.1f} | Max: {max_val:5d} | {bar}", end="", flush=True)
    
    print(f"\n\n📈 Statistics:")
    print(f"  Average level: {np.mean(levels):.1f}")
    print(f"  Max level: {np.max(levels):.1f}")
    print(f"  Min level: {np.min(levels):.1f}")
    
    if np.mean(levels) < 5:
        print("\n⚠️  WARNING: Very low audio levels detected!")
        print("  Possible issues:")
        print("  - Microphone muted")
        print("  - Wrong device selected")
        print("  - Permissions not granted")
        print("  - Gain too low")
    
    stream.stop_stream()
    stream.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible solutions:")
    print("1. Check microphone permissions in System Preferences > Security & Privacy > Microphone")
    print("2. Make sure your microphone is connected and not muted")
    print("3. Try specifying a different device index")

pa.terminate()