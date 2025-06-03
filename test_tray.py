#!/usr/bin/env python3
"""
Test script to verify system tray audio is working
"""

import pyaudio
import numpy as np
import time

def test_audio():
    """Test basic audio capture"""
    print("Testing audio capture...")
    
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
        frames_per_buffer=480  # 30ms at 16kHz
    )
    
    print("\nCapturing audio for 3 seconds...")
    print("Speak to see audio levels...\n")
    
    start_time = time.time()
    while time.time() - start_time < 3:
        try:
            data = stream.read(480, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            level = np.abs(audio_array).mean()
            
            # Show level bar
            bar_length = int(level / 100)
            bar = '█' * min(bar_length, 50)
            print(f"\rLevel: {level:6.1f} |{bar:<50}|", end='', flush=True)
            
        except Exception as e:
            print(f"\nError: {e}")
            break
    
    print("\n\nDone!")
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    test_audio()