import pyaudio
import wave
import numpy as np
import threading

# Audio settings
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate (Hz)
CHUNK = 1024              # Buffer size
RECORD_SECONDS = 5        # Duration of each segment
OUTPUT_FILE_PREFIX = "audio_segment_"  # File name prefix

# Initialize PyAudio
audio = pyaudio.PyAudio()

audio_segments = []

# Function to capture audio and save it to a .wav file
def capture_audio(segment_number):
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

    print(f"Recording segment {segment_number}...")
    frames = []

    # Capture data in chunks for the duration
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    # Combine frames into a single byte string
    raw_audio = b''.join(frames)
    
    # Convert raw audio to a NumPy array
    audio_data = np.frombuffer(raw_audio, dtype=np.int16)
    
    # Store the segment in the global list
    audio_segments.append(audio_data)

    # Save the audio to a .wav file
    file_name = f"{OUTPUT_FILE_PREFIX}{segment_number}.wav"
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(raw_audio)

    print(f"Segment {segment_number} saved as {file_name}. Also stored in NumPy array.")

# Start capturing audio in separate threads
def main():
    segment_number = 1
    while True:
        input("Press Enter to start recording a new segment (Ctrl+C to stop)...")
        thread = threading.Thread(target=capture_audio, args=(segment_number,))
        thread.start()
        thread.join()  # Wait for the thread to finish before proceeding
        segment_number += 1

# Run the program
try:
    main()
except KeyboardInterrupt:
    print("\nStopping audio capture...")
finally:
    audio.terminate()
