import pyaudio
import wave
import numpy as np
import threading
import collections

# Audio settings
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate (Hz)
CHUNK = 1024              # Buffer size
SEGMENT_SECONDS = 5       # Duration of each segment
STEP_SECONDS = 1          # Step size for overlapping segments
# OUTPUT_FILE_PREFIX = "audio_segment_"  # File name prefix

# Initialize PyAudio
audio = pyaudio.PyAudio()

# audio_segments = []
audio_data = np.array([])
buffer_size = int(SEGMENT_SECONDS * RATE)  # Buffer size in samples
step_size = int(STEP_SECONDS * RATE)       # Step size in samples

# Circular buffer to hold audio samples
circular_buffer = collections.deque(maxlen=buffer_size)

# Global flag to stop the recording
stop_recording = threading.Event()

# Function to capture audio and save overlapping segments
def capture_audio():
    stream = None
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)

        print("Recording... Press Ctrl+C to stop.")
        # global audio_segments
        # segment_number = 1

        while not stop_recording.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)

                # Add new samples to the circular buffer
                circular_buffer.extend(samples)

                # If the buffer is full, save the overlapping segment
                if len(circular_buffer) == buffer_size:
                    # Get the current segment as a NumPy array
                    audio_data = np.array(circular_buffer)

                    # Save the segment to a .wav file
                    # file_name = f"{OUTPUT_FILE_PREFIX}{segment_number}.wav"
                    # with wave.open(file_name, 'wb') as wf:
                    #     wf.setnchannels(CHANNELS)
                    #     wf.setsampwidth(audio.get_sample_size(FORMAT))
                    #     wf.setframerate(RATE)
                    #     wf.writeframes(audio_data.tobytes())

                    # Store the segment in the global list
                    # audio_segments.append(audio_data)

                    # print(f"Segment {segment_number} saved as {file_name}. Also stored in NumPy array.")
                    print(f"Updated NumPy array.")
                    # segment_number += 1

                    # Remove old samples equivalent to the step size
                    for _ in range(step_size):
                        circular_buffer.popleft()
            except IOError as e:
                print(f"Error reading audio stream: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()

# Main function to start recording
def main():
    capture_thread = threading.Thread(target=capture_audio)
    capture_thread.start()
    try:
        while capture_thread.is_alive():
            capture_thread.join(1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
        stop_recording.set()
        capture_thread.join()

# Run the program
try:
    main()
except KeyboardInterrupt:
    print("\nExiting...")
finally:
    audio.terminate()
