import pyaudio
import numpy as np

def record_last_second(sample_rate=44100, channels=1, chunk_size=1024):
    """
    Records the last second of audio and returns it as a NumPy array.

    Args:
        sample_rate (int): The sample rate of the audio recording (default is 44100 Hz).
        channels (int): The number of audio channels (default is 1 for mono).
        chunk_size (int): The size of each audio chunk in frames.

    Returns:
        numpy.ndarray: A NumPy array of shape (sample_rate, channels) containing the last second of audio.
    """
    duration = 1  # seconds
    frames = []

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=channels,
                     rate=sample_rate,
                     input=True,
                     frames_per_buffer=chunk_size)
    
    print("Recording...")
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.float32))
    print("Recording complete.")
    
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    audio_data = np.concatenate(frames, axis=0)
    return audio_data.reshape(-1, channels)

# Example usage
if __name__ == "__main__":
    while True:
        audio_chunk = record_last_second()
        print(f"Recorded audio chunk with shape: {audio_chunk.shape}")
