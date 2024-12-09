import pyaudio
import numpy as np
from network_model import FullyConnectedNN
import torch
import csv
import generate_training_data as gen
import preprocess_audio as pre
from os.path import join

def record_last_second(sample_rate=44100, channels=1, chunk_size=512):
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
    # Make sure to define the same model architecture
    # First we need the number of training features for input size (or just hardcode it if you know it)
    training_dataset = "data_v3.csv"
    with open(training_dataset, 'r') as file:
        reader = csv.reader(file)
        first_line = next(reader)  # Get the first line
        num_features = len(first_line[1:])
    print(f"Features found in dataset: {num_features}")
    audio_chunk = record_last_second()
    audio = pre.process_non_silent_clip(audio_chunk, 44100, clip_duration=1.0, step_size=1.0)
    features = gen.generate_features(audio[0], 44100)
    print(features.shape)
    model = FullyConnectedNN(features.shape[1])
    model_dir = "./models"
    model_name = "model_v1.pth"
    
    # Load only the state dictionary safely
    state_dict = torch.load(join(model_dir,model_name), weights_only=True)
    # Load it into your model
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()  

    print("Model loaded successfully!")
    
    while True:
        audio_chunk = record_last_second()
        audio = pre.process_non_silent_clip(audio_chunk, 44100, clip_duration=1.0, step_size=1.0)
        features = gen.generate_features(audio[0], 44100)
        # model(features)
        preds = torch.argmax(model(features), dim=1)
        print(preds)
        # print(f"Recorded audio chunk with shape: {audio_chunk.shape}")
        
