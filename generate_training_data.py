import utilities as util
from preprocess_audio import preprocess_audio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# CREATES FEATURES FROM AN ARRAY OF AUDIO DATA    
# BUILD ON THIS
# generates a csv that shows the features for each individual audio file
def generate_features(data, sample_rate, verbose=True):
    np.set_printoptions(linewidth=1000000000000)

    features = np.array([], dtype=float)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    # chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    SC = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
    ZC = librosa.feature.zero_crossing_rate(y=data)
    RMSE = librosa.feature.rms(y=data)
    ML = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)

    # features.append(" ".join(map(str, mfcc.flatten())))
    # features.append(mfcc.shape)
    # features.append(mfcc.flatten())
    # features.append(" ".join(map(str, chroma.flatten())))
    # features.append(chroma.shape)
    # features.append(" ".join(map(str, SC.flatten())))
    # features.append(SC.shape)
    # features.append(" ".join(map(str, ZC.flatten())))
    # features.append(ZC.shape)
    # features.append(" ".join(map(str, RMSE.flatten())))
    # features.append(RMSE.shape)
    # features.append(" ".join(map(str, ML.flatten())))
    # features.append(ML.shape)

    features = np.hstack((features,mfcc.flatten()))
    # if verbose: print(f"New Features Shape: {features.shape} - MFCC Shape: {mfcc.shape}")
    # features = np.hstack((features,chroma.flatten()))
    # if verbose: print(f"New Features Shape: {features.shape} - Chroma Shape: {chroma.shape}")
    features = np.hstack((features,SC.flatten()))
    # if verbose: print(f"New Features Shape: {features.shape} - SC Shape: {SC.shape}")
    features = np.hstack((features,ZC.flatten()))
    # if verbose: print(f"New Features Shape: {features.shape} - ZC Shape: {ZC.shape}")
    features = np.hstack((features,RMSE.flatten()))
    # if verbose: print(f"New Features Shape: {features.shape} - RMSE Shape: {RMSE.shape}")
    features = np.hstack((features,ML.flatten()))
    # if verbose: print(f"New Features Shape: {features.shape} - ML Shape: {ML.shape}")
    return features



# Generate training data features and headers to write
def generate_training_data(data_dir,decode_details,stop_after=None):
    data = []
    headers = ["File","Modality","Channel","Emotion","Intensity","Statement",
               "Repetition","Actor", "MFCC", "MFCC shape", "chroma", "chroma shape",
               "SC", "SC shape", "ZC", "ZC shape", "RMSE", "RMSE shape", "ML", "ML shape"]
    files = util.find_wav_files(data_dir)
    print(f"Found {len(files)} wav files in directory \'{data_dir}\'.")
    if stop_after is not None: files = files[:stop_after]
    counts = np.zeros(9, dtype=int)
    for f in files:
        d_audio = [] # [f]
        d_silent = [] # [f]
        # wav_data, sr = util.load_wav_file(f)
        non_silent_data, silent_data, sr = preprocess_audio(f)
        details = util.get_file_details(f,decode_details)
        # for info in details: d.append(details[info])
        d_audio.append(details['emotion'])
        # d_audio.append(details['intensity'])
        d_silent.append("0") # Set emotion to neutral for silent clip
        # d_silent.append("0") # Set intensity to normal for silent clip
        
        # Append feature names to headers as needed
        for clip in non_silent_data:
            counts[details['emotion']] += 1
            audio_data = d_audio.copy()
            features = generate_features(clip, sr)
            print(f" > {len(features)} features - Audio")
            # features = ['test']
            for x in features: audio_data.append(x)
            data.append(audio_data)

        if silent_data is not None:
            counts[0] += 1
            features = generate_features(silent_data, sr)
            print(f" > {len(features)} features - Silent")
            # features = ['test']
            for x in features: d_silent.append(x)
            data.append(d_silent)

    return data, headers, counts


if __name__ == "__main__":
    DATA_DIR = '../data'
    OUTPUT_FILE = './data_v2.csv'
    DECODE_DETAILS = False
    STOP_AFTER = None#None # set to None to go through all data
    data, headers, counts = generate_training_data(DATA_DIR, DECODE_DETAILS, STOP_AFTER)
    # util.write_csv(OUTPUT_FILE,data,headers)
    util.write_csv(OUTPUT_FILE,data)
    # Plotting the bar graph
    x_labels = ["silence", "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]  # Label names
    plt.bar(x_labels, counts, color='skyblue')
    # Add titles and labels
    plt.title("Label Counts")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
