import utilities as util
import librosa
import librosa.display
import numpy as np

# CREATES FEATURES FROM AN ARRAY OF AUDIO DATA    
# BUILD ON THIS
# generates a csv that shows the features for each individual audio file
def generate_features(data, sample_rate):
    np.set_printoptions(linewidth=1000000000000)
    features = []
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    print(mfcc)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    SC = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
    ZC = librosa.feature.zero_crossing_rate(y=data)
    RMSE = librosa.feature.rms(y=data)
    ML = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)

    features.append(" ".join(map(str, mfcc.flatten())))
    features.append(mfcc.shape)
    features.append(" ".join(map(str, chroma.flatten())))
    features.append(chroma.shape)
    features.append(" ".join(map(str, SC.flatten())))
    features.append(SC.shape)
    features.append(" ".join(map(str, ZC.flatten())))
    features.append(ZC.shape)
    features.append(" ".join(map(str, RMSE.flatten())))
    features.append(RMSE.shape)
    features.append(" ".join(map(str, ML.flatten())))
    features.append(ML.shape)
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
    for f in files:
        d = [f]
        wav_data, sample_rate = util.load_wav_file(f)
        details = util.get_file_details(f,decode_details)
        for info in details: d.append(details[info])
        
        # Append feature names to headers as needed
        features = generate_features(wav_data, sample_rate)
        for x in features: d.append(x)
        
        data.append(d)
    return data, headers


if __name__ == "__main__":
    DATA_DIR = './data'
    OUTPUT_FILE = './data.csv'
    DECODE_DETAILS = True
    STOP_AFTER = None # set to None to go through all data
    data, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, STOP_AFTER)
    util.write_csv(OUTPUT_FILE,data,headers)
