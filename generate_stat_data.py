import utilities as util
from preprocess_audio import preprocess_audio
import librosa
import librosa.display
import numpy as np

# file generates a different csv for each feature so each
# feature can be analyzed independently
def generate_features(data, sample_rate, feature_name):
    np.set_printoptions(linewidth=1000000000000)
    features = np.array([], dtype=float)
    if(feature_name == "mfcc"):
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
        features = np.hstack((features,mfcc.flatten()))
    if(feature_name == "chroma"):
        chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
        features = np.hstack((features,chroma.flatten()))
    if(feature_name == "sc"):
        SC = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
        features = np.hstack((features,SC.flatten()))
    if(feature_name == "zc"):
        ZC = librosa.feature.zero_crossing_rate(y=data)
        features = np.hstack((features,ZC.flatten()))
    if(feature_name == "rmse"):
        RMSE = librosa.feature.rms(y=data)
        features = np.hstack((features,RMSE.flatten()))
    if(feature_name == "ml"):
        ML = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
        features = np.hstack((features,ML.flatten()))
    if(feature_name == "chroma_cqt"):
        chroma_cqt = librosa.feature.chroma_cqt(y=data, sr=sample_rate)
        features = np.hstack((features,chroma_cqt.flatten()))
    if(feature_name == "chroma_cens"):
        chroma_cens = librosa.feature.chroma_cens(y=data, sr=sample_rate)
        features = np.hstack((features,chroma_cens.flatten()))
    if(feature_name == "chroma_vqt"):
        chroma_vqt = librosa.feature.chroma_vqt(y=data, sr=sample_rate, intervals='ji5', bins_per_octave=36)
        features = np.hstack((features,chroma_vqt.flatten()))
    if(feature_name == "sb"):
        sb = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)
        features = np.hstack((features,sb.flatten()))
    if(feature_name == "scn"):
        scn = librosa.feature.spectral_contrast(y=data, sr=sample_rate)
        features = np.hstack((features,scn.flatten()))
    if(feature_name == "sf"):
        sf = librosa.feature.spectral_flatness(y=data)
        features = np.hstack((sf.flatten()))
    if(feature_name == "sr"):
        sr = librosa.feature.spectral_rolloff(y=data, sr=sample_rate)
        features = np.hstack((sr.flatten()))
    if(feature_name == "tonnetz"):
        tonnetz = librosa.feature.tonnetz(y=data, sr=sample_rate)
        features = np.hstack((features,tonnetz.flatten()))
    return features



# Generate training data features and headers to write
def generate_training_data(data_dir,decode_details,feature_name,stop_after=None):
    data = []
    headers = ["emotion", "MFCC"]
    files = util.find_wav_files(data_dir)
    print(f"Found {len(files)} wav files in directory \'{data_dir}\'.")
    if stop_after is not None: files = files[:stop_after]
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
        #d_silent.append("0") # Set intensity to normal for silent clip
        
        # Append feature names to headers as needed
        for clip in non_silent_data:
            audio_data = d_audio.copy()
            features = generate_features(clip, sr, feature_name)
            # features = ['test']
            for x in features: audio_data.append(x)
            data.append(audio_data)

        if silent_data is not None:
            features = generate_features(silent_data, sr, feature_name)
            # features = ['test']
            for x in features: d_silent.append(x)
            data.append(d_silent)

    return data, headers


if __name__ == "__main__":
    DATA_DIR = './data'
    # OUTPUT_FILE = './data_stats.csv'
    DECODE_DETAILS = False
    STOP_AFTER = None#None # set to None to go through all data
    # there is 1000% a better way to do this but i spent to long trying to figure that out
    # and gave up

    #data_mfcc, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "mfcc", STOP_AFTER)
    #data_chroma, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "chroma", STOP_AFTER)
    #data_sc, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "sc", STOP_AFTER)
    #data_zc, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "zc", STOP_AFTER)
    #data_rmse, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "rmse", STOP_AFTER)
    #data_ml, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "ml", STOP_AFTER)
    data_chroma_cqt, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "chroma_cqt", STOP_AFTER)
    data_chroma_cens, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "chroma_cens", STOP_AFTER)
    data_chroma_vqt, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "chroma_vqt", STOP_AFTER)
    data_sb, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "sb", STOP_AFTER)
    data_scn, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "scn", STOP_AFTER)
    data_sf, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "sf", STOP_AFTER)
    data_sr, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "sr", STOP_AFTER)
    data_tonnetz, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, "tonnetz", STOP_AFTER)

    #util.write_csv("./mfcc.csv", data_mfcc)
    #util.write_csv("./chroma.csv", data_chroma)
    #util.write_csv("./sc.csv", data_sc)
    #util.write_csv("./zc.csv", data_zc)
    #util.write_csv("./rmse.csv", data_rmse)
    #util.write_csv("./ml.csv", data_ml)
    # util.write_csv(OUTPUT_FILE,data)
    util.write_csv("./csvs/chroma_cqt.csv", data_chroma_cqt)
    util.write_csv("./csvs/chroma_cens.csv", data_chroma_cens)
    util.write_csv("./csvs/chroma_vqt.csv", data_chroma_vqt)
    util.write_csv("./csvs/sb.csv", data_sb)
    util.write_csv("./csvs/scn.csv", data_scn)
    util.write_csv("./csvs/sf.csv", data_sf)
    util.write_csv("./csvs/sr.csv", data_sr)
    util.write_csv("./csvs/tonnetz.csv", data_tonnetz)