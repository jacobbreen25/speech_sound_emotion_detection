import utilities as util

# CREATES FEATURES FROM AN ARRAY OF AUDIO DATA    
# BUILD ON THIS
def generate_features(data, sample_rate):
    features = []
    return features


# Generate training data features and headers to write
def generate_training_data(data_dir,decode_details,stop_after=None):
    data = []
    headers = ["File","Modality","Channel","Emotion","Intensity","Statement","Repetition","Actor"]
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
    DATA_DIR = '../data'
    OUTPUT_FILE = './data.csv'
    DECODE_DETAILS = True
    STOP_AFTER = 3 # set to None to go through all data
    data, headers = generate_training_data(DATA_DIR, DECODE_DETAILS, STOP_AFTER)
    util.write_csv(OUTPUT_FILE,data,headers)

