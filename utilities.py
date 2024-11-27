import os, csv
import librosa

# RAVDESS Dataset: https://zenodo.org/records/1188976 

# Load .wav file data into a numpy array
def load_wav_file(path,verbose=True):
    data, sample_rate = librosa.load(path)
    if verbose:
        print(f"\nFile: \'{path}\' - Size: {data.shape} - Rate: {sample_rate} - Duration: {data.shape[0]/sample_rate:.2f}s")
    return data, sample_rate

# Return list of all .wav file paths in some directory, including all subdirectories 
def find_wav_files(base_directory):
    wav_files = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

# Get the details of the file being processed
def get_file_details(file_path, decode=False, verbose=True):
    file = os.path.basename(file_path)
    identifiers = file[:-4].split('-')
    # print(identifiers)
    details = dict()
    details['modality'] = [None, "full-AV", "video-only", "audio-only"][int(identifiers[0])] if decode else identifiers[0]
    details['channel'] = [None, "speech", "song"][int(identifiers[1])] if decode else identifiers[1]
    details['emotion'] = [None, "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"][int(identifiers[2])] if decode else identifiers[2]
    details['intensity'] = [None, "normal", "strong"][int(identifiers[3])] if decode else identifiers[3]
    details['statement'] = [None, "Kids are talking by the door","Dogs are sitting by the door"][int(identifiers[4])] if decode else identifiers[4]
    details['repetition'] = [None, "1st Repetition", "2nd Repetiion"][int(identifiers[5])] if decode else identifiers[5]
    details['actor'] = str(identifiers[6]) + (" (Female)" if int(identifiers[6])%2==0 else " (Male)") if decode else identifiers[6]
    if verbose:
        s = ""
        for k in details:
            s += details[k] + " - "
        print(s[:-3])
    return details
    

# Export csv data file
def write_csv(file_path, data, headers=None):
    # Extract the directory from the file path
    directory = os.path.dirname(file_path)
    
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        print(f"Creating Directory: {directory}")
        os.makedirs(directory)
    
    # Write the data to the specified file path
    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        if headers is not None:
            csv_writer.writerow(headers)
        csv_writer.writerows(data)

    print(f"Finished writing {file_path}")