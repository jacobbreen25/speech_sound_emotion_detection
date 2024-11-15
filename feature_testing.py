import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import utilities as util

# Load the audio file
data, sample_rate = util.load_wav_file('test.wav')

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
print("Mel-Frequency Cepstral Coefficients Shape:",mfccs.shape)

# Compute the chroma features
chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
print("Chroma Shape:",chroma.shape)

# Compute the spectral centroid
spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
print("Spectral Centroid Shape:",spectral_centroid.shape)

# Compute the zero-crossing rate
zero_crossings = librosa.feature.zero_crossing_rate(y=data)
print("Zero Crossing Shape:",zero_crossings.shape)

# Compute RMSE
rmse = librosa.feature.rms(y=data)
print("Root Mean Square Energy Shape:",rmse.shape)

# Compute the Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
print("Mel Spectrogram Shape:",mel_spectrogram.shape)

# Display Mel Spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time', sr=sample_rate)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()

# Display MFCCs
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate)
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()
