import librosa
import numpy as np
import soundfile as sf  # For saving audio files

def process_silent_clip(clip, sr, clip_duration=1.0):
    clip_length = int(clip_duration * sr)  # Convert duration to samples

    if len(clip) < clip_length:
        # Pad short clips with zeroes
        processed_clip = np.pad(clip, (0, clip_length - len(clip)), mode='constant')
    else:
        # Crop clip if it's too long
        processed_clip = clip[:clip_length]

    return processed_clip

def process_non_silent_clip(clip, sr, clip_duration=1.0):
    clip_length = int(clip_duration * sr)  # Convert duration to samples
    processed_clips = []
    if len(clip) < clip_length:
        # Pad short clips with zeroes
        processed_clips.append(np.pad(clip, (0, clip_length - len(clip)), mode='constant'))
    else:
        #NOTE: This is lazy, and assumes the actual audio clip is less than 2 seconds,
        #   may need to create a more programatic way of breaking these up dynamically
        processed_clips.append(clip[:clip_length])
        processed_clips.append(clip[-clip_length:])

    return processed_clips
    
def remove_silence(audio_path, silence_thresh=-40, min_silence_duration=0.5, sample_rate=22050):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Detect non-silent intervals
    non_silent_intervals = librosa.effects.split(y, top_db=abs(silence_thresh), frame_length=2048, hop_length=512)

    # Initialize lists for silent and non-silent segments
    non_silent_audio = []
    silent_audio = []

    # Extract silent and non-silent audio
    last_end = 0
    min_silence_samples = int(min_silence_duration * sr)  # Convert duration to samples

    for start, end in non_silent_intervals:
        # Append non-silent audio
        non_silent_audio.append(y[start:end])

        # Append silent audio only if it exceeds the minimum duration
        if last_end < start and (start - last_end) >= min_silence_samples:
            silent_audio.append(y[last_end:start])
        last_end = end

    # Handle trailing silence
    if last_end < len(y) and (len(y) - last_end) >= min_silence_samples:
        silent_audio.append(y[last_end:])

    # Concatenate non-silent segments into one array
    combined_non_silent_audio = np.concatenate(non_silent_audio)
    processed_non_silent_audio = process_non_silent_clip(combined_non_silent_audio,sr)

    # Create a warning which goes off if we are potentially removing clips from the middle of the audio
    if len(silent_audio) != 2:
        clip_lengths = [len(clip) / sr for clip in silent_audio]
        print(f"WARNING: {audio_path} had {len(silent_audio)} silent clips. Durations: {clip_lengths}")
    
    if len(silent_audio) > 1:
        combined_silent_audio = np.concatenate(silent_audio)
        processed_silent_audio = process_silent_clip(combined_silent_audio,sr)
    else:
        processed_silent_audio = None

    return processed_non_silent_audio, processed_silent_audio, sr


if __name__ == "__main__":
    audio_path = "test.wav"

    # Call the function to remove silence
    non_silent_audio, silent_audio, sr = remove_silence(audio_path, silence_thresh=-40, min_silence_duration=0.5)
    print(f"Non-Silent Shape: {non_silent_audio.shape} ({len(non_silent_audio)/sr:04f}s)")
    print(f"Silent Shape: {silent_audio.shape} ({len(silent_audio)/sr:04f}s)")

    # Save the non-silent audio
    sf.write("non_silent_audio.wav", non_silent_audio, sr)

    # Save the combined silent clips
    sf.write("silent_audio.wav", silent_audio, sr)

