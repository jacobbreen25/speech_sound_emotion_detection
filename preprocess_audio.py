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

    testing = False
    if testing: 
        sf.write("silence_concatenated.wav",clip,sr)
        sf.write("silence_trimmed.wav",processed_clip,sr)

    return processed_clip

def process_non_silent_clip(clip, sr, clip_duration=1.0, step_size=1.0):
    testing = True

    clip_length = int(clip_duration * sr)  # Convert duration to samples
    processed_clips = []
    if len(clip) < clip_length:
        # Pad short clips with zeroes
        processed_clips.append(np.pad(clip, (0, clip_length - len(clip)), mode='constant'))
    else:
        start = 0
        end = clip_length
        while end < len(clip):
            if testing: 
                print(f"Start: {start} - End: {end}")
            processed_clips.append(clip[start:end])
            start += int(step_size * clip_length)
            end = start + clip_length
        # Add a clip from the very end of the audio to make sure everything is captured
        processed_clips.append(clip[-clip_length:])

    # For presentation, save individual clips
    if testing:
        i=1
        print(f"Clip len: {clip_length}")
        for c in processed_clips:
            print(f"> len - {len(c)}")
            sf.write(f"speech_{i}.wav",c,sr)
            i += 1
    return processed_clips
    
def preprocess_audio(audio_path, silence_thresh=-40, min_silence_duration=0.4):
    # Load the audio file
    y, sr = librosa.load(audio_path)

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
    audio_path = "03-01-03-02-02-02-03.wav"

    # Call the function to remove silence
    non_silent_audio, silent_audio, sr = preprocess_audio(audio_path, silence_thresh=-40, min_silence_duration=0.5)
    # print(f"Non-Silent Shape: {non_silent_audio.shape} ({len(non_silent_audio)/sr:04f}s)")
    # print(f"Silent Shape: {silent_audio.shape} ({len(silent_audio)/sr:04f}s)")

    # Save the non-silent audio
    # sf.write("non_silent_audio.wav", non_silent_audio, sr)

    # Save the combined silent clips
    # sf.write("silent_audio.wav", silent_audio, sr)

