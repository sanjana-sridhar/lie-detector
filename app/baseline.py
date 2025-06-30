import librosa  # Library for audio analysis
import numpy as np  # Library for numerical operations

# Function to analyze the baseline pitch from a calm audio recording
def analyze_baseline(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)  # y = audio signal, sr = sample rate

    # Get pitch and magnitude values using the piptrack algorithm
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = []

    # Loop through each time frame (column in the pitch matrix)
    for i in range(pitches.shape[1]):
        # Find the index of the strongest frequency (loudest pitch) in that frame
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]

        # Ignore silence or invalid pitch values (which are 0)
        if pitch > 0:
            pitch_values.append(pitch)

    # Calculate the average pitch (only if valid pitches were found)
    avg_pitch = np.mean(pitch_values) if pitch_values else 0

    # Print the result to the user
    print(f"🎤 Your baseline pitch is: {avg_pitch:.2f} Hz")
    return avg_pitch

# Example usage:
# Analyze the pitch of a calm (baseline) voice recording
# Replace this with your actual audio file path
baseline_pitch = analyze_baseline("/Users/sanjana/PycharmProjects/baseline2.wav")


