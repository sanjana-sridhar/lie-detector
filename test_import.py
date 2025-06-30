import librosa
import numpy as np

def analyze_baseline(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)

    avg_pitch = np.mean(pitch_values) if pitch_values else 0
    print(f"🎤 Your baseline pitch is: {avg_pitch:.2f} Hz")
    return avg_pitch

# Replace this with your actual file path
baseline_pitch = analyze_baseline("/Users/sanjana/PycharmProjects/baseline2.wav")

