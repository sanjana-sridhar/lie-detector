import librosa  # To load audio
import numpy as np  # For math calculations

# Function to calculate baseline pitch from user-selected audio
def analyze_baseline():
    while True:
        audio_path = input("🎙️ Enter path to your baseline (calm) voice audio file: ").strip()
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

            pitch_values = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitch_values.append(pitch)

            avg_pitch = np.mean(pitch_values) if pitch_values else 0
            print(f"\n✅ Your baseline pitch is: {avg_pitch:.2f} Hz")

            # Optional: save to file so the analyzer can access it
            with open("baseline.txt", "w") as f:
                f.write(str(avg_pitch))

            print("📄 Baseline pitch saved to 'baseline.txt'")
            return avg_pitch
        except Exception as e:
            print(f"❌ Error loading audio: {e}\nPlease try again.")

# Run it!
if __name__ == "__main__":
    analyze_baseline()
