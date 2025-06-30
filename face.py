import librosa
import numpy as np

# Global variable to store baseline pitch
baseline_pitch = None

def get_average_pitch(y, sr):
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        return np.mean(f0)
    except:
        return 0

def get_energy(y):
    return np.sum(np.abs(y)) / len(y)

def analyze_audio(audio_path):
    global baseline_pitch

    y, sr = librosa.load(audio_path, sr=None)
    y = y / np.max(np.abs(y))  # Normalize

    avg_pitch = get_average_pitch(y, sr)
    energy = get_energy(y)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    print("\n--- Audio Analysis ---")
    print(f"Average Pitch:            {avg_pitch:.2f} Hz")
    pitch_diff = avg_pitch - baseline_pitch
    print(f"Baseline Pitch:           {baseline_pitch:.2f} Hz")
    print(f"Pitch Change:             {pitch_diff:+.2f} Hz")
    print(f"Energy:                   {energy:.4f}")
    print(f"Zero Crossing Rate:       {zcr:.4f}")

    # Simple stress detection
    stressed = False
    if avg_pitch - baseline_pitch > 30:
        stressed = True

    if stressed or energy > 0.1 or zcr > 0.1:
        print("⚠️  Stress/Lie suspected!")
    else:
        print("✅ No significant stress indicators.")

def set_baseline():
    global baseline_pitch
    while True:
        audio_path = input("Enter path to your baseline (calm/normal voice) audio file: ")
        try:
            y, sr = librosa.load(audio_path, sr=None)
            y = y / np.max(np.abs(y))
            baseline_pitch = get_average_pitch(y, sr)
            print(f"\n✅ Baseline pitch set to: {baseline_pitch:.2f} Hz\n")
            break
        except Exception as e:
            print(f"Error loading baseline audio: {e}\nPlease try again.")

def main():
    print("🎙️ Welcome to the Stress Voice Analyzer!")
    print("Before you begin, let's capture your baseline voice.\n")
    set_baseline()

    while True:
        choice = input("Do you want to analyze a new audio? (yes/exit): ").strip().lower()
        if choice == "yes":
            audio_path = input("Enter path to the audio file: ")
            try:
                analyze_audio(audio_path)
            except Exception as e:
                print(f"Error analyzing audio: {e}")
        elif choice == "exit":
            print("Goodbye!")
            break
        else:
            print("Please enter 'yes' or 'exit'.")

main()




