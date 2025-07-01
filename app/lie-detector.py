import librosa  # For audio processing
import numpy as np  # For numerical calculations

# This stores the baseline pitch of your calm voice
baseline_pitch = None

# Function to calculate average pitch (frequency) of an audio signal
def get_average_pitch(y, sr):
    try:
        # librosa.yin estimates pitch for each frame; we take the average pitch overall
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        return np.mean(f0)
    except:
        # Return 0 if pitch can't be calculated
        return 0

# Function to calculate the 'energy' of the audio signal (loudness proxy)
def get_energy(y):
    # Sum of absolute amplitudes normalized by length
    return np.sum(np.abs(y)) / len(y)

# Analyze a single audio file for pitch, energy, and zero crossing rate (ZCR)
def analyze_audio(audio_path):
    global baseline_pitch

    # Load audio file; y = audio samples, sr = sample rate
    y, sr = librosa.load(audio_path, sr=None)
    # Normalize audio so the loudest point is 1 (prevents scale issues)
    y = y / np.max(np.abs(y))

    avg_pitch = get_average_pitch(y, sr)
    energy = get_energy(y)
    # ZCR roughly measures how often the sound waveform crosses zero; linked to noisiness or stress
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    print("\n--- Audio Analysis ---")
    print(f"Average Pitch:            {avg_pitch:.2f} Hz")
    pitch_diff = avg_pitch - baseline_pitch
    print(f"Baseline Pitch:           {baseline_pitch:.2f} Hz")
    print(f"Pitch Change:             {pitch_diff:+.2f} Hz")
    print(f"Energy:                   {energy:.4f}")
    print(f"Zero Crossing Rate:       {zcr:.4f}")

    # Simple stress detection logic based on pitch change, energy, and ZCR
    stressed = False
    # Determine if pitch suggests stress
    stressed = (avg_pitch - baseline_pitch > 30)

    # Evaluate each indicator
    stress_signs = 0

    if stressed:
        stress_signs += 1
    if energy > 0.1:
        stress_signs += 1
    if zcr > 0.1:
        stress_signs += 1

    # Final decision based on 2 or more stress indicators
    if stress_signs >= 2:
        print("⚠️  Stress/Lie suspected!")
    else:
        print("✅ No significant stress indicators.")


# Ask user to provide baseline audio to establish calm voice pitch
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

# Main program interaction loop
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





