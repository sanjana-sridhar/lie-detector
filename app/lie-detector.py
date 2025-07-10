import librosa
import numpy as np
import pandas as pd
import json
import os

# Load baseline pitch from file
with open("baseline.txt") as f:
    baseline_pitch = float(f.read().strip())

# Load thresholds from tuner
with open("thresholds.json") as f:
    thresholds = json.load(f)

# CSV path to store results
csv_path = "voice_analysis_log.csv"

# Define expected columns
columns = [
    "filename", "avg_pitch", "pitch_change", "pitch_std", "energy",
    "energy_std", "zcr", "voiced_ratio", "triggers", "verdict"
]

# Load CSV if it exists and has content, otherwise create new
try:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        log_df = pd.read_csv(csv_path)
        # Check if required columns exist
        if not set(columns).issubset(log_df.columns):
            raise ValueError("Missing columns in CSV. Reinitializing log.")
    else:
        raise FileNotFoundError
except (pd.errors.EmptyDataError, FileNotFoundError, ValueError):
    # If file is missing, empty, or invalid, create a new one
    log_df = pd.DataFrame(columns=columns)
    print("🆕 New CSV log initialized.")


# Function to calculate average energy (loudness)
def get_energy(y):
    return np.sum(np.abs(y)) / len(y)

# Main function to analyze a new audio file
def analyze_audio(audio_path):
    global log_df

    # Load and normalize audio
    y, sr = librosa.load(audio_path, sr=None)
    y = y / np.max(np.abs(y))

    # Extract only voiced (speaking) sections
    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) == 0:
        print("⚠️ No speech detected.")
        return

    voiced = np.concatenate([y[start:end] for start, end in intervals])
    if len(voiced) == 0:
        print("⚠️ Voiced segment extraction failed.")
        return

    # Calculate vocal features
    f0 = librosa.yin(voiced, fmin=50, fmax=500, sr=sr)
    avg_pitch = np.mean(f0)
    pitch_std = np.std(f0)
    pitch_change = avg_pitch - baseline_pitch

    energy = get_energy(voiced)
    rms = librosa.feature.rms(y=voiced)[0]
    energy_std = np.std(rms)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=voiced))
    voiced_ratio = sum((end - start) for start, end in intervals) / len(y)

    # Show results
    print("\n--- Audio Analysis ---")
    print(f"Average Pitch:         {avg_pitch:.2f} Hz")
    print(f"Pitch Change:          {pitch_change:.2f} Hz")
    print(f"Pitch Std (jitter):    {pitch_std:.2f}")
    print(f"Energy:                {energy:.4f}")
    print(f"Energy Std (shimmer):  {energy_std:.4f}")
    print(f"Zero Crossing Rate:    {zcr:.4f}")
    print(f"Voiced Ratio (0-1):    {voiced_ratio:.2f}")

    # Check against thresholds
    triggers = []

    if pitch_change > thresholds["pitch_diff"]:
        triggers.append("🔺 High pitch deviation")
    if energy > thresholds["energy"]:
        triggers.append("⚡ High energy")
    if pitch_std > thresholds["pitch_std"]:
        triggers.append("🎤 High pitch variability (jitter)")
    if energy_std > thresholds["energy_std"]:
        triggers.append("🔊 Energy variability (shimmer)")
    if zcr > thresholds["zcr"]:
        triggers.append("🎯 High zero-crossing rate")
    if voiced_ratio < thresholds["voiced_ratio"]:
        triggers.append("⏸️ Too many pauses (low voiced ratio)")

    # Decide verdict
    print("\n--- Triggers ---")
    if triggers:
        for t in triggers:
            print(f"❗ {t}")
    else:
        print("✅ No stress features triggered.")

    if len(triggers) >= 3:
        verdict = "Lie"
        print("\n⚠️ Verdict: STRESSED or LYING (3+ triggers)\n")
    else:
        verdict = "Truth"
        print("\n✅ Verdict: Likely Truthful (fewer than 3 triggers)\n")

    # Log results
    log_df.loc[len(log_df)] = {
        "filename": audio_path,
        "avg_pitch": avg_pitch,
        "pitch_change": pitch_change,
        "pitch_std": pitch_std,
        "energy": energy,
        "energy_std": energy_std,
        "zcr": zcr,
        "voiced_ratio": voiced_ratio,
        "triggers": "\n".join(triggers),
        "verdict": verdict
    }

    log_df.to_csv(csv_path, index=False)
    print(f"📄 Logged result to {csv_path}")

# Main loop to analyze multiple files
def main():
    print("🎤 Voice Analyzer Ready!")
    while True:
        choice = input("Analyze a new audio file? (yes/exit): ").strip().lower()
        if choice == "yes":
            audio_path = input("Enter path to audio file: ").strip()
            try:
                analyze_audio(audio_path)
            except Exception as e:
                print(f"❌ Error analyzing file: {e}")
        elif choice == "exit":
            print("👋 Exiting analyzer.")
            break
        else:
            print("Please type 'yes' or 'exit'.")

# Run the program
if __name__ == "__main__":
    main()
