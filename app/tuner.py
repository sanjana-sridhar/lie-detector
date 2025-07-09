# tuner.py
import librosa
import numpy as np
import json
import os

# === Settings ===
FEATURES = ["pitch_diff", "energy", "zcr", "pitch_std", "energy_std", "voiced_ratio"]

# --- Function to extract features from an audio file ---
def extract_features(audio_path, baseline_pitch):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y = y / np.max(np.abs(y))
        intervals = librosa.effects.split(y, top_db=30)
        voiced = np.concatenate([y[start:end] for start, end in intervals])

        f0 = librosa.yin(voiced, fmin=50, fmax=500, sr=sr)
        avg_pitch = np.mean(f0)
        pitch_std = np.std(f0)

        rms = librosa.feature.rms(y=voiced)[0]
        energy = np.sum(np.abs(voiced)) / len(voiced)
        energy_std = np.std(rms)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=voiced))
        voiced_ratio = sum((end - start) for start, end in intervals) / len(y)

        return {
            "pitch_diff": avg_pitch - baseline_pitch,
            "pitch_std": pitch_std,
            "energy": energy,
            "energy_std": energy_std,
            "zcr": zcr,
            "voiced_ratio": voiced_ratio
        }
    except:
        return None

# --- Load baseline pitch from file ---
def get_baseline_pitch():
    try:
        with open("baseline.txt") as f:
            baseline_pitch = float(f.read().strip())
            print(f"📄 Loaded baseline pitch: {baseline_pitch:.2f} Hz")
            return baseline_pitch
    except Exception as e:
        print(f"❌ Could not load baseline: {e}")
        exit()

# --- Prompt user for truth or lie samples ---
def load_samples(label, baseline_pitch):
    features = []
    print(f"\nUpload your {label.upper()} samples. Type 'done' when finished.")
    while True:
        path = input(f"Enter path to {label} sample: ").strip()
        if path.lower() == "done":
            break
        elif os.path.exists(path):
            f = extract_features(path, baseline_pitch)
            if f:
                f["label"] = label
                features.append(f)
            else:
                print("⚠️ Failed to extract features.")
        else:
            print("⚠️ File not found.")
    return features

# --- Function to evaluate thresholds ---
def evaluate_thresholds(data, thresholds):
    correct = 0
    for sample in data:
        triggers = 0
        if sample["pitch_diff"] > thresholds["pitch_diff"]:
            triggers += 1
        if sample["energy"] > thresholds["energy"]:
            triggers += 1
        if sample["zcr"] > thresholds["zcr"]:
            triggers += 1
        if sample["pitch_std"] > thresholds["pitch_std"]:
            triggers += 1
        if sample["energy_std"] > thresholds["energy_std"]:
            triggers += 1
        if sample["voiced_ratio"] < thresholds["voiced_ratio"]:
            triggers += 1

        prediction = "lie" if triggers > 2 else "truth"
        if prediction == sample["label"]:
            correct += 1
    return correct / len(data)

# --- Try different combinations of thresholds ---
def tune_thresholds(data):
    best_accuracy = 0
    best_combo = {}

    for pd in range(20, 80, 5):
        for en in np.arange(0.1, 0.3, 0.02):
            for zc in np.arange(0.05, 0.2, 0.02):
                for ps in np.arange(2, 10, 1):
                    for es in np.arange(0.005, 0.04, 0.005):
                        for vr in np.arange(0.5, 0.95, 0.05):
                            thresholds = {
                                "pitch_diff": pd,
                                "energy": en,
                                "zcr": zc,
                                "pitch_std": ps,
                                "energy_std": es,
                                "voiced_ratio": vr
                            }
                            acc = evaluate_thresholds(data, thresholds)
                            if acc > best_accuracy:
                                best_accuracy = acc
                                best_combo = thresholds.copy()

    return best_combo, best_accuracy

# --- Save best thresholds to JSON (after converting types) ---
def save_thresholds(thresholds):
    # Convert all values to plain float (or int)
    clean_thresholds = {k: float(v) for k, v in thresholds.items()}
    with open("thresholds.json", "w") as f:
        json.dump(clean_thresholds, f, indent=4)
    print("✅ Saved best thresholds to thresholds.json")


# === MAIN SCRIPT ===
def main():
    print("🔧 Welcome to the Threshold Tuner!")
    baseline = get_baseline_pitch()
    print(f"Baseline pitch: {baseline:.2f} Hz")

    truth_data = load_samples("truth", baseline)
    lie_data = load_samples("lie", baseline)

    all_data = truth_data + lie_data
    if len(all_data) < 4:
        print("⚠️ Not enough samples to tune reliably.")
        return

    print("\n🎯 Tuning thresholds... This may take a minute...")
    best_thresholds, accuracy = tune_thresholds(all_data)
    print(f"\n📈 Best accuracy: {accuracy:.2%}")
    print("Best thresholds found:")
    for key, val in best_thresholds.items():
        print(f"  {key}: {val}")

    save_thresholds(best_thresholds)

if __name__ == "__main__":
    main()
