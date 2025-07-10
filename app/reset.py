import os

# Files to delete
files_to_remove = ["baseline.txt", "thresholds.json"]

print("🧹 Resetting baseline and thresholds...\n")

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"✅ Deleted {file}")
    else:
        print(f"ℹ️ {file} not found (already deleted or never created)")

print("\n✨ Done! You can now re-run baseline.py and tuner.py to start fresh.")
