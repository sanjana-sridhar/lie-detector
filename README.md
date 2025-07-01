# Stress Voice Analyzer

A Python-based voice analysis tool that compares your speaking voice to a calm baseline to detect signs of stress or possible deception. It uses pitch, energy, and zero-crossing rate (ZCR) to flag irregularities in speech.

## Features

- **Baseline Pitch Capture**
  - Users first input a voice recording of their calm or normal speaking voice.
  - The program calculates and stores the average pitch as a reference.

- **Audio Analysis**
  - Users can upload new recordings for analysis.
  - The app measures:
    - **Pitch** (average frequency)
    - **Energy** (volume/loudness)
    - **Zero-Crossing Rate** (waviness or stress)

- **Stress Detection**
  - If pitch increases by more than 30 Hz,
  - or energy > 0.1,
  - or ZCR > 0.1,  
    ➝ the app prints a warning for stress or deception.

- **Looping Menu**
  - After each analysis, users can choose to test another recording or exit the app.

## What I Learned

- How to extract pitch and audio features using `librosa`
- Basic audio normalization and signal processing
- Using conditionals and thresholds to simulate detection logic
- Structuring programs with reusable functions and user prompts
- Building console-based tools with a clean looped interface

## Future Improvements

- Add microphone input for live analysis
- Train a machine learning model for better accuracy
- Visualize waveform and pitch graphically with `matplotlib` or a GUI

## License

This project is licensed under the [MIT License](LICENSE).

## Requirements 

- Python 3.x
- The following Python packages:

To make installation easier, you can create a `requirements.txt` file in your project root with the following:

librosa
numpy


Then install dependencies with:

```bash
pip install -r requirements.txt
```

## Flowchart

![App Flowchart](images/lie-detector-flowchart.png)
