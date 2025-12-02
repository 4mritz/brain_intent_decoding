# Brain Intent Decoding (Motor Imagery EEG)

This project implements an end-to-end pipeline for decoding brain intent
(left vs right hand motor imagery) from EEG signals using:

- Classical ML: CSP + LDA
- Deep Learning: EEGNet-style CNN in PyTorch

Dataset: PhysioNet EEG Motor Movement/Imagery (subject 1, runs 6 and 10).

## Pipeline

1. Load raw EEG using MNE + PhysioNet (`src/data_loading.py`)
2. Epoch signals around motor imagery events (T1/T2)
3. Bandpass filter 8–30 Hz and standardize (`src/preprocessing.py`)
4. Train CSP + LDA baseline (`src/classical_models.py`)
5. Train SimpleEEGNet deep model (`src/deep_models.py`)
6. Evaluate accuracy and visualize CSP patterns.

Both models currently achieve 100% test accuracy on subject 1.
