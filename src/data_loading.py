import mne
import numpy as np
from mne.datasets import eegbci

def load_subject_physionet(subject=1, runs=(6, 10), data_path="../data/physionet"):
    """Load raw EEG for a given subject and runs."""
    files = eegbci.load_data(subject, runs, path=data_path)
    raw_list = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raw_list)
    return raw

def epochs_from_raw(raw, tmin=0.5, tmax=4.0):
    """Create epochs and return epochs + event_id."""
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
    )
    return epochs, event_id

def extract_motor_imagery(epochs, event_id):
    """Return X, y (only T1/T2) and binary labels y_bin (0/1)."""
    X = epochs.get_data()
    y = epochs.events[:, -1]

    mask = (y == event_id["T1"]) | (y == event_id["T2"])
    X = X[mask]
    y = y[mask]

    y_bin = (y == event_id["T2"]).astype(int)  # e.g. T1->0, T2->1
    return X, y_bin
