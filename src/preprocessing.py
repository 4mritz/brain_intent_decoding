import numpy as np

def bandpass_filter(epochs, l_freq=8.0, h_freq=30.0):
    """Apply bandpass filter on MNE epochs; return filtered copy."""
    epochs_filt = epochs.copy().filter(l_freq, h_freq, fir_design="firwin")
    return epochs_filt

def standardize_epochs(X):
    """
    Standardize epochs per channel across time.
    X: (n_epochs, n_channels, n_times)
    """
    return (X - X.mean(axis=-1, keepdims=True)) / X.std(axis=-1, keepdims=True)
