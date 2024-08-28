from mne import time_frequency
import numpy as np
from typing import List, Tuple


def compute_relative_power(
    x: np.ndarray,
    psd_method: str = "multitaper",
    fmax: int = 150,
    s_freq: int = 1000,
    w_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the relative average power spectrum.
    (Mendoza-Halliday, D., 2022).

    Args:
        x (np.ndarray): signal.
        psd_method (str, optional): Method to compute power spectral density (PSD).
                                    Defaults to "multitaper".
        fmax (int, optional): max frequency in the PSD. Defaults to 150.
        s_freq (int, optional): Sampling frequency. Defaults to 1000.[Hz]
        w_size (int, optional): Length of each Welch segment. Defaults to 200.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - avg_psd: average power spectrum.
            - freqs: the frequencies.
    """
    if psd_method == "multitaper":
        psd, freqs = time_frequency.psd_array_multitaper(x, fmax=fmax, sfreq=s_freq)
    elif psd_method == "welch":
        psd, freqs = time_frequency.psd_array_welch(
            x, fmax=fmax, sfreq=s_freq, n_per_seg=w_size
        )
    else:
        raise ValueError("psd_method must be multitaper or welch")
    # Relative power (RP)
    # Compute the average power spectrum for each channel (avg across trials)
    avg_psd = psd.mean(axis=(0))
    # Compute the RP
    avg_psd = avg_psd / np.max(avg_psd, axis=0)

    return avg_psd, freqs
