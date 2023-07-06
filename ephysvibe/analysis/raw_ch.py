import numpy as np
from phylib.io.traces import get_ephys_reader
from scipy.signal import butter, sosfilt
import logging


def rolling_window(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """_summary_

    Args:
        x (np.ndarray): _description_
        window (int): _description_
        step (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    # TODO: move to another script
    shape = x.shape[:-1] + ((x.shape[-1] - window + 1) // step, window)
    strides = (x.strides[0] * step,) + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def detect_spikes(x: np.ndarray, win: int = 1000) -> np.ndarray:
    """Detect spikes using a threshold.
    Threshold is defined as 3 times the st deviation of x in a window.

    Args:
        x (np.ndarray): local field potentials.
        win (int, optional): window size for computing the std. Defaults to 1000.

    Returns:
        np.ndarray: array with shape (n_channels, n_time_pts - win_size) containing:
                    - 1 if spike
                    - 0 otherwise
    """
    shape_0, shape_1 = x.shape
    sp_seg = np.zeros((shape_0, shape_1 - win))
    for ch in range(shape_0):  # iterate by channel
        # std in sliding win
        std = np.std(rolling_window(x[ch], window=win, step=1), axis=1)
        # find values > than 3 times std
        idx = (x[ch, : len(std)] >= 3 * std).astype(int)  #
        idx[0] = 0
        # find intervals with values above threshold
        diff = np.diff(idx)
        diff_start = np.where(diff == 1)[0]
        diff_end = np.where(diff == -1)[0]
        sp_ch_seg = []
        for i_start, i_end in zip(diff_start, diff_end):
            # select values between start and end of values above the threshold
            sp_idx = x[ch, i_start : i_end + 1]
            # the spike is the larger value among the selected points
            sp_ch_seg.append(np.argmax(sp_idx) + i_start)
        sp_seg[ch][sp_ch_seg] = 1  # fill in with one where there are spikes
    return sp_seg


def load_continuous_dat(obj, **kwargs):
    """load continuous.dat.
    Parameters
    ----------
    obj : str or Path
        Path to the raw data file.
    sample_rate : float
        The data sampling rate, in Hz.
    n_channels_dat : int
        The number of columns in the raw data file.
    dtype : str
        The NumPy data type of the raw binary file.
    offset : int
        The header offset in bytes.
    """
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ("sample_rate", "n_channels_dat", "dtype", "offset")
    }

    traces = get_ephys_reader(obj, **kwargs)

    return traces


def filter_continuous(x, fs, fc_hp=None, fc_lp=None, axis=1):
    """Basic filtering of the continuous LFP before cutting it into epochs.
    The signal is expected to be already low-pass filtered at 250Hz
    """
    # fs_lfp = 30000  # sampling frequency of the signal in Hz
    # fc_hp = 250  # cut-off frequency of the high pass filter in Hz

    # perform a High pass filter - butterworth filter of order 6 at 1Hz
    if not (fc_hp is None):
        sos = butter(6, fc_hp, "hp", fs=fs, output="sos")
    else:
        sos = butter(6, fc_lp, "lp", fs=fs, output="sos")
    x_hp = sosfilt(sos, x)

    return x_hp
