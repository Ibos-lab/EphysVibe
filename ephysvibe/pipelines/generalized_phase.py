import numpy as np

import argparse
from pathlib import Path
from ..task import task_constants
from ..structures.trials_data import TrialsData
from ..analysis import signal
import logging
import os

from ephysvibe.task import task_constants, def_task
from ephysvibe.spike_sorting import utils_oe

from numpy.core.umath import (
    pi,
    add,
    arctan2,
    frompyfunc,
    cos,
    less_equal,
    sqrt,
    sin,
    mod,
    exp,
    not_equal,
    subtract,
)
import matplotlib.pyplot as plt

from scipy.signal import hilbert, butter, sosfilt, filtfilt

from scipy.fft import fft, fftfreq, ifft
import numpy.core.numeric as _nx
from mne import time_frequency, create_info, EpochsArray
from phylib.io.traces import get_ephys_reader

import seaborn as sns

from mne import time_frequency, create_info, EpochsArray

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


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


def main(
    path: Path,
    output_dir: Path,
):
    logging.info("--- Start ---")

    if not os.path.exists(path):
        raise FileExistsError
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    s_path = os.path.normpath(path).split(os.sep)
    ss_path = s_path[-1][:-3]

    continuous_path = "/envau/work/invibe/USERS/IBOS/openephys/Riesling/2023-03-17_10-11-51/Record Node 102/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/continuous.dat"
    c_samples = np.load(
        "/envau/work/invibe/USERS/IBOS/openephys/Riesling/2023-03-17_10-11-51/Record Node 102/experiment1/recording1/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy"
    )
    shape_0 = len(c_samples)
    # Load data
    logging.info("Loading, %s" % (ss_path))

    n_channels = 163
    sample_rate = 30000
    print(shape_0, n_channels)
    args = {"sample_rate": sample_rate, "n_channels_dat": n_channels, "dtype": "int16"}
    cont = load_continuous_dat(continuous_path, **args)
    # Parameters
    fs = 1000
    dt = 1 / fs
    order = 4
    lp_f = 50
    hp_f = 5
    passband = [hp_f / (fs / 2), lp_f / (fs / 2)]
    # Compute GP
    min_20 = 60 * 60 * 1000 * 30
    min_50 = 100 * 60 * 1000 * 30
    seg = np.arange(0, int((min_50 - min_20) / 30), 10000)
    seg_phase_spikes = []
    idx_ds = np.arange(0, 300000 - 10000, 30)
    n_channels = 32
    for i_seg in seg:
        # seg_lfp,seg_sp = mat_select_seg(all_lfp,all_sp,i_seg)
        # Select segment of the data

        # spike data
        seg_hp = filter_continuous(
            cont[i_seg : i_seg + 300000, :32], fs=30000, fc_hp=500, axis=0
        )  # High pass filter
        ## subsample
        # subsample = idx_ds[i_seg:i_seg+10000].tolist()
        seg_hp = seg_hp[idx_ds, :32].T
        ## detect spikes
        seg_sp = detect_spikes(seg_hp)

        # lfp data
        seg_lp = filter_continuous(
            cont[i_seg : i_seg + 300000, :n_channels], fs=30000, fc_lp=300, axis=0
        )  # Low pass filter
        b, a = butter(order, passband, "bandpass")
        x_lfp = filtfilt(
            b, a, seg_lp, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1), axis=0
        )
        x_lfp = x_lfp[idx_ds, :n_channels].T
        # compute gp
        xgp, wt = compute_generalized_phase(x_lfp, dt)
        phase_lfp = np.angle(xgp.astype(complex))

        # Spike phases
        phase_spikes = []  # np.zeros(n_channels,n_channels)

        for lfp_ch in range(0, n_channels):

            ch_phase_spikes = []
            for sp_ch in range(0, n_channels):

                if np.sum(seg_sp[sp_ch]) != 0:  # No spikes? No phase
                    ch_phase_spikes.append(
                        phase_lfp[lfp_ch, seg_sp[sp_ch].astype(bool)]
                    )
                else:
                    ch_phase_spikes.append([np.nan])
            phase_spikes.append(ch_phase_spikes)

        seg_phase_spikes.append(phase_spikes)
    # Reorganise list
    ps = []
    for lfp_ch in range(0, n_channels):
        ps_2 = []
        for sp_ch in range(0, n_channels):
            ps_1 = []
            for i_seg in np.arange(0, len(seg_phase_spikes)):
                ps_1.append(seg_phase_spikes[i_seg][lfp_ch][sp_ch])
            ps_2.append(np.concatenate(ps_1))
        ps.append(ps_2)
    # spike phase index (SPI): 1 all spikes occur at a single phase, 0 perfectly uniform spike-phase distribution
    spike_phase = np.full((n_channels, n_channels), np.nan)
    pref_phase = np.full((n_channels, n_channels), np.nan)
    phase_corr = np.full((n_channels, n_channels), np.nan)
    for lfp_ch in range(0, n_channels):
        for sp_ch in range(0, n_channels):
            # phase_sp = np.concatenate(ps[lfp_ch][sp_ch])
            phase_sp = ps[lfp_ch][sp_ch][~np.isnan(ps[lfp_ch][sp_ch])]
            spike_phase[lfp_ch, sp_ch] = resultant_vector_length(alpha=phase_sp)
            pref_phase[lfp_ch, sp_ch] = mean_direction(alpha=phase_sp)
            phase_corr[lfp_ch, sp_ch] = corrcc(phase_lfp[lfp_ch], phase_lfp[sp_ch])
    # Plot and save phase_corr
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(phase_corr)
    ax.invert_yaxis()
    ax.invert_xaxis()
    logging.info("Saving phase_corr figure")
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["phase_corr_" + ss_path + ".jpg"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    # Plot and save pref_phase
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pref_phase.T, cmap="viridis")
    ax.invert_yaxis()
    ax.invert_xaxis()
    logging.info("Saving pref_phase figure")
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["pref_phase_" + ss_path + ".jpg"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    # Plot and save
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(spike_phase.T, cmap="hot")
    ax.invert_yaxis()
    ax.invert_xaxis()
    logging.info("Saving spike_phase figure")
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["spike_phase_" + ss_path + ".jpg"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    logging.info("-- end --")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path to continuous file", type=Path)

    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )

    args = parser.parse_args()
    try:
        main(
            args.path,
            args.output_dir,
        )
    except FileExistsError:
        logging.error("filepath does not exist")
