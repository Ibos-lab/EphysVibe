import numpy as np
import argparse
from pathlib import Path
from ..analysis import raw_ch, circular_stats
import logging
import os
from numpy.core.umath import pi, mod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, butter, sosfilt, filtfilt

from scipy.fft import fft, fftfreq, ifft
import numpy.core.numeric as _nx
import mne

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    path: Path,
    output_dir: Path,
):
    logging.info("--- Start ---")
    if not os.path.exists(path):
        raise FileExistsError
    s_path = os.path.normpath(path).split(os.sep)
    if "continuous.dat" != s_path[-1]:
        logging.error("Path should end with /continuous.dat")
        raise NameError
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ss_path = s_path[-7]
    samples_path = "/".join(s_path[:-1] + ["sample_numbers.npy"])
    logging.info("Loading samples_numbers.py")
    c_samples = np.load(samples_path)

    shape_0 = len(c_samples)
    # Load data
    n_channels = 163
    sample_rate = 30000
    print(shape_0, n_channels)
    args = {"sample_rate": sample_rate, "n_channels_dat": n_channels, "dtype": "int16"}
    logging.info("Loading %s" % (ss_path))
    cont = raw_ch.load_continuous_dat(path, **args)
    # Parameters
    fs = 1000
    dt = 1 / fs
    order = 4
    lp_f = 40
    hp_f = 5
    passband = [hp_f, lp_f]

    # Compute GP
    session_duration = (cont.shape[0] / 30000) / 60
    if session_duration <= 60:
        logging.error("Session duration should be at least 60 min")
        raise ValueError

    minute_min = 20 * 60 * 1000 * 30
    minute_max = minute_min + (40 * 60 * 1000 * 30)
    raw_step = 300000
    raw_1sec = 30000
    seg = np.arange(0, int((minute_max - minute_min)), raw_step) + minute_min
    seg_phase_spikes = []
    idx_ds_sp = np.arange(0, raw_step + raw_1sec, 30)
    idx_ds_lfp = np.arange(0, raw_step, 30)
    n_channels = 32

    logging.info("Computing gp")
    for i_seg in seg:
        seg_hp = cont[i_seg : i_seg + raw_step + raw_1sec, :n_channels].astype(float).T
        avg_lfp_ch = np.median(seg_hp, axis=1).reshape(-1, 1)
        seg_hp = seg_hp - avg_lfp_ch
        # avg_lfp = np.median(seg_hp, axis=1).reshape(-1, 1)
        seg_hp = mne.filter.filter_data(
            seg_hp, sfreq=30000, l_freq=500, h_freq=None, method="fir", verbose=False
        )  # High pass filter
        ## subsample
        seg_hp = seg_hp[:, idx_ds_sp]
        ## detect spikes
        seg_sp = raw_ch.detect_spikes(seg_hp, win=1000)
        # LFP
        seg_lp = (
            cont[i_seg : i_seg + raw_step, :n_channels].astype(float).T
        )  # Low pass filter
        avg_lfp_ch = np.median(seg_lp, axis=1).reshape(-1, 1)
        seg_lp = seg_lp - avg_lfp_ch
        seg_lp = mne.filter.filter_data(
            seg_lp, sfreq=30000, l_freq=None, h_freq=300, method="fir", verbose=False
        )
        seg_lp = seg_lp[:, idx_ds_lfp]
        b, a = butter(order, passband, "bandpass", fs=1000)
        x_lfp = filtfilt(
            b, a, seg_lp, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1), axis=1
        )
        # compute gp
        xgp, wt = circular_stats.compute_generalized_phase(x_lfp, dt)
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
    logging.info("Computing circular stats")
    for lfp_ch in range(0, n_channels):
        for sp_ch in range(0, n_channels):
            # phase_sp = np.concatenate(ps[lfp_ch][sp_ch])
            phase_sp = ps[lfp_ch][sp_ch][~np.isnan(ps[lfp_ch][sp_ch])]
            spike_phase[lfp_ch, sp_ch] = circular_stats.resultant_vector_length(
                alpha=phase_sp
            )
            pref_phase[lfp_ch, sp_ch] = circular_stats.mean_direction(alpha=phase_sp)
            phase_corr[lfp_ch, sp_ch] = circular_stats.corrcc(
                phase_lfp[lfp_ch], phase_lfp[sp_ch]
            )
    # Plot and save phase_corr
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(phase_corr)
    ax.invert_yaxis()
    ax.invert_xaxis()
    fig.suptitle("Phase correlation")
    ax.set(xlabel="Channel number", ylabel="Channel number")
    logging.info("Saving phase_corr figure")
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["phase_corr_" + ss_path + ".jpg"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    # Plot and save pref_phase
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pref_phase.T,
        cmap="RdBu_r",
        cbar_kws={"label": "Preferred phase (rad)"},
        vmin=-pi,
        vmax=pi,
    )
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
    sns.heatmap(spike_phase.T, cmap="hot", cbar_kws={"label": "SPI"})
    ax.invert_yaxis()
    ax.invert_xaxis()
    fig.suptitle("Spike phase index (SPI)")
    ax.set(xlabel="LFP channel", ylabel="MUA channel")
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
