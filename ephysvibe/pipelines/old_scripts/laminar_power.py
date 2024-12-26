import numpy as np
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from ..task import task_constants
from ..structures.trials_data import TrialsData
from ..analysis import signal
import logging
import os


import seaborn as sns

from mne import time_frequency, create_info, EpochsArray

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    path: Path,
    output_dir: Path,
    block: int = 1,
):
    logging.info("--- Start ---")

    if not os.path.exists(path):
        raise FileExistsError
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    s_path = os.path.normpath(path).split(os.sep)
    ss_path = s_path[-1][:-3]
    # Load data
    logging.info("Loading, %s" % (ss_path))
    data = TrialsData.from_python_hdf5(path)
    # Select correct trials in a specific block
    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == block))[0]
    # Align trials to stimulus presentation
    if block == 1:
        event = task_constants.EVENTS_B1["sample_on"]
    else:
        event = task_constants.EVENTS_B2["target_on"]
    trials_s_on = data.code_samples[
        trial_idx,
        np.where(data.code_numbers[trial_idx] == event)[1],
    ]
    t_before = 500
    shifts = -(trials_s_on - t_before).astype(int)
    shifts = shifts[:, np.newaxis]
    shift_lfp = TrialsData.indep_roll(data.lfp_values[trial_idx], shifts, axis=2)
    # Compute CSD
    inter_channel_distance = 50

    n_channels = shift_lfp.shape[1]
    fmax = 150
    s_freq = 1000
    w_size = 200
    x = shift_lfp[:, :, :1000]

    avg_psd, freqs = signal.compute_relative_power(
        x, psd_method="multitaper", fmax=fmax, s_freq=s_freq, w_size=w_size
    )

    # compute relative power maps
    rp_alpha_beta = avg_psd[:, np.where(np.logical_and(freqs > 10, freqs < 30))[0]]
    rp_gamma = avg_psd[:, np.where(np.logical_and(freqs > 50, freqs < 150))[0]]

    max_depth = (
        inter_channel_distance * (n_channels - data.clusters_ch[0])
        + data.clusterdepth[0]
    )
    ch_depth = np.concatenate(
        [
            np.arange(
                inter_channel_distance, data.clusterdepth[0], inter_channel_distance
            ),
            np.arange(data.clusterdepth[0], max_depth, inter_channel_distance),
        ]
    )
    # Plot and save relative power
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(avg_psd, cmap="viridis", ax=ax)
    ax.set_title("Relative power")
    ax.set(xlabel="Frequency (Hz)", ylabel="Laminar Depth (um)")
    ax.set_yticks(np.flip(np.arange(0.5, n_channels + 0.5)))
    ax.set_yticklabels(ch_depth.astype(int), rotation=0)
    ax.invert_yaxis()
    logging.info("Saving Relative power figure")
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["power_" + ss_path + ".jpg"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    # Plot and save relative power avg
    fig, ax = plt.subplots(figsize=(5, 5))
    a = ax.plot(
        rp_alpha_beta.mean(axis=1),
        np.arange(0, 32),
        label="alpha_beta (10-30)",
        color="teal",
    )
    a = ax.plot(
        rp_gamma.mean(axis=1), np.arange(0, 32), label="gamma (50-150)", color="tomato"
    )
    a = ax.set_yticks(np.arange(0, 32))
    ax.legend(
        fontsize=9, columnspacing=0.5, facecolor="white", framealpha=1, loc="upper left"
    )
    ax.set_title("Relative power averaged")
    ax.set(xlabel="Relative power", ylabel="Laminar Depth (um)")
    ax.set_yticks(np.flip(np.arange(0, n_channels)))
    a = ax.set_yticklabels(ch_depth.astype(int), rotation=0)
    logging.info("Saving Relative power averaged figure")
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["power_avg" + ss_path + ".jpg"]),
        bbox_inches="tight",
    )
    plt.close(fig)
    logging.info("-- end --")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Path to TrialsData file", type=Path)

    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--block", "-b", default=1, help="block 1 or 2", type=int)

    args = parser.parse_args()
    try:
        main(
            args.path,
            args.output_dir,
            args.block,
        )
    except FileExistsError:
        logging.error("filepath does not exist")
