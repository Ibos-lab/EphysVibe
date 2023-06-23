import numpy as np
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from ..task import def_task, task_constants
from ..structures.trials_data import TrialsData
from ..analysis import layers
from typing import Dict
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
    step: int = 2,
):
    logging.info("--- Start ---")

    if not os.path.exists(path):
        raise FileExistsError
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Load data
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
    csd = layers.compute_csd(shift_lfp.mean(axis=0), inter_channel_distance, step=step)

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
    # save plot
    fig, ax = plt.subplots(figsize=(20, 5))
    start_plot = t_before - 50
    sns.heatmap(csd[:, start_plot : start_plot + 250], cmap="viridis", ax=ax)
    ax.vlines(
        [t_before - start_plot], 0, n_channels - step * 2, colors="r", label="sample_on"
    )
    fig.legend(fontsize=9, loc="upper center")
    ax.set_title("CSD")
    ax.set(xlabel="Time (ms)", ylabel="Channels")
    ax.set_yticks(np.arange(0.5, n_channels - 2 * step + 0.5))
    ax.set_yticklabels(ch_depth[step:-step].astype(int), rotation=0)
    s_path = os.path.normpath(path).split(os.sep)
    ss_path = s_path[-1][:-3]

    logging.info("Saving figure, %s" % (ss_path))
    fig.savefig(
        "/".join([os.path.normpath(output_dir)] + ["CSD_" + ss_path + ".jpg"]),
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
    parser.add_argument("--step", "-s", default=1, help="steps", type=int)

    args = parser.parse_args()
    try:
        main(
            args.path,
            args.output_dir,
            args.block,
            args.step,
        )
    except FileExistsError:
        logging.error("filepath does not exist")
