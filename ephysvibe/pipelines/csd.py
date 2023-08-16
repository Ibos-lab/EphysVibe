import numpy as np
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from ..task import def_task, task_constants
from ..structures.bhv_data import BhvData
from ..structures.lfp_data import LfpData
from ..analysis import layers
from typing import Dict
import logging
import os
from ..spike_sorting import config

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
    data_lfp = LfpData.from_python_hdf5(path)
    #! add path bhv
    data_bhv = BhvData.from_python_hdf5(path)
    # Select correct trials in a specific block
    inside_rf = np.logical_or(
        np.logical_or(
            np.logical_or(data_bhv.sacc_code == 125, data_bhv.sacc_code == 126),
            data_bhv.sacc_code == 127,
        ),
        data_bhv.sacc_code == 120,
    )
    trial_idx = np.where(
        np.logical_and(
            np.logical_and(data_bhv.trial_error == 0, data_bhv.block == 2), inside_rf
        )
    )[0]
    # Align trials to stimulus presentation
    trials_s_on = data_bhv.code_samples[
        trial_idx,
        np.where(
            data_bhv.code_numbers[trial_idx] == task_constants.EVENTS_B2["target_on"]
        )[  #
            1
        ],
    ]
    t_before = 10
    shifts = -(trials_s_on - t_before).astype(int)
    shifts = shifts[:, np.newaxis]
    idx_start = data_lfp.idx_start[trial_idx] - t_before
    idx_start = (
        data_bhv.code_samples[trial_idx][
            data_bhv.code_numbers[trial_idx] == task_constants.EVENTS_B2["target_on"]
        ]
        + idx_start
    ).astype(int)
    idx_end = (
        data_bhv.code_samples[trial_idx][
            data_bhv.code_numbers[trial_idx] == config.END_CODE
        ]
        + idx_start
    ).astype(int)

    n_trials = len(idx_end)
    n_ch = 32
    max_len = int(max(idx_end - idx_start))
    lfp_trials = np.full((n_trials, n_ch, max_len), np.nan)
    for i_tr in range(n_trials):
        dur = int(idx_end[i_tr] - idx_start[i_tr])
        lfp_trials[i_tr, :, :dur] = data_lfp.lfp_values[
            :, idx_start[i_tr] : idx_end[i_tr]
        ]

    # Compute CSD
    inter_channel_distance = 50

    csd = layers.compute_csd(
        lfp_trials.mean(axis=0)[:, :1500], inter_channel_distance, step=step
    )

    # max_depth = (
    #     inter_channel_distance * (n_channels - data_lfp.clusters_ch[0])
    #     + data_lfp.clusterdepth[0]
    # )
    # ch_depth = np.concatenate(
    #     [
    #         np.arange(
    #             inter_channel_distance, data_lfp.clusterdepth[0], inter_channel_distance
    #         ),
    #         np.arange(data_lfp.clusterdepth[0], max_depth, inter_channel_distance),
    #     ]
    # )
    # save plot
    fig, ax = plt.subplots(figsize=(20, 5))
    start_plot = t_before - 50
    fig, ax = plt.subplots(figsize=(10, 5))
    start_plot = 0  # t_before-50
    sns.heatmap(csd[:, start_plot : start_plot + 500], cmap="viridis", ax=ax)
    fig.legend(fontsize=9, loc="upper center")
    ax.set_title("CSD")
    ax.set(xlabel="Time (ms)", ylabel="Channels")

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
