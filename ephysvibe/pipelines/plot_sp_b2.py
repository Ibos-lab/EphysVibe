import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from ..trials.spikes import firing_rate, sp_constants
from ..spike_sorting import config
from ..task import task_constants
from ephysvibe.structures.trials_data import TrialsData
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(filepath: Path, output_dir: Path, e_align: str, t_before: int):
    s_path = os.path.normpath(filepath).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-2]])
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if filepath exist
    if not os.path.exists(filepath):
        raise FileExistsError
    logging.info("-- Start --")
    data = TrialsData.from_python_hdf5(filepath)
    # Select trials and create task frame
    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == 2))[0]
    logging.info("Number of clusters: %d" % len(data.clustersgroup))
    # Define target codes
    target_codes = {
        # code: [ML axis], [plot axis]
        "127": [[10, 0], [1, 2]],
        "126": [[7, 7], [0, 2]],
        "125": [[0, 10], [0, 1]],
        "124": [[-7, 7], [0, 0]],
        "123": [[-10, 0], [1, 0]],
        "122": [[-7, -7], [2, 0]],
        "121": [[0, -10], [2, 1]],
        "120": [[7, -7], [2, 2]],
    }
    # create dict with the trials that have each code
    trials_idx = {}
    for i_key, key in enumerate(target_codes.keys()):
        trials = []
        code_idx = []
        for i_trial, code in zip(trial_idx, data.code_numbers[trial_idx]):
            idx = np.where(int(key) == code)[0]
            if len(idx) != 0:
                code_idx.append(idx[0])
                trials.append(i_trial)
        trials_idx[key] = {"code_idx": code_idx, "trials_idx": trials}
    # kernel parameters
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    # select only individual neurons
    i_neuron, i_mua = 1, 1
    for i_n, cluster in enumerate(data.clustersgroup):
        if cluster == "good":
            i_cluster = i_neuron
            i_neuron += 1
            cluster = "neuron"
        else:
            i_cluster = i_mua
            i_mua += 1
        fig, _ = plt.subplots(figsize=(10, 10), sharex=True, sharey=True)
        all_ax, all_ax2 = [], []
        all_max_conv, max_num_trials = 0, 0
        for code in target_codes.keys():
            target_t_idx = trials_idx[code][
                "trials_idx"
            ]  # select trials with the same stimulus
            trials_s_on = data.code_samples[
                target_t_idx,
                np.where(
                    data.code_numbers[target_t_idx] == task_constants.EVENTS_B2[e_align]
                )[1],
            ]
            shift_sp = TrialsData.indep_roll(
                data.sp_samples[target_t_idx, i_n],
                -(trials_s_on - t_before).astype(int),
                axis=1,
            )[:, :2300]
            mean_sp = shift_sp.mean(axis=0)
            conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
            conv_max = max(conv)
            all_max_conv = conv_max if conv_max > all_max_conv else all_max_conv
            num_trials = shift_sp.shape[0]
            max_num_trials = (
                num_trials if num_trials > max_num_trials else max_num_trials
            )
            axis = target_codes[code][1]
            ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
            time = np.arange(0, len(conv)) - t_before
            # ----- plot ----------
            ax2 = ax.twinx()
            ax.plot(time, conv)

            num_trials = shift_sp.shape[0]

            rows, cols = np.where(shift_sp >= 1)
            cols = cols - t_before
            rows = rows + rows * 2
            ax2.scatter(cols, rows, marker="|", alpha=1, color="grey")
            ax.set_title("Code %s" % (code), fontsize=8)
            all_ax.append(ax)
            all_ax2.append(ax2)
        avg_events = []
        for event in ["target_on", "target_off", "fix_spot_off", "correct_response"]:
            trials_event = data.code_samples[
                target_t_idx,
                np.where(
                    data.code_numbers[target_t_idx] == task_constants.EVENTS_B2[event]
                )[1],
            ]
            avg_events.append(np.mean(trials_event - trials_s_on))
        num_trials = shift_sp.shape[0]
        for ax, ax2 in zip(all_ax, all_ax2):
            for ev in avg_events:
                ax.vlines(
                    ev,
                    0,
                    all_max_conv + max_num_trials * 3,
                    color="k",
                    linestyles="dashed",
                )  # target_on
            ax.set_ylim(0, all_max_conv + max_num_trials * 3)
            ax.set_yticks(np.arange(0, all_max_conv, 10))
            ax2.set_ylim(-all_max_conv, max_num_trials)
            ax2.set_yticks(np.arange(-all_max_conv, max_num_trials * 3, 10))
            ax.set(xlabel="Time (s)", ylabel="Average firing rate")
            ax2.set(xlabel="Time (s)", ylabel="trials")
            plt.setp(ax2.get_yticklabels(), visible=False)

        fig.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
        fig.suptitle("%s: %s %d" % (s_path[-2], cluster, i_cluster), x=0)
        fig.text(
            0.5,
            0.5,
            s="Aligned with %s" % e_align,
            horizontalalignment="center",
            verticalalignment="center",
        )
        # ----- end plot ----
        logging.info("Saving figure, %s: %d" % (cluster, i_cluster))
        plt.savefig(
            "/".join(
                [os.path.normpath(output_dir)]
                + [ss_path + "_" + cluster + "_" + str(i_cluster) + ".jpg"]
            ),
            bbox_inches="tight",
        )

    logging.info("-- end --")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file_path", help="Path to the continuous file (.dat)", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument(
        "--e_align",
        "-e",
        default="target_on",
        help="Event to aligne the spikes",
        type=str,
    )
    parser.add_argument(
        "--t_before",
        "-t",
        default=500,
        help="Time before e_aligne",
        type=int,
    )
    args = parser.parse_args()
    try:
        main(args.file_path, args.output_dir, args.e_align, args.t_before)
    except FileExistsError:
        logging.error("filepath does not exist")
