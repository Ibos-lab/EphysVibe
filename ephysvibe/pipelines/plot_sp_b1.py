# plot spiking activity task b1
import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
from ..trials.spikes import firing_rate, sp_constants
from ..task import def_task
from ..spike_sorting import config
from ..task import task_constants
import warnings
from matplotlib import pyplot as plt
from ..structures.trials_data import TrialsData

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(filepath: Path, output_dir: Path, e_align: str, t_before: int):
    """Compute and plot firing rate during task b1.

    Args:
        filepath (Path): path to the sorted file (.npy).
        output_dir (Path): output directory.
        in_out (int): 1 for trials with stimuli in, -1 out.
        e_align (int): event to which align the spikes.
        cgroup (str): "good" for individual units, "mua" for multiunits.
    """
    s_path = os.path.normpath(filepath).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-2]])

    # check if filepath exist
    if not os.path.exists(filepath):
        raise FileExistsError
    logging.info("-- Start --")
    data = TrialsData.from_python_hdf5(filepath)
    # Select trials and create task frame
    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == 1))[0]
    task = def_task.create_task_frame(
        condition=data.condition[trial_idx],
        test_stimuli=data.test_stimuli[trial_idx],
        samples_cond=task_constants.SAMPLES_COND,
        neuron_cond=data.neuron_cond,
    )
    logging.info("Number of clusters: %d" % len(data.clustersgroup))
    # define kernel for convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    # select the trials
    trials_sp = data.sp_samples[trial_idx]
    trials_s_on = data.code_samples[
        trial_idx,
        np.where(data.code_numbers[trial_idx] == task_constants.EVENTS_B1[e_align])[1],
    ]
    trials_s_off = data.code_samples[
        trial_idx,
        np.where(
            data.code_numbers[trial_idx] == task_constants.EVENTS_B1["sample_off"]
        )[1],
    ]
    trials_t_on_1 = data.code_samples[
        trial_idx,
        np.where(data.code_numbers[trial_idx] == task_constants.EVENTS_B1["test_on_1"])[
            1
        ],
    ]
    mean_s_off = round((trials_s_off - trials_s_on).mean())
    mean_t_on_1 = round((trials_t_on_1 - trials_s_on).mean())
    samples = np.sort(np.unique(task["sample"].values))
    # plot fr for each neuron
    i_neuron, i_mua = 1, 1
    for i_n, cluster in enumerate(data.clustersgroup):
        if cluster == "good":
            i_cluster = i_neuron
            i_neuron += 1
            cluster = "neuron"
        else:
            i_cluster = i_mua
            i_mua += 1
        neuron_sp = trials_sp[:, i_n, :]
        shift_sp = TrialsData.indep_roll(
            neuron_sp, -(trials_s_on - t_before).astype(int), axis=1
        )[:, :1600]
        # Iterate by sample and condition
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
        ax2 = [ax[0].twinx(), ax[1].twinx()]
        all_max_conv = 0
        all_max_trial = 0
        for i_ax, cond in enumerate(["in", "out"]):
            count_trials = 0
            max_conv = 0
            for i_sample in samples:
                sample_idx = task[
                    np.logical_and(
                        task["i_neuron"] == i_neuron,
                        np.logical_and(
                            task["in_out"] == cond, task["sample"] == i_sample
                        ),
                    )
                ]["trial_idx"].values
                mean_sp = shift_sp[sample_idx].mean(axis=0)
                conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
                max_conv = np.max(conv) if np.max(conv) > max_conv else max_conv
                time = np.arange(0, len(conv)) - t_before
                ax[i_ax].plot(time, conv, color=task_constants.PALETTE_B1[i_sample])
                # Plot spikes
                count_t = len(sample_idx)
                rows, cols = np.where(shift_sp[sample_idx] == 1)
                ax2[i_ax].scatter(
                    cols - t_before,
                    rows + count_trials,
                    marker=2,
                    linewidths=0.5,
                    alpha=1,
                    edgecolors="none",
                    color=task_constants.PALETTE_B1[i_sample],
                    label="Sample %s" % i_sample,
                )
                count_trials += count_t
            all_max_conv = max_conv if max_conv > all_max_conv else all_max_conv
            all_max_trial = (
                count_trials if count_trials > all_max_trial else all_max_trial
            )
            ax[i_ax].set_title(cond)
        for i_ax in range(2):
            ax[i_ax].set_ylim(0, all_max_conv + all_max_trial + 5)
            ax[i_ax].set_yticks(np.arange(0, all_max_conv + 5, 10))
            ax2[i_ax].set_yticks(np.arange(-all_max_conv - 5, all_max_trial))
            plt.setp(ax2[i_ax].get_yticklabels(), visible=False)
            ax[i_ax].vlines(
                [0, mean_s_off, mean_t_on_1],
                0,
                all_max_conv + all_max_trial + 5,
                color="k",
                linestyles="dashed",
            )
        ax[0].set(xlabel="Time (s)", ylabel="Average firing rate")
        ax2[1].set(xlabel="Time (s)", ylabel="trials")
        ax2[1].legend(
            fontsize=9,
            scatterpoints=5,
            columnspacing=0.5,
            facecolor="white",
            framealpha=1,
            loc="upper right",
        )
        fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)
        fig.text(
            0.5,
            0.99,
            s="%s - Aligned with %s" % (ss_path[:10], e_align),
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.suptitle("%s: %s %d" % (s_path[-2], cluster, i_cluster), x=0.05, y=0.99)
        logging.info("Saving figure, %s: %d" % (cluster, i_cluster))
        fig.savefig(
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
    parser.add_argument("filepath", help="Path to the sorted file (.npy)", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument(
        "--e_align",
        "-e",
        default="sample_on",
        help="Event to aligne the spikes",
        type=str,
    )
    parser.add_argument(
        "--t_before",
        "-t",
        default=200,
        help="Time before e_aligne",
        type=int,
    )
    args = parser.parse_args()
    try:
        main(args.filepath, args.output_dir, args.e_align, args.t_before)
    except FileExistsError:
        logging.error("filepath does not exist")
