# plot spiking activity task b1
import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
from ..trials import select_trials
from ..trials.spikes import firing_rate, sp_constants
from ..task import def_task
from ..spike_sorting import config
from ..task import task_constants
import warnings
from matplotlib import pyplot as plt
from ..structures.trials_data import TrialsData

warnings.filterwarnings("ignore")


def indep_roll(arr, shifts, axis=1):
    """Apply an independent roll for each dimensions of a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Array of any shape.

    shifts : np.ndarray
        How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.

    axis : int
        Axis along which elements are shifted.
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]

    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def main(
    filepath: Path,
    output_dir: Path,
    in_out: int,
    e_align: int,
    cgroup: str,
):
    """Compute and plot firing rate during task b1.

    Args:
        filepath (Path): path to the sorted file (.npy).
        output_dir (Path): output directory.
        in_out (int): 1 for trials with stimuli in, -1 out.
        e_align (int): event to which align the spikes.
        cgroup (str): "good" for individual units, "mua" for multiunits.
    """
    x_lim_max = 4
    x_lim_min = -0.7
    s_path = os.path.normpath(filepath).split(os.sep)
    ss_path = s_path[-1][:-4]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-2]])
    log_output = output_dir + "/" + ss_path + "_plot_sp_b1.log"
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        handlers=[logging.FileHandler(log_output), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s | %(message)s ",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
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
    )
    # fig_task, _ = def_task.info_task(task)
    neurons = np.where(data.clustersgroup == cgroup)[0]
    logging.info("Number of clusters: %d" % len(data.clustersgroup))
    logging.info("Number of %s units: %d" % (cgroup, len(neurons)))
    logging.info("in_out: %d" % in_out)
    # define kernel for convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    # select the trials
    trials_sp = data.sp_samples[trial_idx]
    trials_s_on = data.code_samples[
        trial_idx,
        np.where(data.code_numbers[trial_idx] == task_constants.EVENTS_B1["sample_on"])[
            1
        ],
    ]
    samples = np.sort(np.unique(task["sample"].values))

    # plot fr for each neuron
    for i_neuron, neuron in enumerate(neurons):
        neuron_sp = trials_sp[:, neuron, :]
        shift_sp = indep_roll(neuron_sp, -(trials_s_on + 1 - 200).astype(int), axis=1)[
            :, :1300
        ]

        # Iterate by sample and condition
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), sharey=True)
        ax2 = [ax[0].twinx(), ax[1].twinx()]
        all_max_conv = 0
        for i_ax, cond in enumerate(["in", "out"]):
            count_trials = 0
            max_conv = 0
            for i_s, i_sample in enumerate(samples):
                sample_idx = task[
                    np.logical_and(task["in_out"] == cond, task["sample"] == i_sample)
                ]["trial_idx"].values
                mean_sp = shift_sp[sample_idx].mean(axis=0)
                conv = np.convolve(mean_sp, kernel, mode="same") * 1000
                max_conv = np.max(conv) if np.max(conv) > max_conv else max_conv
                time = np.arange(0, len(conv)) - 200
                ax[i_ax].plot(time, conv, color=task_constants.PALETTE_B1[i_sample])
                # Plot spikes
                count_t = len(sample_idx)
                rows, cols = np.where(shift_sp[sample_idx] == 1)
                ax2[i_ax].scatter(
                    cols - 200,
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
            ax[i_ax].set_title(cond)

        for i_ax in range(2):
            ax[i_ax].set_ylim(0, all_max_conv + count_trials + 5)
            ax[i_ax].set_yticks(np.arange(0, all_max_conv + 5, 5))
            ax2[i_ax].set_yticks(np.arange(-all_max_conv - 5, count_trials))
            plt.setp(ax2[i_ax].get_yticklabels(), visible=False)
            ax[i_ax].vlines(
                0,
                0,
                all_max_conv + count_trials + 5,
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
        # fig.legend()
        fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)
        fig.text(
            0.5,
            0.99,
            s="%s - Aligned with %s"
            % (ss_path[:10], list(task_constants.EVENTS_B1.keys())[2]),
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.suptitle(
            "%s: neuron %d (%s)" % (s_path[-2], i_neuron + 1, cgroup), x=0.05, y=0.99
        )

        if output_dir:

            logging.info("Saving figure, neuron: %d" % (i_neuron + 1))
            fig.savefig(
                "/".join(
                    [os.path.normpath(output_dir)]
                    + [
                        ss_path
                        + "_n"
                        + str(i_neuron + 1)
                        + "_"
                        + cgroup
                        + "_"
                        + cond
                        + "_b1.jpg"
                    ]
                )
            )
    # fig_task.savefig(
    #     "/".join([os.path.normpath(output_dir)] + [s_path + "_info_task_b1.jpg"])
    # )
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
    parser.add_argument("--in_out", default=1, help="1 in, -1 out of the rf", type=int)
    parser.add_argument(
        "--e_align", "-e", default=2, help="Event to aligne the spikes", type=int
    )
    parser.add_argument(
        "--cgroup", "-g", default="good", help="cluster goup, good or mua", type=str
    )
    args = parser.parse_args()
    try:
        main(args.filepath, args.output_dir, args.in_out, args.e_align, args.cgroup)
    except FileExistsError:
        logging.error("filepath does not exist")
