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
    s_path = os.path.normpath(filepath).split(os.sep)[-1][:-4]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path])
    log_output = output_dir + "/" + s_path + "_plot_sp_b1.log"
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
    # target_trials_idx = task["trial_idx"].values
    # plot fr for each neuron
    for i_neuron, neuron in enumerate(neurons):
        neuron_sp = trials_sp[:, i_neuron, :]
        shift_sp = indep_roll(neuron_sp, -(trials_s_on + 1 - 200).astype(int), axis=1)[
            :, :1200
        ]

        # Iterate by sample and condition
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharey=True)

        for i_ax, cond in enumerate(["in", "out"]):
            trials_conv_fr, all_mask = [], []
            ax2 = ax[i_ax].twinx()
            for i_sample in samples:
                sample_idx = task[
                    np.logical_and(task["in_out"] == cond, task["sample"] == i_sample)
                ]["trial_idx"].values
                mean_sp = shift_sp[sample_idx].mean(axis=0)
                conv = (
                    np.convolve(mean_sp, kernel, mode="same") * 1000
                )  # todo: check fs and change the 1000

                ax.plot(
                    conv,
                    color=task_constants.PALETTE_B1[i_sample],
                    label="Sample %s" % i_sample,
                )

            # firing_rate.plot_b1(
            #     ax[i_ax],
            #     samples,
            #     trials_conv_fr,
            #     trials_time,
            #     neuron_trials_shift[task["in_out"] == cond],
            #     events_shift,
            #     cond,
            #     x_lim_min,
            #     x_lim_max,
            #     all_mask,
            # )
        ax[1].legend(fontsize=9)
        # fig.legend()
        fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
        fig.text(
            0.10,
            0.01,
            s="%s - Aligned with %s"
            % (s_path[:10], list(task_constants.EVENTS_B1.keys())[e_align]),
            horizontalalignment="center",
            verticalalignment="center",
        )
        fig.suptitle("Neuron (%s) %d" % (cgroup, i_neuron + 1), x=0.10)

        if output_dir:

            logging.info("Saving figure, neuron: %d" % (i_neuron + 1))
            fig.savefig(
                "/".join(
                    [os.path.normpath(output_dir)]
                    + [
                        s_path
                        + "_n"
                        + str(i_neuron + 1)
                        + "_"
                        + cgroup
                        + "_"
                        + condition[in_out]
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
