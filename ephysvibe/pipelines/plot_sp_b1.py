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

warnings.filterwarnings("ignore")


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
    s_path = os.path.normpath(filepath).split(os.sep)[-1][:-4]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path])
    log_output = output_dir + "/" + s_path + "_plot_sp_b2.log"
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
    file = np.load(filepath, allow_pickle=True).item(0)
    sp, bhv = file["sp_data"], file["bhv"]
    # Select trials and create task frame
    trial_idx = select_trials.select_trials_block(sp, n_block=1)
    trial_idx = select_trials.select_correct_trials(bhv, trial_idx)
    task = def_task.create_task_frame(trial_idx, bhv, task_constants.SAMPLES_COND)
    fig_task, data_task = def_task.info_task(task)
    fig_task.savefig(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_info_task_b1.jpg"])
    )
    data_task.to_csv(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_data_task_b1" + ".csv"])
    )
    neurons = np.where(sp["clustersgroup"] == cgroup)[0]
    logging.info("Number of clusters: %d" % len(sp["clustersgroup"]))
    logging.info("Number of %s units: %d" % (cgroup, len(neurons)))
    logging.info("in_out: %d" % in_out)
    logging.info("e_align: %s" % list(task_constants.EVENTS_B1.keys())[e_align])
    # define kernel for convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    # compute and plot fr
    fr_samples = firing_rate.fr_by_sample_neuron(
        sp=sp,
        neurons=neurons,
        task=task,
        in_out=in_out,
        kernel=kernel,
        e_align=e_align,
        output_dir=output_dir,
        filename=s_path,
    )
    logging.info("Saving .svc file: %s" % ("fr_samples_b1"))
    fr_samples.to_csv(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_fr_samples_b1" + ".csv"])
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
