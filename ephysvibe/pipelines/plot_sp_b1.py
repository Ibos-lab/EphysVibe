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


def main(file_path, output_dir, in_out, e_align, plot=True):
    s_path = os.path.normpath(file_path).split(os.sep)[-1][:-4]
    log_output = "/".join([os.path.normpath(output_dir)] + [s_path + "_plot_sp_b2.log"])
    logging.basicConfig(
        handlers=[logging.FileHandler(log_output), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s | %(message)s ",
        datefmt="%d/%m/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    logging.info("-- Start --")
    file = np.load(file_path, allow_pickle=True).item(0)
    sp, bhv = file["sp_data"], file["bhv"]
    trial_idx = select_trials.select_trials_block(sp, n_block=1)
    trial_idx = select_trials.select_correct_trials(bhv, trial_idx)
    b_sp_samples = [sp["sp_samples"][i] for i in trial_idx]
    logging.info("Number of clusters: %d" % len(b_sp_samples[0]))
    task = def_task.create_task_frame(trial_idx, bhv, task_constants.SAMPLES_COND)
    fig_task, data_task = def_task.info_task(task)
    fig_task.savefig(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_info_task_b1.jpg"])
    )
    data_task.to_csv(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_data_task_b1" + ".csv"])
    )
    # define kernel for convolution
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    neurons = np.where(sp["clustersgroup"] == "good")[0]
    logging.info("in_out: %d" % in_out)
    logging.info("e_align: %s" % list(task_constants.EVENTS_B1.keys())[e_align])
    # compute fr
    fr_samples, _ = firing_rate.fr_by_sample_neuron(
        sp=sp,
        neurons=neurons,
        task=task,
        in_out=in_out,
        kernel=kernel,
        e_align=e_align,
        output_dir=output_dir,
        filename=s_path,
        fs_ds=fs_ds,
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
    parser.add_argument(
        "file_path", help="Path to the continuous file (.dat)", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--in_out", default=1, help="1 in, -1 out of the rf", type=int)
    parser.add_argument(
        "--e_align", "-e", default=2, help="Event to aligne the spikes", type=int
    )
    args = parser.parse_args()
    main(args.file_path, args.output_dir, args.in_out, args.e_align)
