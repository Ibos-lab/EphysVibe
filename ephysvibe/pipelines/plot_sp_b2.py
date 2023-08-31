import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from ..trials.spikes import firing_rate, sp_constants, plot_raster
from ..spike_sorting import config
from ..task import task_constants
from ..structures.spike_data import SpikeData
from ..structures.bhv_data import BhvData
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(sp_path: Path, output_dir: Path, e_align: str, t_before: int):
    s_path = os.path.normpath(sp_path).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-3]])
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if filepath exist
    if not os.path.exists(sp_path):
        raise FileExistsError

    logging.info("-- Start --")
    data = SpikeData.from_python_hdf5(sp_path)

    # Select trials and create task frame
    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == 2))[0]
    logging.info("Number of clusters: %d" % len(data.clustersgroup))
    # Define target codes
    position_codes = {
        # code: [[MonkeyLogic axis], [plot axis]]
        "127": [[10, 0], [1, 2]],
        "126": [[7, 7], [0, 2]],
        "125": [[0, 10], [0, 1]],
        "124": [[-7, 7], [0, 0]],
        "123": [[-10, 0], [1, 0]],
        "122": [[-7, -7], [2, 0]],
        "121": [[0, -10], [2, 1]],
        "120": [[7, -7], [2, 2]],
    }
    # create dict with the trials having each code
    target_codes = {}
    for i_key, key in enumerate(position_codes.keys()):
        trials = []
        code_idx = []
        for i_trial, code in zip(trial_idx, data.code_numbers[trial_idx]):
            idx = np.where(int(key) == code)[0]
            if len(idx) != 0:
                code_idx.append(idx[0])
                trials.append(i_trial)
        target_codes[key] = {
            "code_idx": code_idx,
            "trial_idx": trials,
            "position_codes": position_codes[key][1],
        }
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

        code_samples = data.code_samples
        code_numbers = data.code_numbers
        sp_samples = data.sp_samples
        e_code_align = task_constants.EVENTS_B2[e_align]
        fig, _ = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)  # define figure
        (
            all_ax,
            all_ax2,
            all_max_conv,
            max_num_trials,
        ) = plot_raster.plot_activity_location(
            target_codes,
            code_samples,
            code_numbers,
            sp_samples,
            i_n,
            e_code_align,
            t_before,
            fs_ds,
            kernel,
            rf_t_test=pd.DataFrame(),
        )

        avg_events = [-500, 0, 100, 1100]

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
        fig.suptitle(
            "%s: %s %d" % (s_path[-3], cluster, i_cluster),
            x=0,
        )
        fig.text(
            0.5,
            0.5,
            s="Depth: %d" % data.clusterdepth[i_n],
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
    parser.add_argument("sp_path", help="Path to the spikes file (sp.h5)", type=Path)
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
        main(args.sp_path, args.output_dir, args.e_align, args.t_before)
    except FileExistsError:
        logging.error("filepath does not exist")
