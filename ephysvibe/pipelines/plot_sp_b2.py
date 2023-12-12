import os
import argparse
from pathlib import Path
import logging
import numpy as np
from matplotlib import pyplot as plt
from ..trials.spikes import firing_rate
from ..trials import align_trials
from ..structures.neuron_data import NeuronData
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(neu_path: Path, output_dir: Path):
    s_path = os.path.normpath(neu_path).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-3]])
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if filepath exist
    if not os.path.exists(neu_path):
        raise FileExistsError

    logging.info("-- Start --")
    logging.info(neu_path)
    neu_data = NeuronData.from_python_hdf5(neu_path)
    # parameters
    time_before = 500
    select_block = 2
    start = -200
    end = 1500
    idx_start = time_before + start
    idx_end = time_before + end
    # Define target codes
    position_codes = {
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
    # select correct trials, block two, position, and align with target onset
    sp_target_on_all = []
    conv_all = []
    n_trials = []
    for code in position_codes.keys():
        sp_target_on, mask_in = align_trials.align_on(
            sp_samples=neu_data.sp_samples,
            code_samples=neu_data.code_samples,
            code_numbers=neu_data.code_numbers,
            trial_error=neu_data.trial_error,
            block=neu_data.block,
            pos_code=neu_data.pos_code,
            select_block=select_block,
            select_pos=int(code),
            event="target_on",
            time_before=time_before,
            error_type=0,
        )

        sp_target_on_all.append(sp_target_on[:, idx_start:idx_end])
        arr = sp_target_on.mean(axis=0)
        conv_all.append(
            firing_rate.convolve_signal(
                arr=arr, fs=1000, w_size=0.1, w_std=0.015, axis=0
            )[idx_start:idx_end]
        )
        n_trials.append(mask_in.sum())
    # plot
    fig, _ = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)
    conv_max = np.nanmax(conv_all)
    max_num_trials = max(n_trials)
    for code, sp_target_on, conv in zip(
        position_codes.keys(), sp_target_on_all, conv_all
    ):
        axis = position_codes[code][1]
        ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
        time = np.arange(0, len(conv)) + start
        ax2 = ax.twinx()
        # ----- plot conv----------
        ax.plot(time, conv, color="navy")
        # ----- plot spikes----------
        rows, cols = np.where(sp_target_on >= 1)
        cols = cols + start
        rows = rows + rows * 2
        ax2.scatter(cols, rows, marker="|", alpha=1, color="grey")
        ax.set_title("Code %s" % (code), fontsize=8)

        ax.set_ylim(0, conv_max + max_num_trials * 3)
        ax.set_yticks(np.arange(0, conv_max, 10))
        ax2.set_ylim(-conv_max, max_num_trials)
        ax2.set_yticks(np.arange(-conv_max, max_num_trials * 3, 10))

        plt.setp(ax2.get_yticklabels(), visible=False)

        ax.set_ylabel(ylabel="Average firing rate", fontsize=10, loc="bottom")
        ax.set_xlabel(xlabel="Time (s)", fontsize=10)
        ax.vlines(
            [0, 100, 1100],
            0,
            conv_max + max_num_trials * 3,
            color="k",
            linestyles="dashed",
        )  # target_on

    fig.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
    fig.suptitle(
        "%s: %s %s %d "
        % (
            neu_data.date_time,
            neu_data.area,
            neu_data.cluster_group,
            neu_data.cluster_number,
        ),
        x=0.5,
        y=1.05,
        fontsize=12,
    )
    # ----- end plot ----
    logging.info(
        "Saving figure, %s: %d" % (neu_data.cluster_group, neu_data.cluster_number)
    )
    plt.savefig(
        "/".join(
            [os.path.normpath(output_dir)]
            + [
                ss_path
                + "_"
                + neu_data.cluster_group
                + "_"
                + str(neu_data.cluster_number)
                + ".jpg"
            ]
        ),
        bbox_inches="tight",
    )

    logging.info("-- end --")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("neu_path", help="Path to neuron data (neu.h5)", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()
    try:
        main(args.neu_path, args.output_dir)
    except FileExistsError:
        logging.error("filepath does not exist")
