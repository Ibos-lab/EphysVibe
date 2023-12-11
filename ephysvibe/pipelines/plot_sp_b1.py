# plot spiking activity task b1
import os
import argparse
from pathlib import Path
import logging
import numpy as np
from ..trials.spikes import firing_rate
from ..trials import select_trials, align_trials
from ..task import task_constants
import warnings
from matplotlib import pyplot as plt
from ..structures.neuron_data import NeuronData

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def plot_raster_conv(sp_sample, start, conv, time, ax1, ax2, colors):
    n_trials = 0
    for key, sp in sp_sample.items():
        # ----- plot conv----------
        ax1.plot(time, conv[key], color=colors[key])
        # ----- plot spikes----------
        rows, cols = np.where(sp >= 1)
        ax2.scatter(
            cols + start,
            rows + n_trials,
            marker="|",
            alpha=1,
            edgecolors="none",
            color=colors[key],
            label="sample %s" % key,
        )
        n_trials = n_trials + sp.shape[0]

    return ax1, ax2, n_trials


def main(neuron_path: Path, output_dir: Path):
    """Compute and plot firing rate during task b1.

    Args:
        neuron_path (Path): _description_
        output_dir (Path): _description_
    """
    s_path = os.path.normpath(neuron_path).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-3]])

    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if filepath exist
    if not os.path.exists(neuron_path):
        raise FileExistsError
    logging.info("-- Read neuron data --")
    neu_data = NeuronData.from_python_hdf5(neuron_path)
    # parameters
    time_before = 500
    select_block = 1
    start = -200
    end = 1200
    idx_start = time_before + start
    idx_end = time_before + end
    # ---------------
    conv_all = {"in": {}, "out": {}}
    all_sp_by_sample = {"in": {}, "out": {}}
    pos_label = ["in", "out"]
    pos_code = [1, -1]
    count_trials = 0
    all_max_conv = 0
    all_max_trial = 0
    logging.info("-- Start --")
    for code, pos in zip(pos_code, pos_label):
        # select correct trials, block one, inside/outside RF, and align with sample onset
        sp_sample_on, mask = align_trials.align_on(
            sp_samples=neu_data.sp_samples,
            code_samples=neu_data.code_samples,
            code_numbers=neu_data.code_numbers,
            trial_error=neu_data.trial_error,
            block=neu_data.block,
            pos_code=neu_data.pos_code,
            select_block=select_block,
            select_pos=code,
            event="sample_on",
            time_before=time_before,
            error_type=0,
        )

        sample_id = neu_data.sample_id[mask]
        sp_by_sample = select_trials.get_sp_by_sample(sp_sample_on, sample_id)
        for key, value in sp_by_sample.items():
            arr = value.mean(axis=0)
            conv = firing_rate.convolve_signal(
                arr=arr, fs=1000, w_size=0.1, w_std=0.015, axis=0
            )[idx_start:idx_end]
            conv_all[pos][key] = conv
            sp_by_sample[key] = value[:, idx_start:idx_end]
            count_trials = value.shape[0]
            max_conv = np.max(conv)
            all_max_conv = max_conv if max_conv > all_max_conv else all_max_conv
            all_max_trial = (
                count_trials if count_trials > all_max_trial else all_max_trial
            )
        all_sp_by_sample[pos] = sp_by_sample

    fig, (ax_in, ax_out) = plt.subplots(nrows=1, ncols=2, figsize=(25, 10), sharey=True)

    time = np.arange(0, end - start) + start
    ax_in_2 = ax_in.twinx()
    pos = "in"
    ax_in, ax_in_2, n_trials_in = plot_raster_conv(
        all_sp_by_sample[pos],
        start,
        conv_all[pos],
        time,
        ax_in,
        ax_in_2,
        colors=task_constants.PALETTE_B1,
    )
    ax_out_2 = ax_out.twinx()
    pos = "out"
    ax_out, ax_out_2, n_trials_out = plot_raster_conv(
        all_sp_by_sample[pos],
        start,
        conv_all[pos],
        time,
        ax_out,
        ax_out_2,
        colors=task_constants.PALETTE_B1,
    )

    all_max_trial = max([n_trials_in, n_trials_out])
    # -----parameters in plot
    ax_in_2.axes.set_yticks(np.arange(-all_max_conv - 5, all_max_trial))
    ax_in.axes.set_ylim(0, all_max_conv + all_max_trial + 5)
    ax_in.axes.set_yticks(np.arange(0, all_max_conv + 5, 10))
    plt.setp(ax_in_2.axes.get_yticklabels(), visible=False)
    plt.setp(ax_in_2.axes.get_yaxis(), visible=False)
    ax_in.set_title("in", fontsize=15)
    ax_in.vlines(
        [0, 450, 900],
        0,
        all_max_conv + all_max_trial + 5,
        color="k",
        linestyles="dashed",
    )
    ax_in.spines["right"].set_visible(False)
    ax_in.spines["top"].set_visible(False)
    ax_in_2.spines["right"].set_visible(False)
    ax_in_2.spines["top"].set_visible(False)
    ax_in.set_xlabel(xlabel="Time (ms)", fontsize=18)
    ax_in.set_ylabel(ylabel="Average firing rate", fontsize=15, loc="bottom")
    for xtick in ax_in.xaxis.get_major_ticks():
        xtick.label1.set_fontsize(15)
    for ytick in ax_in.yaxis.get_major_ticks():
        ytick.label1.set_fontsize(15)
    for xtick in ax_in_2.xaxis.get_major_ticks():
        xtick.label1.set_fontsize(15)
    # -----parameters out plot
    ax_out_2.axes.set_yticks(np.arange(-all_max_conv - 5, all_max_trial))
    ax_out.axes.set_ylim(0, all_max_conv + all_max_trial + 5)
    ax_out.axes.set_yticks(np.arange(0, all_max_conv + 5, 10))
    plt.setp(ax_out_2.axes.get_yticklabels(), visible=False)
    plt.setp(ax_out_2.axes.get_yaxis(), visible=False)
    ax_out.set_title("out", fontsize=15)
    ax_out.vlines(
        [0, 450, 900],
        0,
        all_max_conv + all_max_trial + 5,
        color="k",
        linestyles="dashed",
    )
    ax_out.spines["right"].set_visible(False)
    ax_out.spines["top"].set_visible(False)
    ax_out_2.spines["right"].set_visible(False)
    ax_out_2.spines["top"].set_visible(False)
    ax_out.set_xlabel(xlabel="Time (ms)", fontsize=18)
    for xtick in ax_out.xaxis.get_major_ticks():
        xtick.label1.set_fontsize(15)
    for ytick in ax_out.yaxis.get_major_ticks():
        ytick.label1.set_fontsize(15)
    for xtick in ax_out_2.xaxis.get_major_ticks():
        xtick.label1.set_fontsize(15)
    ax_out_2.legend(
        fontsize=15, scatterpoints=5, columnspacing=0.5, framealpha=0, loc="upper right"
    )
    ## --- figure
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)
    fig.suptitle(
        "%s: %s %s %d "
        % (
            neu_data.date_time,
            neu_data.area,
            neu_data.cluster_group,
            neu_data.cluster_number,
        ),
        x=0.05,
        y=0.99,
        fontsize=15,
    )

    logging.info(
        "Saving figure, %s: %d" % (neu_data.cluster_group, neu_data.cluster_number)
    )
    fig.savefig(
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
    parser.add_argument("neuron_path", help="Path to neuron data (neu.h5)", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()
    try:
        main(args.neuron_path, args.output_dir)
    except FileExistsError:
        logging.error("filepath does not exist")
