import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict
from ..trials import select_trials
from ..trials import firing_rate
from ..task import def_task
from ..spike_sorting import config
from . import constants
import warnings

warnings.filterwarnings("ignore")


def plot_b1(
    samples,
    all_trials_mean_fr,
    neuron_trials_shift,
    events_shift,
    max_shift,
    conv_max,
    fs,
    num_trials,
    in_out,
    x_lim_min,
    x_lim_max,
    i_neuron,
):
    conv_max = conv_max + 1
    fig, ax = plt.subplots(figsize=(10, 6), sharex=True, sharey=True)
    ax2 = ax.twinx()
    for i_s, i_sample in enumerate(samples):
        ax.plot(
            (np.arange(len(all_trials_mean_fr[i_s])) - max_shift) / fs,
            all_trials_mean_fr[i_s],
            color=constants.PALETTE_B1[i_sample],
        )
    line_len = 0.3
    ax2.eventplot(
        neuron_trials_shift / fs,
        color=".2",
        lineoffsets=np.arange(conv_max, (num_trials * line_len) + conv_max, line_len),
        linewidths=0.5,
        linelengths=line_len,
    )
    # events
    for n_event, n_color in zip(range(1, 6), ["k", "b", "k", "k", "k"]):
        ax.vlines(
            events_shift[n_event] / fs,
            0,
            conv_max + (num_trials * line_len),  # ( +
            color=n_color,
            linestyles="dashed",
        )
    ax.set(xlabel="Time (s)", ylabel="Average firing rate")
    ax2.set(xlabel="Time (s)", ylabel="trials")

    ax.set_title("in_out: %d" % (in_out))
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(0)
    ax2.set_ylim(0)
    ax2.set_yticks(np.arange(conv_max, (num_trials * line_len) + conv_max, line_len))
    plt.setp(ax2.get_yticklabels(), visible=False)
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    fig.legend(
        [
            "Sample o1_c1",
            "Sample o1_c5",
            "Sample o5_c1",
            "Sample o5_c5",
            "Sample o0_c0",
            "Fixation",
            "Sample on",
            "Sample off",
            "Test on",
            "Test off",
        ],
        fontsize=9,
    )
    fig.suptitle("Neuron %d" % (i_neuron + 1), x=0)
    return fig


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
    sp = file["sp_data"]
    bhv = file["bhv"]

    # Selec trials in a block
    n_block = 1
    trial_idx = select_trials.select_trials_block(sp, n_block)
    # Selec correct trials
    trial_idx = select_trials.select_correct_trials(bhv, trial_idx)
    b_sp_samples = [sp["sp_samples"][i] for i in trial_idx]
    logging.info("Number of clusters: %d" % len(b_sp_samples[0]))

    task = def_task.create_task_frame(trial_idx, bhv, constants.SAMPLES_COND)
    fig_task, data_task = def_task.info_task(task)
    fig_task.savefig(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_info_task_b1.jpg"])
    )
    data_task.to_csv(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_data_task_b1" + ".csv"])
    )
    # define kernel for the convolution
    w = 0.015  # seconds = 15 ms
    w_size = 0.1  # seconds = 100ms
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(w, w_size, fs=fs_ds)
    fr_samples: Dict[str, list] = defaultdict(list)
    x_lim_max = 1.5
    x_lim_min = -0.7
    neurons = np.where(sp["clustersgroup"] == "good")[0]
    logging.info("in_out: %d" % in_out)
    logging.info("e_align: %s" % list(constants.EVENTS_B1.keys())[e_align])
    samples = ["o1_c1", "o1_c5", "o5_c1", "o5_c5", "o0_c0"]
    neuron_max_shift = np.zeros(len(neurons))
    for i, neuron in enumerate(neurons):

        sample_id = task[(task["in_out"] == in_out)]["sample_id"].values
        target_trials_idx = task[(task["in_out"] == in_out)]["idx_trial"].values
        # events
        ev_ts = firing_rate.select_events_timestamps(
            sp, target_trials_idx, constants.EVENTS_B1
        )  # select events timestamps for all trials
        shift_ev_ts = np.floor(((ev_ts.T - ev_ts.T[0]).T) / config.DOWNSAMPLE)
        # sp
        neuron_trials = firing_rate.align_neuron_spikes(
            target_trials_idx, sp, neuron, ev_ts[:, 0]
        )  # set the 0 at the event start trial
        trials_sp = firing_rate.sp_from_timestime_to_binary(
            neuron_trials, config.DOWNSAMPLE
        )
        sp_shift, max_shift, events_shift = firing_rate.reshape_sp_list(
            trials_sp, event_timestamps=shift_ev_ts, align_event=e_align
        )
        conv = firing_rate.compute_fr(
            neuron_trials, kernel, config.FS / config.DOWNSAMPLE, config.DOWNSAMPLE
        )
        all_trials_fr, _, _ = firing_rate.aligne_conv_fr(
            conv=conv, event_timestamps=shift_ev_ts, align_event=e_align
        )
        neuron_max_shift[i] = int(max_shift)

        all_trials_mean_fr = []
        for i_sample in samples:
            mask_sample = sample_id == i_sample
            sp_sample = sp_shift[mask_sample, :]
            conv_sample = all_trials_fr[mask_sample, :]
            # ev_sample = shift_ev_ts[mask_sample]
            # save in dict
            for t in np.arange(sp_sample.shape[1]):
                fr_samples["t" + str(t)] += sp_sample[:, t].tolist()
            fr_samples["neuron"] += [i + 1] * len(sp_sample)
            fr_samples["sample"] += [i_sample] * len(sp_sample)
            fr_samples["trial_idx"] += target_trials_idx[mask_sample].tolist()
            all_trials_mean_fr.append(np.mean(conv_sample, axis=0))

        if plot == True:
            all_trials_mean_fr = np.array(all_trials_mean_fr)
            events_shift = np.mean(events_shift, axis=0) - max_shift
            conv_max = int(np.floor(np.max(all_trials_mean_fr)))
            num_trials = len(neuron_trials)
            # lineoffsets = np.arange(conv_max, num_trials * 0.2 + conv_max)
            neuron_trials_shift = (
                firing_rate.align_neuron_spikes(
                    target_trials_idx, sp, neuron, ev_ts[:, 2]
                )
                / config.DOWNSAMPLE
            )  # align sp with stim onset
            fig = plot_b1(
                samples,
                all_trials_mean_fr,
                neuron_trials_shift,
                events_shift,
                max_shift,
                conv_max,
                fs_ds,
                num_trials,
                in_out,
                x_lim_min,
                x_lim_max,
                i,
            )
            logging.info("Saving figure, neuron: %d" % (i + 1))
            fig.savefig(
                "/".join(
                    [os.path.normpath(output_dir)]
                    + [s_path + "_n" + str(i + 1) + "_b1.jpg"]
                )
            )
    fr_samples = pd.DataFrame(fr_samples)
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
