import os
import sys
import argparse
from pathlib import Path
import logging
from typing import Dict
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pandas as pd
from ..trials import firing_rate
from ..spike_sorting import config
import warnings

warnings.filterwarnings("ignore")


def main(file_path, output_dir, e_align):
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

    # Select relevant trials
    # Selec trials in a block
    block = 2
    block_idx = np.where(sp["blocks"] == block)[0]
    logging.info("Number of trials in block 2: %d" % len(block_idx))
    # Selec correct trials
    correct_mask = []
    for n_trial in block_idx:
        correct_mask.append(bhv[n_trial]["TrialError"][0][0] == 0.0)
    logging.info("Number of correct trials in block 2: %d" % sum(correct_mask))
    block_idx = block_idx[correct_mask]
    logging.info("Number of clusters: %d" % len(sp["sp_samples"][0]))

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
    for key in target_codes.keys():
        trials = []
        code_idx = []
        for i_trial, code in zip(block_idx, sp["code_numbers"][block_idx]):
            idx = np.where(int(key) == code)[0]
            if len(idx) != 0:
                code_idx.append(idx[0])
                trials.append(i_trial)
        trials_idx[key] = {"code_idx": code_idx, "trials_idx": trials}

    # Plot rasters for each neuron for each code
    events = {
        "start_trial": 9,
        "target_on": 37,
        "target_off": 38,
        "fix_spot_off": 36,
        "correct_response": 40,
        "end_trial": 18,
    }
    # kernel parameters
    fs_ds = config.FS / config.DOWNSAMPLE
    w = 0.015  # seconds = 15 ms
    w_size = 0.1  # seconds = 100ms
    kernel = firing_rate.define_kernel(w, w_size, fs=fs_ds)
    # plots x lim
    x_lim_max = 2
    x_lim_min = -0.7
    logging.info("e_align: %s" % list(events.keys())[e_align])
    # select only individual neurons
    neurons = np.where(sp["clustersgroup"] == "good")[0]
    mean_fr: Dict[str, list] = defaultdict(list)
    for i, neuron in enumerate(neurons):
        fig, _ = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)
        for code in target_codes.keys():
            target_trials_idx = trials_idx[code][
                "trials_idx"
            ]  # select trials with the same stimulus
            ev_timestamps = firing_rate.select_events_timestamps(
                sp, target_trials_idx, events
            )  # select events timestamps for all trials
            neuron_trials = firing_rate.align_neuron_spikes(
                target_trials_idx, sp, neuron, ev_timestamps[:, 0]
            )  # set the 0 at the event start trial
            shift_ev_timestamps = np.floor(
                ((ev_timestamps.T - ev_timestamps.T[0]).T) / config.DOWNSAMPLE
            )
            conv = firing_rate.compute_fr(
                neuron_trials, kernel, config.FS / config.DOWNSAMPLE, config.DOWNSAMPLE
            )
            visual_mean_fr = firing_rate.fr_in_window(
                x=conv, start=shift_ev_timestamps[:, 1], end=shift_ev_timestamps[:, 2]
            )
            delay_mean_fr = firing_rate.fr_in_window(
                x=conv, start=shift_ev_timestamps[:, 2], end=shift_ev_timestamps[:, 3]
            )
            saccade_mean_fr = firing_rate.fr_in_window(
                x=conv, start=shift_ev_timestamps[:, 3], end=shift_ev_timestamps[:, 4]
            )
            # add values to dict
            length = len(target_trials_idx)
            mean_fr["neuron"] += [i + 1] * length * 3
            mean_fr["code"] += [code] * length * 3
            mean_fr["event"] += ["visual"] * length
            mean_fr["event"] += ["delay"] * length
            mean_fr["event"] += ["saccade"] * length
            mean_fr["trial"] += (
                target_trials_idx * 3
            )  # 3 times trials idx (for each window)
            mean_fr["fr"] += np.concatenate(
                (visual_mean_fr, delay_mean_fr, saccade_mean_fr)
            ).tolist()
            # plot
            all_trials_fr, max_shift, events_shift = firing_rate.aligne_conv_fr(
                conv=conv, event_timestamps=shift_ev_timestamps, align_event=e_align
            )
            all_trials_fr = np.mean(all_trials_fr, axis=0)
            axis = target_codes[code][1]
            ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
            neuron_trials = (
                firing_rate.align_neuron_spikes(
                    target_trials_idx, sp, neuron, ev_timestamps[:, e_align]
                )
                / config.DOWNSAMPLE
            )  # align sp with stim onset
            events_shift = np.mean(events_shift, axis=0) - max_shift
            fig = firing_rate.plot_raster_fr(
                all_trials_fr,
                max_shift,
                fs_ds,
                neuron_trials,
                code,
                ax,
                fig,
                i,
                x_lim_max,
                x_lim_min,
                events=events_shift,
            )

        fig.text(
            0.5,
            0.5,
            s="Aligned with %s" % list(events.keys())[e_align],
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(
            "/".join(
                [os.path.normpath(output_dir)] + [s_path + "_n" + str(i + 1) + ".jpg"]
            )
        )
    mean_fr = pd.DataFrame(mean_fr)
    mean_fr.to_csv(
        "/".join([os.path.normpath(output_dir)] + [s_path + "_mean_fr" + ".csv"])
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
        "--e_align", "-e", default=1, help="Event to aligne the spikes", type=int
    )
    args = parser.parse_args()
    main(args.file_path, args.output_dir, args.e_align)
