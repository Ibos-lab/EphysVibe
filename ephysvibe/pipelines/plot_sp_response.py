import os
import argparse
from pathlib import Path
import logging
from typing import Dict
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pandas as pd
from ..trials.spikes import firing_rate


def main(file_path, output_dir):
    s_path = os.path.normpath(file_path).split(os.sep)[-1][:-4]
    log_output = "/".join(
        [os.path.normpath(output_dir)] + [s_path + "plot_sp_response.log"]
    )
    logging.basicConfig(
        filename=log_output,
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
        "target_on": 37,
        "target_off": 38,
        "fix_spot_off": 36,
        "correct_response": 40,
        "start_trial": 9,
        "end_trial": 18,
    }
    # kernel parameters
    fs = 30000
    w = 0.015  # seconds = 15 ms
    w_size = 0.1  # seconds = 100ms

    # define kernel for the convolution
    kernel = signal.gaussian(M=w_size * fs, std=w * fs)
    kernel = kernel / sum(kernel)  # area of the kernel must be one

    fs = 30000
    # plots x lim
    x_lim_max = 2
    x_lim_min = -0.7
    # select only individual neurons
    neurons = np.where(sp["clustersgroup"] == "good")[0]
    logging.info("Number of neurons: %d" % len(neurons))

    mean_fr: Dict[str, list] = defaultdict(list)
    for i, neuron in enumerate(neurons):
        mean_fr["neuron"] += [i + 1] * len(target_codes.keys())
        fig, _ = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)
        for code in target_codes.keys():
            target_trials_idx = trials_idx[code][
                "trials_idx"
            ]  # select trials with the same stimulus
            events_timestamps = firing_rate.select_events_timestamps(
                sp, target_trials_idx, events
            )  # select events timestamps for all trials
            neuron_trials = firing_rate.align_neuron_spikes(
                target_trials_idx, sp, neuron, events_timestamps[:, 0]
            )
            axis = target_codes[code][1]
            ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
            if len(neuron_trials[0]) != 0:
                # Compute trial average fr
                trial_average_sp, sorted_sp_neuron = firing_rate.trial_average_fr(
                    neuron_trials
                )
            else:
                trial_average_sp = [0] * len(kernel)
                sorted_sp_neuron = [0] * len(kernel)
            conv = np.convolve(trial_average_sp, kernel, mode="same") * fs
            # compute mean of conv between events
            mean_idx_events = np.floor(
                ((events_timestamps.T - events_timestamps.T[0]).T).mean(axis=0)
            )
            conv_visual = np.mean(conv[mean_idx_events[0] : mean_idx_events[1]])
            conv_delay = np.mean(conv[mean_idx_events[1] : mean_idx_events[2]])
            conv_saccade = np.mean(conv[mean_idx_events[2] : mean_idx_events[-3]])
            # add values to dict
            mean_fr["code"] += [code]
            mean_fr["conv_visual"] += [conv_visual]
            mean_fr["conv_delay"] += [conv_delay]
            mean_fr["conv_saccade"] += [conv_saccade]
            # plot
            fig = firing_rate.plot_raster_fr(
                trial_average_sp,
                sorted_sp_neuron,
                conv,
                fs,
                neuron_trials,
                code,
                ax,
                fig,
                i,
                x_lim_max,
                x_lim_min,
                events=mean_idx_events,
            )
        fig.legend(["Target ON"], fontsize=9)
        plt.savefig(
            "/".join(
                [os.path.normpath(output_dir)] + [s_path + "_n" + str(i + 1) + ".jpg"]
            )
        )
    mean_fr = pd.DataFrame(mean_fr)
    mean_fr.to_csv(
        "/".join([os.path.normpath(output_dir)] + [s_path + "mean_fr" + ".csv"])
    )


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
    args = parser.parse_args()
    main(args.file_path, args.output_dir)
