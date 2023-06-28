import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, Optional
from ephysvibe.structures.trials_data import TrialsData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.task import task_constants, def_task
import os


def plot_activity_location(
    target_codes,
    code_samples,
    code_numbers,
    sp_samples,
    i_n,
    e_code_align,
    t_before,
    fs_ds,
    kernel,
    rf_t_test: Optional[pd.DataFrame] = pd.DataFrame(),
):
    all_ax, all_ax2 = [], []
    all_max_conv, max_num_trials = 0, 0
    for code in target_codes.keys():

        target_t_idx = target_codes[code][
            "trial_idx"
        ]  # select trials with the same stimulus
        trials_s_on = code_samples[
            target_t_idx,
            np.where(code_numbers[target_t_idx] == e_code_align)[1],
        ]  # moment e_align occurs in each trial
        shift_sp = TrialsData.indep_roll(
            sp_samples[target_t_idx, i_n],
            -(trials_s_on - t_before).astype(int),
            axis=1,
        )[
            :, :2300
        ]  # align trials on event
        mean_sp = shift_sp.mean(axis=0)  # mean of all trials
        conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
        conv_max = max(conv)
        all_max_conv = conv_max if conv_max > all_max_conv else all_max_conv
        num_trials = shift_sp.shape[0]
        max_num_trials = num_trials if num_trials > max_num_trials else max_num_trials
        axis = target_codes[code]["position_codes"]
        ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))

        time = np.arange(0, len(conv)) - t_before
        # ----- plot ----------
        ax2 = ax.twinx()
        ax.plot(time, conv, color="navy")
        num_trials = shift_sp.shape[0]
        rows, cols = np.where(shift_sp >= 1)
        cols = cols - t_before
        rows = rows + rows * 2
        ax2.scatter(cols, rows, marker="|", alpha=1, color="grey")
        ax.set_title("Code %s" % (code), fontsize=8)
        if not rf_t_test.empty:
            if (
                code
                in rf_t_test[
                    (rf_t_test["array_position"] == i_n) & (rf_t_test["p"] < 0.05)
                ]["rf"].values
            ):
                ax.set_facecolor("bisque")
        all_ax.append(ax)
        all_ax2.append(ax2)
    return all_ax, all_ax2, all_max_conv, max_num_trials
