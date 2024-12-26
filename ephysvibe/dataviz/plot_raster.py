from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials import firing_rate, select_trials
from ephysvibe.task import task_constants
from scipy import stats
from ephysvibe.spike_sorting import config
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def prepare_data_plotb1(
    neu,
    rf_stim_loc: list = ["contra", "ipsi"],
    percentile: bool = False,
    cerotr: bool = False,
):
    samples = [0, 11, 15, 55, 51]
    # IN
    sp_sampleon_0, mask_sampleon_0 = neu.align_on(
        select_block=1,
        event="sample_on",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[0],
    )
    samples_sampleon_0 = select_trials.get_sp_by_sample(
        sp_sampleon_0, neu.sample_id[mask_sampleon_0], samples=samples
    )
    sp_test_0, mask_test_0 = neu.align_on(
        select_block=1,
        event="test_on_1",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[0],
    )
    samples_test_0 = select_trials.get_sp_by_sample(
        sp_test_0, neu.sample_id[mask_test_0], samples=samples
    )
    # OUT
    sp_sampleon_1, mask_sampleon_1 = neu.align_on(
        select_block=1,
        event="sample_on",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[1],
    )
    samples_sampleon_1 = select_trials.get_sp_by_sample(
        sp_sampleon_1, neu.sample_id[mask_sampleon_1], samples=samples
    )
    sp_test_1, mask_test_1 = neu.align_on(
        select_block=1,
        event="test_on_1",
        time_before=500,
        error_type=0,
        rf_stim_loc=rf_stim_loc[1],
    )
    samples_test_1 = select_trials.get_sp_by_sample(
        sp_test_1, neu.sample_id[mask_test_1], samples=samples
    )

    # Check trials
    samplesstr = ["0", "11", "15", "55", "51"]
    if percentile or cerotr:
        for isamp in samplesstr:
            if ~np.all((np.isnan(samples_sampleon_0[isamp]))):
                temp = np.concatenate(
                    (
                        samples_sampleon_0[isamp][:, 300 : 500 + 450 + 400],
                        samples_test_0[isamp][:, 100 : 500 + 500],
                    ),
                    axis=1,
                )
                masknocero = np.full(temp.shape[0], True)
                maskper = np.full(temp.shape[0], True)
                if cerotr:
                    masknocero = np.sum(temp, axis=1) != 0
                if percentile:
                    maskper = select_trials.select_trials_by_percentile(
                        temp, masknocero
                    )
                mask = np.logical_and(masknocero, maskper)
                if np.sum(mask) < 10:
                    mask = np.full(temp.shape[0], True)
                samples_sampleon_0[isamp] = samples_sampleon_0[isamp][mask]
                samples_test_0[isamp] = samples_test_0[isamp][mask]

            if ~np.all((np.isnan(samples_sampleon_1[isamp]))):
                temp = np.concatenate(
                    (
                        samples_sampleon_1[isamp][:, 300 : 500 + 450 + 400],
                        samples_test_1[isamp][:, 100 : 500 + 500],
                    ),
                    axis=1,
                )
                masknocero = np.full(temp.shape[0], True)
                maskper = np.full(temp.shape[0], True)
                if cerotr:
                    masknocero = np.sum(temp, axis=1) != 0
                if percentile:
                    maskper = select_trials.select_trials_by_percentile(
                        temp, masknocero
                    )
                mask = np.logical_and(masknocero, maskper)
                if np.sum(mask) < 10:
                    mask = np.full(temp.shape[0], True)
                samples_sampleon_1[isamp] = samples_sampleon_1[isamp][mask]
                samples_test_1[isamp] = samples_test_1[isamp][mask]

    # Start convolution
    # IN
    conv_0 = {}
    samples_0 = {}
    for isamp in samples_sampleon_0.keys():
        if np.all((np.isnan(samples_sampleon_0[isamp]))):
            conv_0[isamp] = np.zeros((1, 1950))
            samples_0[isamp] = np.zeros((1, 1950))
            continue
        mean_fr = np.mean(samples_sampleon_0[isamp], axis=0)
        conv_sonin = firing_rate.convolve_signal(mean_fr, axis=0)[300 : 500 + 450 + 400]
        mean_fr = np.mean(samples_test_0[isamp], axis=0)
        conv_testin = firing_rate.convolve_signal(mean_fr, axis=0)[100 : 500 + 500]

        conv_0[isamp] = np.concatenate((conv_sonin, conv_testin))
        samples_0[isamp] = np.concatenate(
            (
                samples_sampleon_0[isamp][:, 300 : 500 + 450 + 400],
                samples_test_0[isamp][:, 100 : 500 + 500],
            ),
            axis=1,
        )

    # OUT
    conv_1 = {}
    samples_1 = {}
    for isamp in samples_sampleon_1.keys():

        if np.all((np.isnan(samples_sampleon_1[isamp]))):
            conv_1[isamp] = np.zeros((1, 1950))
            samples_1[isamp] = np.zeros((1, 1950))
            continue
        mean_fr = np.mean(samples_sampleon_1[isamp], axis=0)
        conv_sonout = firing_rate.convolve_signal(mean_fr, axis=0)[
            300 : 500 + 450 + 400
        ]
        mean_fr = np.mean(samples_test_1[isamp], axis=0)
        conv_testout = firing_rate.convolve_signal(mean_fr, axis=0)[100 : 500 + 500]

        conv_1[isamp] = np.concatenate((conv_sonout, conv_testout))
        samples_1[isamp] = np.concatenate(
            (
                samples_sampleon_1[isamp][:, 300 : 500 + 450 + 400],
                samples_test_1[isamp][:, 100 : 500 + 500],
            ),
            axis=1,
        )
    sp = {rf_stim_loc[0]: samples_0, rf_stim_loc[1]: samples_1}
    conv = {rf_stim_loc[0]: conv_0, rf_stim_loc[1]: conv_1}

    return sp, conv


def prepare_data_plotb2(neu):
    align_sp, alig_mask = neu.align_on(
        select_block=2, event="target_on", time_before=400, error_type=0
    )
    pos_code = neu.pos_code[alig_mask]
    sp_pos = {}
    conv_pos = {}
    conv_all, n_trials = [], []
    max_n_tr = 0
    conv_max = 0
    for code in np.unique(pos_code):
        code_mask = pos_code == code
        sp_pos[str(int(code))] = align_sp[code_mask][:, 200:1900]
        mean_fr = np.mean(align_sp[code_mask], axis=0)[:2100]
        conv_fr = firing_rate.convolve_signal(mean_fr, axis=0)[200:1900]
        conv_pos[str(int(code))] = conv_fr
        conv_all.append(np.max(conv_fr))
        n_trials.append(align_sp[code_mask].shape[0])

    if len(n_trials) > 0:
        max_n_tr = np.max(n_trials)
        conv_max = np.max(conv_all)
    return sp_pos, conv_pos, max_n_tr, conv_max


def plot_sp_b1(neu, sp: Dict, conv: Dict):
    key1, key2 = list(sp.keys())
    # define kernel for convolution

    sampleco = {
        "0": "neutral",
        "11": "o1 c1",
        "15": "o1 c5",
        "51": "o5 c1",
        "55": "o5 c5",
    }
    t_before = 200
    # Iterate by sample and condition
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10), sharey=True)
    ax2 = [ax[0].twinx(), ax[1].twinx()]
    all_max_conv = 0
    all_max_trial = 0
    conv = {key1: conv[key1], key2: conv[key2]}
    sp = {key1: sp[key1], key2: sp[key2]}
    for i_ax, cond in enumerate([key1, key2]):
        count_trials = 0
        max_conv = 0
        for i_sample in conv[cond].keys():
            max_conv = (
                np.max(conv[cond][i_sample])
                if np.max(conv[cond][i_sample]) > max_conv
                else max_conv
            )
            time = np.arange(0, len(conv[cond][i_sample])) - t_before
            ax[i_ax].plot(
                time,
                conv[cond][i_sample],
                color=task_constants.PALETTE_B1[i_sample],
            )
            # Plot spikes
            count_t = len(sp[cond][i_sample])
            rows, cols = np.where(sp[cond][i_sample] >= 1)
            ax2[i_ax].scatter(
                cols - t_before,
                rows + count_trials,
                marker="|",
                alpha=1,
                color=task_constants.PALETTE_B1[i_sample],
                label=sampleco[i_sample],
            )
            count_trials += count_t
        all_max_conv = max_conv if max_conv > all_max_conv else all_max_conv
        all_max_trial = count_trials if count_trials > all_max_trial else all_max_trial
        ax[i_ax].set_title(cond, fontsize=15)
    for i_ax in range(2):
        ax[i_ax].set_ylim(0, all_max_conv + all_max_trial + 5)
        ax[i_ax].set_yticks(np.arange(0, all_max_conv + 5, 10))
        ax2[i_ax].set_yticks(np.arange(-all_max_conv - 5, all_max_trial))
        plt.setp(ax2[i_ax].get_yticklabels(), visible=False)
        plt.setp(ax2[i_ax].get_yaxis(), visible=False)
        ax[i_ax].set_xticks([0, 450, 1250, 1700])
        ax[i_ax].set_xticklabels(["0", "450", "0", "450"])

        ax[i_ax].vlines(
            [0, 450, 450 + 400 + 400],
            0,
            all_max_conv + all_max_trial + 5,
            color="k",
            linestyles="-",
        )
        ax[i_ax].vlines(
            [850],
            0,
            all_max_conv + all_max_trial + 5,
            color="k",
            linestyles="--",
        )
        ax2[i_ax].spines["right"].set_visible(False)
        ax2[i_ax].spines["top"].set_visible(False)
        ax[i_ax].spines["right"].set_visible(False)
        ax[i_ax].spines["top"].set_visible(False)
    ax[0].set(xlabel="Time (ms)", ylabel="Average firing rate")
    ax2[1].set(xlabel="Time (ms)", ylabel="trials")
    ax[1].set_xlabel(xlabel="Time (ms)", fontsize=18)
    # ax[1].set_xticks(fontsize=15)
    ax[0].set_xlabel(xlabel="Time (ms)", fontsize=18)
    ax[0].set_ylabel(ylabel="Average firing rate", fontsize=15)
    for xtick in ax[0].xaxis.get_major_ticks():
        xtick.label1.set_fontsize(15)
    for ytick in ax[0].yaxis.get_major_ticks():
        ytick.label1.set_fontsize(15)
    for xtick in ax[1].xaxis.get_major_ticks():
        xtick.label1.set_fontsize(15)
    ax2[1].legend(
        fontsize=15,
        scatterpoints=5,
        columnspacing=0.5,
        framealpha=0,
        loc="upper right",
    )
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)
    fig.suptitle(
        "%s: %s %d " % (neu.area.upper(), neu.cluster_group, neu.cluster_number),
        x=0.05,
        y=0.99,
        fontsize=15,
    )
    return fig


def plot_sp_b2(neu, sp_pos, conv_pos, max_n_tr, conv_max, visual_rf=True, inout=1):
    start = -200
    PAC = task_constants.CODES_AND_POS
    # Get samples position in b1 and b2
    b1_mask = neu.block == 1
    b2_mask = neu.block == 2
    in_mask = neu.pos_code == 1
    u_pos, u_count = np.unique(
        neu.position[np.logical_and(b1_mask, in_mask)], axis=0, return_counts=True
    )
    imax = np.argmax(u_count)
    x_pos_b1, y_pos_b1 = u_pos[imax][0][0], u_pos[imax][0][1]
    pos_b2 = np.unique(neu.position[b2_mask][:, 0], axis=0)
    x_pos_b2, y_pos_b2 = pos_b2[:, 0], pos_b2[:, 1]
    # Start plot
    fig, axs = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)
    axs.set_axis_off()
    for code in sp_pos.keys():
        axis = PAC[code][1]
        sp = sp_pos[code]
        conv = conv_pos[code]
        ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
        time = np.arange(0, len(conv)) + start
        ax2 = ax.twinx()
        # ----- plot conv----------
        ax.plot(time, conv, color="navy")
        # ----- plot spikes----------
        rows, cols = np.where(sp >= 1)
        cols = cols + start
        rows = rows + rows * 2
        ax2.scatter(cols, rows, marker="|", alpha=1, color="grey")
        ax.set_title("Code %s" % (code), fontsize=8)
        ax.set_ylim(0, conv_max + max_n_tr * 3)
        ax.set_yticks(np.arange(0, conv_max, 10))
        ax2.set_ylim(-conv_max, max_n_tr)
        ax2.set_yticks(np.arange(-conv_max, max_n_tr * 3, 10))
        if visual_rf:
            bl_sp = np.mean(sp[:, :200], axis=1)
            tg_sp = np.mean(sp[:, 200:400], axis=1)
            n_tr = len(bl_sp)
            fr = np.sum(tg_sp != 0)
            if n_tr < 5 or fr < 5:
                p = False
            else:
                p = stats.ttest_rel(bl_sp, tg_sp)
                p = p[1] < 0.05
            if p:
                ax.set_facecolor("bisque")
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax.vlines(
            [0, 100, 1100],
            0,
            conv_max + max_n_tr * 3,
            color="k",
            linestyles="dashed",
        )
        if code == "122":
            ax.set_ylabel(ylabel="Average firing rate", fontsize=10, loc="bottom")
            ax.set_xlabel(xlabel="Time (s)", fontsize=10)
    ax = plt.subplot2grid((3, 3), (1, 1))
    ax.scatter(x_pos_b2, y_pos_b2)
    ax.scatter(inout * x_pos_b1, inout * y_pos_b1)
    fig.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
    fig.suptitle(
        "%s: %s %s %d "
        % (neu.date_time, neu.area, neu.cluster_group, neu.cluster_number),
        x=0.5,
        y=1.05,
        fontsize=12,
    )
    return fig


# import numpy as np
# from matplotlib import pyplot as plt
# import pandas as pd
# from typing import Dict, Optional
# from ephysvibe.structures.spike_data import SpikeData
# from ephysvibe.structures.bhv_data import BhvData
# from collections import defaultdict
# from scipy import stats

# seed = 2023


# def get_trials(code, target_codes):
#     code_order = np.array(["127", "126", "125", "124", "123", "122", "121", "120"])
#     code_pos = np.where(code_order == code)[0][0]
#     next = code_pos + 1
#     if next == len(code_order):
#         next = 0
#     next = code_order[next]
#     prev = code_pos - 1
#     prev = code_order[prev]
#     next_idx = np.array(target_codes[next]["trial_idx"]).astype(int)
#     prev_idx = np.array(target_codes[prev]["trial_idx"]).astype(int)
#     return next_idx, prev_idx


# def plot_activity_location(
#     target_codes: Dict,
#     code_samples: np.ndarray,
#     code_numbers: np.ndarray,
#     sp_samples: np.ndarray,
#     i_n: int,
#     e_code_align: int,
#     t_before: int,
#     fs_ds: int,
#     kernel: np.ndarray,
#     rf_t_test: Optional[pd.DataFrame] = pd.DataFrame(),
# ):
#     all_ax, all_ax2 = [], []
#     all_max_conv, max_num_trials = 0, 0
#     for code in target_codes.keys():
#         target_t_idx = target_codes[code][
#             "trial_idx"
#         ]  # select trials with the same stimulus
#         trials_s_on = code_samples[
#             target_t_idx,
#             np.where(code_numbers[target_t_idx] == e_code_align)[1],
#         ]  # moment e_align occurs in each trial
#         shift_sp = SpikeData.indep_roll(
#             sp_samples[target_t_idx, i_n],
#             -(trials_s_on - t_before).astype(int),
#             axis=1,
#         )  # align trials on event
#         # select trials with at least 1 spike
#         shift_sp = shift_sp[np.nansum(shift_sp[:, : t_before + 1500], axis=1) > 0][
#             :, : t_before + 1500
#         ]
#         mean_sp = shift_sp.mean(axis=0)  # mean of all trials
#         conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
#         conv_max = max(conv)
#         all_max_conv = conv_max if conv_max > all_max_conv else all_max_conv
#         num_trials = shift_sp.shape[0]
#         max_num_trials = num_trials if num_trials > max_num_trials else max_num_trials
#         axis = target_codes[code]["position_codes"]
#         ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
#         time = np.arange(0, len(conv)) - t_before
#         # ----- plot ----------
#         ax2 = ax.twinx()
#         ax.plot(time, conv, color="navy")
#         num_trials = shift_sp.shape[0]
#         rows, cols = np.where(shift_sp >= 1)
#         cols = cols - t_before
#         rows = rows + rows * 2
#         ax2.scatter(cols, rows, marker="|", alpha=1, color="grey")
#         # ax.set_title("Code %s" % (code), fontsize=8)
#         if not rf_t_test.empty:
#             if (
#                 code
#                 in rf_t_test[
#                     (rf_t_test["array_position"] == i_n)  # )& (rf_t_test["p"] < 0.05
#                 ]["code"].values
#             ):
#                 ax.set_facecolor("bisque")
#             if code in rf_t_test[(rf_t_test["array_position"] == i_n)]["code"].values:
#                 vm_index = rf_t_test[
#                     (rf_t_test["array_position"] == i_n) & (rf_t_test["code"] == code)
#                 ]["vm_index"].values[0]
#                 ax.set_title("Code %s  vm_index: %.2f" % (code, vm_index), fontsize=8)

#         all_ax.append(ax)
#         all_ax2.append(ax2)
#     return all_ax, all_ax2, all_max_conv, max_num_trials


# def get_neurons_info(
#     sp_samples: np.ndarray,
#     dur_fix: int,
#     neuron_type: np.ndarray,
#     target_codes: Dict,
#     ipsi: np.ndarray,
#     st_v: int,
#     end_v: int,
#     st_d: int,
#     end_d: int,
#     neuron_idx: np.ndarray = None,
#     min_trials: int = 3,
# ) -> pd.DataFrame:
#     """Check if neurons respond to the task.
#     Compares baseline activity with activity during the task.

#     Args:
#         sp_samples (np.ndarray): _description_
#         dur_fix (int): baseline duration. [ms].
#         neuron_type (np.ndarray): array containing whether is a neuron or mua.
#         target_codes (Dict): Dictionary with the position of stimuli as the key containing the idx of the trials.
#         ipsi (np.ndarray): array containing the ipsilateral codes.
#         st_v (int): start of the visual epoch [ms].
#         end_v (int): end of the visual epoch [ms].
#         st_d (int): start of the delay epoch [ms].
#         end_d (int): end of the delay epoch [ms].
#         neuron_idx (np.ndarray, optional): array containing neurons to be analysed. Defaults to None (all neurons).
#         min_trials (int, optional): Min number of trials required to take the neuron into account. Defaults to 3.

#     Returns:
#         pd.DataFrame: df with the results for each neuron.
#     """
#     if neuron_idx is None:
#         neuron_idx = np.arange(0, len(neuron_type))
#     codes = target_codes.keys()
#     neurons_info: Dict[str, list] = defaultdict(list)
#     i_good, i_mua, n_type = 0, 0, 0
#     for i_neuron, type_neuron in zip(neuron_idx, neuron_type):
#         # check and count type of unit
#         if type_neuron == "good":
#             i_good += 1
#             n_type = i_good
#         elif type_neuron == "mua":
#             i_mua += 1
#             n_type = i_mua
#         for code in codes:  # iterate by code'
#             trial_idx = np.array(target_codes[code]["trial_idx"]).astype(int)
#             if len(trial_idx) != 0:
#                 trial_idx = trial_idx[
#                     (np.nansum(sp_samples[trial_idx, i_neuron], axis=1) > 0)
#                 ]
#             n_tr = len(trial_idx)
#             larger = False
#             p = None
#             if n_tr <= min_trials:  # if less than x tr, use tr from adjacent locations
#                 next_idx, prev_idx = get_trials(code, target_codes)
#                 if len(next_idx) != 0:
#                     next_idx = next_idx[
#                         (sp_samples[next_idx, i_neuron].sum(axis=1) > 0)
#                     ]
#                 if len(prev_idx) != 0:
#                     prev_idx = prev_idx[
#                         (sp_samples[prev_idx, i_neuron].sum(axis=1) > 0)
#                     ]
#                 n_tr_min = np.min([len(next_idx), len(prev_idx)])
#                 rng = np.random.default_rng(seed=seed)
#                 next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
#                 prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
#                 trial_idx = np.concatenate([trial_idx, next_idx, prev_idx]).astype(int)
#             n_tr = len(trial_idx)
#             if n_tr >= min_trials:  # if enough tr, compute p value
#                 mean_visual = sp_samples[
#                     trial_idx, i_neuron, dur_fix + st_v : dur_fix + end_v
#                 ].mean(axis=1)
#                 mean_prep = sp_samples[
#                     trial_idx, i_neuron, dur_fix + st_d : dur_fix + end_d
#                 ].mean(axis=1)
#                 mean_bl = sp_samples[trial_idx, i_neuron, :dur_fix].mean(axis=1)
#                 v_larger = mean_bl.mean() < mean_visual.mean()
#                 p_larger = mean_bl.mean() < mean_prep.mean()
#                 larger = v_larger or p_larger
#                 p_v = stats.ttest_rel(mean_bl, mean_visual)[1]
#                 p_p = stats.ttest_rel(mean_bl, mean_prep)[1]
#                 p = np.min([p_v, p_p])
#             if code in ipsi:
#                 laterality = "ipsi"
#             else:
#                 laterality = "contra"
#             neurons_info["code"] += [code]
#             neurons_info["p"] += [p]
#             neurons_info["larger"] += [larger]
#             neurons_info["laterality"] += [laterality]
#             neurons_info["cluster"] += [n_type]
#             neurons_info["group"] += [type_neuron]
#             neurons_info["array_position"] += [i_neuron]
#     neurons_info = pd.DataFrame(neurons_info)
#     return neurons_info


# def get_responding_neurons(
#     neurons_info: pd.DataFrame,
#     epochs: Dict,
#     before_trial: int,
#     code_samples: np.ndarray,
#     code_numbers: np.ndarray,
#     sp_samples: np.ndarray,
#     align_event: int,
#     target_codes: Dict,
# ) -> pd.DataFrame:
#     end_time = np.array(epochs["end_time"]) + before_trial
#     start_time = np.array(epochs["start_time"]) + before_trial
#     test_involved: Dict[str, list] = defaultdict(list)

#     for _, row in neurons_info.iterrows():
#         i_neuron = row["array_position"]
#         code = row["code"]
#         for i_st, i_end, i_epoch in zip(
#             start_time, end_time, epochs["name"]
#         ):  # iterate by event
#             target_t_idx = target_codes[code][
#                 "trial_idx"
#             ]  # select trials with the same stimulus location
#             trials_event_time = code_samples[
#                 target_t_idx, np.where(code_numbers[target_t_idx] == align_event)[1]
#             ]  # moment when the target_on ocurrs in each trial
#             shift_sp = SpikeData.indep_roll(
#                 sp_samples[target_t_idx, i_neuron],
#                 -(trials_event_time - before_trial).astype(int),
#                 axis=1,
#             )  # align trials with (target_on - before_trial)
#             # select trials with at least  5sp/sec
#             shift_sp = shift_sp[
#                 np.nansum(shift_sp[:, before_trial : before_trial + 1100], axis=1) > 0
#             ]
#             # check number of trials
#             if (
#                 shift_sp.shape[0] > 1
#             ):  # if at least 2 trials, compute and save t-test in pd.DataFrame
#                 # mean fr during event
#                 mean_sp = (
#                     shift_sp[:, i_st:i_end].sum(axis=0)
#                     / shift_sp.shape[0]
#                     * (i_end - i_st)
#                 )  # Average fr of all trials
#                 # mean fr during fixation
#                 mean_sp_fix = (
#                     shift_sp[:, :before_trial].sum(axis=0)
#                     / shift_sp.shape[0]
#                     * before_trial
#                 )  # Average fr of all trials
#                 p = stats.ttest_ind(mean_sp, mean_sp_fix, equal_var=False)[1]
#                 message = ""
#             else:
#                 p = np.nan
#                 message = "less than %s spikes" % 0
#             test_involved["code"] += [code]
#             test_involved["laterality"] += [row["laterality"]]
#             test_involved["cluster"] += [row["cluster"]]
#             test_involved["group"] += [row["group"]]
#             test_involved["array_position"] += [i_neuron]
#             test_involved["event"] += [i_epoch]
#             test_involved["p"] += [p]
#             test_involved["message"] += [message]
#     test_involved = pd.DataFrame(test_involved)

#     return test_involved


# def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
#     d_shape = data.shape
#     d_avg = np.zeros((d_shape[0], int(np.floor(d_shape[1] / step))))
#     count = 0
#     for i_step in np.arange(0, d_shape[1] - step, step):
#         d_avg[:, count] = np.mean(data[:, i_step : i_step + win], axis=1)
#         count += 1
#     return d_avg


# def get_rf(
#     th_involved: pd.DataFrame,
#     sp_samples: np.ndarray,
#     ipsi: np.ndarray,
#     contra: np.ndarray,
#     target_codes: Dict,
#     dur_fix: int,
#     st_v: int,
#     end_v: int,
#     st_d: int,
#     end_d: int,
#     min_trials: int = 3,
# ) -> pd.DataFrame:
#     test_rf: Dict[str, list] = defaultdict(list)
#     for _, row in th_involved.iterrows():
#         p, p_v, p_p = np.nan, np.nan, np.nan
#         larger, v_larger, p_larger = False, False, False
#         i_neuron = row["array_position"]
#         code = row["code"]

#         if code in ipsi:
#             idx = np.where(ipsi == code)[0]
#             opposite_code = contra[idx][0]
#         else:
#             idx = np.where(contra == code)[0]
#             opposite_code = ipsi[idx][0]
#         # code
#         code_t_idx = np.array(
#             target_codes[code]["trial_idx"], dtype=int
#         )  # select trials with the same stimulus
#         if len(code_t_idx) != 0:
#             code_t_idx = code_t_idx[(sp_samples[code_t_idx, i_neuron].sum(axis=1) > 0)]
#         if code_t_idx.shape[0] < min_trials:
#             next_idx, prev_idx = get_trials(code, target_codes)
#             next_idx = next_idx[(sp_samples[next_idx, i_neuron].sum(axis=1) > 0)]
#             prev_idx = prev_idx[(sp_samples[prev_idx, i_neuron].sum(axis=1) > 0)]
#             n_tr_min = np.min([len(next_idx), len(prev_idx)])

#             rng = np.random.default_rng(seed=seed)
#             next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
#             prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
#             code_t_idx = np.concatenate([code_t_idx, next_idx, prev_idx])
#             if len(code_t_idx) != 0:
#                 code_t_idx = code_t_idx[
#                     (sp_samples[code_t_idx, i_neuron].sum(axis=1) > 0)
#                 ]

#         # opposite_code
#         oppos_t_idx = np.array(
#             target_codes[opposite_code]["trial_idx"], dtype=int
#         )  # select trials with the same stimulus
#         if len(oppos_t_idx) != 0:
#             oppos_t_idx = oppos_t_idx[
#                 (sp_samples[oppos_t_idx, i_neuron].sum(axis=1) > 0)
#             ]
#         if oppos_t_idx.shape[0] < min_trials:
#             next_idx, prev_idx = get_trials(opposite_code, target_codes)
#             next_idx = next_idx[(sp_samples[next_idx, i_neuron].sum(axis=1) > 0)]
#             prev_idx = prev_idx[(sp_samples[prev_idx, i_neuron].sum(axis=1) > 0)]
#             n_tr_min = np.min([len(next_idx), len(prev_idx)])
#             rng = np.random.default_rng(seed=seed)
#             next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
#             prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
#             oppos_t_idx = np.concatenate([oppos_t_idx, next_idx, prev_idx])
#             if len(oppos_t_idx) != 0:
#                 oppos_t_idx = oppos_t_idx[
#                     (sp_samples[oppos_t_idx, i_neuron].sum(axis=1) > 0)
#                 ]

#         if code_t_idx.shape[0] >= min_trials and oppos_t_idx.shape[0] >= min_trials:
#             sp_code = sp_samples[code_t_idx, i_neuron, dur_fix:]
#             sp_oppos = sp_samples[oppos_t_idx, i_neuron, dur_fix:]
#             # visual
#             mean_sp_code = sp_code[:, st_v:end_v].mean(axis=1)
#             mean_sp_opposite = sp_oppos[:, st_v:end_v].mean(axis=1)
#             p_v = stats.ttest_ind(mean_sp_code, mean_sp_opposite)[1]
#             v_larger = mean_sp_code.mean() > mean_sp_opposite.mean()
#             # preparatory
#             mean_sp_code = sp_code[:, st_d:end_d].mean(axis=1)
#             mean_sp_opposite = sp_oppos[:, st_d:end_d].mean(axis=1)
#             p_p = stats.ttest_ind(mean_sp_code, mean_sp_opposite)[1]
#             p_larger = mean_sp_code.mean() > mean_sp_opposite.mean()
#             p = np.min([p_v, p_p])
#             larger = v_larger or p_larger

#         test_rf["code"] += [code]
#         test_rf["array_position"] += [i_neuron]
#         test_rf["p"] += [p]
#         test_rf["larger"] += [larger]
#         test_rf["p_v"] += [p_v]
#         test_rf["v_larger"] += [v_larger]
#         test_rf["p_p"] += [p_p]
#         test_rf["p_larger"] += [p_larger]
#         test_rf["cluster"] += [row["cluster"]]
#         test_rf["group"] += [row["group"]]
#     test_rf = pd.DataFrame(test_rf)
#     return test_rf


# def get_vm_index(
#     th_rf: pd.DataFrame,
#     target_codes: Dict,
#     sp_samples: np.ndarray,
#     dur_fix: int,
#     st_v: int,
#     end_v: int,
#     st_d: int,
#     end_d: int,
#     min_trials: int,
# ):
#     test_vm: Dict[str, list] = defaultdict(list)
#     for _, row in th_rf.iterrows():
#         i_neuron = row["array_position"]
#         code = row["code"]
#         vm_index = np.nan
#         sig_type = np.nan
#         # select trials

#         target_t_idx = np.array(
#             target_codes[code]["trial_idx"], dtype=int
#         )  # select trials with the same stimulus
#         if len(target_t_idx) != 0:
#             target_t_idx = target_t_idx[
#                 (sp_samples[target_t_idx, i_neuron].sum(axis=1) > 0)
#             ]
#         all_trials_sp = []
#         for i_code in target_codes.keys():
#             all_trials = np.array(
#                 target_codes[i_code]["trial_idx"], dtype=int
#             )  # select trials with the same stimulus
#             if len(all_trials) != 0:
#                 all_trials = all_trials[
#                     (sp_samples[all_trials, i_neuron].sum(axis=1) > 0)
#                 ]
#             all_trials = sp_samples[all_trials, i_neuron, dur_fix:]  # .mean(axis=0)

#             all_trials_sp.append(
#                 [all_trials[:, st_v:end_v].mean(), all_trials[:, st_d:end_d].mean()]
#             )

#         all_trials_sp = np.concatenate(all_trials_sp)
#         min_sp = np.nanmin(all_trials_sp)
#         max_sp = np.nanmax(all_trials_sp)  # - min_sp

#         if target_t_idx.shape[0] <= min_trials:
#             next_idx, prev_idx = get_trials(code, target_codes)
#             n_tr_min = np.min([len(next_idx), len(prev_idx)])
#             rng = np.random.default_rng(seed=seed)
#             next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
#             prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
#             target_t_idx = np.concatenate([target_t_idx, next_idx, prev_idx])
#             target_t_idx = target_t_idx[
#                 (sp_samples[target_t_idx, i_neuron].sum(axis=1) > 0)
#             ]
#         if target_t_idx.shape[0] >= min_trials:
#             sp_code = sp_samples[target_t_idx, i_neuron, dur_fix:]
#             sp_trial_avg = sp_code.mean(axis=0)
#             v_mean = (sp_trial_avg[st_v:end_v].mean()) / max_sp  # - min_sp
#             m_mean = (sp_trial_avg[st_d:end_d].mean()) / max_sp  # - min_sp

#             vm_index = (m_mean - v_mean) / (v_mean + m_mean)
#             if vm_index <= 0:
#                 sig_type = "visual"
#             else:
#                 sig_type = "anticipation"
#         # save results
#         test_vm["code"] += [code]
#         test_vm["array_position"] += [i_neuron]
#         test_vm["vm_index"] += [vm_index]
#         test_vm["sig_type"] += [sig_type]
#         test_vm["cluster"] += [row["cluster"]]
#         test_vm["group"] += [row["group"]]
#         test_vm["p_v"] += [row["p_v"]]
#         test_vm["v_larger"] += [row["v_larger"]]
#         test_vm["p_p"] += [row["p_p"]]
#         test_vm["p_larger"] += [row["p_larger"]]
#     test_vm = pd.DataFrame(test_vm)
#     return test_vm


# def get_laterality_idx(
#     th_involved: pd.DataFrame,
#     sp_samples: np.ndarray,
#     ipsi: np.ndarray,
#     contra: np.ndarray,
#     target_codes: Dict,
#     code_samples: np.ndarray,
#     align_event: int,
#     code_numbers: np.ndarray,
#     dur_v: int,
#     st_m: int,
#     end_m: int,
#     kernel: np.ndarray,
#     fs_ds: int,
# ) -> pd.DataFrame:
#     lat_index_df: Dict[str, list] = defaultdict(list)
#     n_involved = th_involved["array_position"].unique()
#     contra_idx = np.concatenate(pd.DataFrame(target_codes).iloc[1][contra].values)
#     ipsi_idx = np.concatenate(pd.DataFrame(target_codes).iloc[1][ipsi].values)
#     for i_neuron in n_involved:
#         # contra
#         trials_event_time = code_samples[
#             contra_idx, np.where(code_numbers[contra_idx] == align_event)[1]
#         ]  # moment when the target_on ocurrs in each trial
#         sp_contra = SpikeData.indep_roll(
#             sp_samples[contra_idx, i_neuron], -(trials_event_time).astype(int), axis=1
#         )  # align trials with event onset
#         sp_contra = sp_contra[
#             np.nansum(sp_contra, axis=1) > 0
#         ]  # Select trials with at least one spike
#         # ipsi
#         trials_event_time = code_samples[
#             ipsi_idx, np.where(code_numbers[ipsi_idx] == align_event)[1]
#         ]  # moment when the target_on ocurrs in each trial
#         sp_ipsi = SpikeData.indep_roll(
#             sp_samples[ipsi_idx, i_neuron], -(trials_event_time).astype(int), axis=1
#         )  # align trials with event onset
#         sp_ipsi = sp_ipsi[
#             np.nansum(sp_ipsi, axis=1) > 0
#         ]  # Select trials with at least one spike
#         # Average fr of all trials
#         mean_sp_contra = np.nanmean(sp_contra[:, :1100], axis=0)
#         mean_sp_ipsi = np.nanmean(sp_ipsi[:, :1100], axis=0)
#         # convolution
#         conv_contra = np.convolve(mean_sp_contra, kernel, mode="same") * fs_ds
#         conv_ipsi = np.convolve(mean_sp_ipsi, kernel, mode="same") * fs_ds
#         min_value = np.concatenate([conv_contra, conv_ipsi]).min()
#         conv_contra = np.nanmean(conv_contra - min_value)
#         conv_ipsi = np.nanmean(conv_ipsi - min_value)
#         p = stats.ttest_ind(mean_sp_contra, mean_sp_ipsi)[1]
#         lat_index = (conv_contra - conv_ipsi) / (conv_contra + conv_ipsi)

#         lat_index_df["array_position"] += [i_neuron]
#         lat_index_df["lat_index"] += [lat_index]
#         lat_index_df["p_lat"] += [p]

#     lat_index_df = pd.DataFrame(lat_index_df)
#     return lat_index_df


# def get_max_fr(
#     target_codes,
#     sp_samples,
#     code_samples,
#     code_numbers,
#     i_n,
#     kernel,
#     dur_fix,
#     end_v,
#     e_code_align,
#     test_vm,
#     fs_ds,
# ):
#     fr_max_visual, fr_max_motor, fr_angle, fr_max_codes = [], [], [], []
#     v_significant, m_significant = [], []
#     for code in target_codes.keys():
#         target_t_idx = target_codes[code][
#             "trial_idx"
#         ]  # select trials with the same stimulus
#         trials_s_on = code_samples[
#             target_t_idx, np.where(code_numbers[target_t_idx] == e_code_align)[1]
#         ]
#         shift_sp = SpikeData.indep_roll(
#             sp_samples[target_t_idx, i_n], -(trials_s_on).astype(int), axis=1
#         )  # align trials on event
#         # select trials with at least one spike
#         shift_sp = shift_sp[
#             np.nansum(shift_sp[:, : 1100 + dur_fix], axis=1) > 0, dur_fix:
#         ]
#         mean_sp = np.nanmean(shift_sp, axis=0)  # mean of all trials
#         if shift_sp.shape[0] == 0:
#             conv = np.zeros((1100))
#         else:
#             conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
#         fr_max_visual.append(np.nanmax(conv[50:end_v]))
#         fr_angle.append(target_codes[code]["angle_codes"])
#         fr_max_motor.append(np.nanmax(conv[700:1100]))
#         fr_max_codes.append(np.nanmax(conv[:1100]))
#         if (
#             code
#             in test_vm[
#                 (test_vm["array_position"] == i_n) & (test_vm["sig_type"] == "visual")
#             ]["code"].values
#         ):
#             v_significant.append(True)
#         else:
#             v_significant.append(False)
#         if (
#             code
#             in test_vm[
#                 (test_vm["array_position"] == i_n)
#                 & (test_vm["sig_type"] == "anticipation")
#             ]["code"].values
#         ):
#             m_significant.append(True)
#         else:
#             m_significant.append(False)
#     return (
#         np.array(fr_max_visual),
#         np.array(fr_max_motor),
#         np.array(fr_angle),
#         np.array(fr_max_codes),
#         np.array(v_significant),
#         np.array(m_significant),
#     )
