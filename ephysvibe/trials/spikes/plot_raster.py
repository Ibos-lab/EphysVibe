import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, Optional
from ephysvibe.structures.trials_data import TrialsData
from collections import defaultdict
from scipy import stats


def plot_activity_location(
    target_codes: Dict,
    code_samples: np.ndarray,
    code_numbers: np.ndarray,
    sp_samples: np.ndarray,
    i_n: int,
    e_code_align: int,
    t_before: int,
    fs_ds: int,
    kernel: np.ndarray,
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
        )  # align trials on event
        # select trials with at least one spike
        shift_sp = shift_sp[np.nansum(shift_sp, axis=1) > 0][:, :2300]
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
                    (rf_t_test["array_position"] == i_n)  # )& (rf_t_test["p"] < 0.05
                ]["code"].values
            ):
                ax.set_facecolor("bisque")
            if code in rf_t_test[(rf_t_test["array_position"] == i_n)]["code"].values:
                vm_index = rf_t_test[
                    (rf_t_test["array_position"] == i_n) & (rf_t_test["code"] == code)
                ]["vm_index"].values[0]
                ax.set_title("Code %s  vm_index: %.2f" % (code, vm_index), fontsize=8)

        all_ax.append(ax)
        all_ax2.append(ax2)
    return all_ax, all_ax2, all_max_conv, max_num_trials


def get_neurons_info(
    neuron_type: np.ndarray,
    target_codes: Dict,
    ipsi: np.ndarray,
    contra: np.ndarray,
    neuron_idx: np.ndarray = None,
) -> pd.DataFrame:
    if neuron_idx is None:
        neuron_idx = np.arange(0, len(neuron_type))
    codes = target_codes.keys()
    neurons_info: Dict[str, list] = defaultdict(list)
    i_good, i_mua, n_type = 0, 0, 0
    for i_neuron, type_neuron in zip(neuron_idx, neuron_type):
        # check and count type of unit
        if type_neuron == "good":
            i_good += 1
            n_type = i_good
        elif type_neuron == "mua":
            i_mua += 1
            n_type = i_mua
        for code in codes:  # iterate by code'
            if code in ipsi:
                laterality = "ipsi"
            else:
                laterality = "contra"
            neurons_info["code"] += [code]
            neurons_info["laterality"] += [laterality]
            neurons_info["cluster"] += [n_type]
            neurons_info["group"] += [type_neuron]
            neurons_info["array_position"] += [i_neuron]
    neurons_info = pd.DataFrame(neurons_info)
    return neurons_info


def get_responding_neurons(
    neurons_info: pd.DataFrame,
    epochs: Dict,
    before_trial: int,
    code_samples: np.ndarray,
    code_numbers: np.ndarray,
    sp_samples: np.ndarray,
    align_event: int,
    target_codes: Dict,
    n_spikes_sec: np.ndarray = 5,
) -> pd.DataFrame:
    end_time = np.array(epochs["end_time"]) + before_trial
    start_time = np.array(epochs["start_time"]) + before_trial
    test_involved: Dict[str, list] = defaultdict(list)

    for _, row in neurons_info.iterrows():
        i_neuron = row["array_position"]
        code = row["code"]
        for i_st, i_end, i_epoch in zip(
            start_time, end_time, epochs["name"]
        ):  # iterate by event
            target_t_idx = target_codes[code][
                "trial_idx"
            ]  # select trials with the same stimulus location
            trials_event_time = code_samples[
                target_t_idx, np.where(code_numbers[target_t_idx] == align_event)[1]
            ]  # moment when the target_on ocurrs in each trial
            shift_sp = TrialsData.indep_roll(
                sp_samples[target_t_idx, i_neuron],
                -(trials_event_time - before_trial).astype(int),
                axis=1,
            )  # align trials with (target_on - before_trial)
            # select trials with at least one spike
            shift_sp = shift_sp[np.nansum(shift_sp, axis=1) > 0]
            # check number of spikes, at least 5/sec
            if np.any(
                np.sum(
                    shift_sp[:, before_trial : epochs["end_time"][-1] + before_trial],
                    axis=1,
                )
                >= n_spikes_sec * (epochs["end_time"][-1] / 1000)
            ):  # if at least n_spikes_sec, compute and save t-test in pd.DataFrame
                # mean fr during event
                mean_sp = shift_sp[:, i_st:i_end].mean(
                    axis=0
                )  # Average fr of all trials
                # mean fr during fixation
                mean_sp_fix = shift_sp[:, :before_trial].mean(
                    axis=0
                )  # Average fr of all trials
                p = stats.ttest_ind(mean_sp, mean_sp_fix)[1]
                message = ""
            else:
                p = np.nan
                message = "less than %s spikes/sec" % n_spikes_sec
            test_involved["code"] += [code]
            test_involved["laterality"] += [row["laterality"]]
            test_involved["cluster"] += [row["cluster"]]
            test_involved["group"] += [row["group"]]
            test_involved["array_position"] += [i_neuron]
            test_involved["event"] += [i_epoch]
            test_involved["p"] += [p]
            test_involved["message"] += [message]
    test_involved = pd.DataFrame(test_involved)

    return test_involved


def get_rf(
    th_involved: pd.DataFrame,
    sp_samples: np.ndarray,
    ipsi: np.ndarray,
    contra: np.ndarray,
    target_codes: Dict,
    code_samples: np.ndarray,
    align_event: int,
    code_numbers: np.ndarray,
    dur_v: int,
    st_m: int,
    end_m: int,
) -> pd.DataFrame:
    test_rf: Dict[str, list] = defaultdict(list)
    for _, row in th_involved.iterrows():
        i_neuron = row["array_position"]
        code = row["code"]
        event = row["event"]
        if code in ipsi:
            idx = np.where(ipsi == code)[0]
            opposite_code = contra[idx][0]
        else:
            idx = np.where(contra == code)[0]
            opposite_code = ipsi[idx][0]
        # code
        target_t_idx = target_codes[code][
            "trial_idx"
        ]  # select trials with the same stimulus
        trials_event_time = code_samples[
            target_t_idx, np.where(code_numbers[target_t_idx] == align_event)[1]
        ]  # moment when the target_on ocurrs in each trial
        shift_sp_r = TrialsData.indep_roll(
            sp_samples[target_t_idx, i_neuron], -(trials_event_time).astype(int), axis=1
        )  # align trials with event onset
        shift_sp_r = shift_sp_r[
            np.nansum(shift_sp_r, axis=1) > 0
        ]  # Select trials with at least one spike
        # opposite_code
        target_t_idx = target_codes[opposite_code][
            "trial_idx"
        ]  # select trials with the same stimulus
        trials_event_time = code_samples[
            target_t_idx, np.where(code_numbers[target_t_idx] == align_event)[1]
        ]  # moment when the target_on ocurrs in each trial
        shift_sp_l = TrialsData.indep_roll(
            sp_samples[target_t_idx, i_neuron], -(trials_event_time).astype(int), axis=1
        )  # align trials with event onset
        shift_sp_l = shift_sp_l[
            np.nansum(shift_sp_l, axis=1) > 0
        ]  # Select trials with at least one spike
        # Average fr of all trials
        if event == "visual":  # visuel
            mean_sp_code = shift_sp_r[:, :dur_v].mean(axis=0)
            mean_sp_opposite = shift_sp_l[:, :dur_v].mean(axis=0)
        elif event == "anticipation":  # motor
            mean_sp_code = shift_sp_r[:, st_m:end_m].mean(axis=0)
            mean_sp_opposite = shift_sp_l[:, st_m:end_m].mean(axis=0)
        else:  # i_vm_idx <= -vm_threshold: # visuomotor
            mean_sp_code = shift_sp_r[:, :1100].mean(axis=0)
            mean_sp_opposite = shift_sp_l[:, :1100].mean(axis=0)
        p = stats.ttest_ind(mean_sp_code, mean_sp_opposite)[1]
        larger = mean_sp_code.mean() > mean_sp_opposite.mean()
        test_rf["code"] += [code]
        test_rf["array_position"] += [i_neuron]
        test_rf["p"] += [p]
        test_rf["larger"] += [larger]
        test_rf["type"] += [event]
        test_rf["cluster"] += [row["cluster"]]
        test_rf["group"] += [row["group"]]
    test_rf = pd.DataFrame(test_rf)
    return test_rf


def get_vm_index(
    th_rf,
    target_codes,
    code_samples,
    code_numbers,
    sp_samples,
    align_event,
    fix_t,
    dur_v,
    st_m,
    end_m,
    fs_ds,
    kernel,
):
    test_vm: Dict[str, list] = defaultdict(list)
    for _, row in th_rf.iterrows():
        i_neuron = row["array_position"]
        code = row["code"]
        target_t_idx = target_codes[code][
            "trial_idx"
        ]  # select trials with the same stimulus
        # select trials
        trials_event_time = code_samples[
            target_t_idx, np.where(code_numbers[target_t_idx] == align_event)[1]
        ]  # moment when the target_on ocurrs in each trial
        shift_sp = TrialsData.indep_roll(
            sp_samples[target_t_idx, i_neuron],
            -(trials_event_time - fix_t).astype(int),
            axis=1,
        )  # align trials with event onset
        shift_sp = shift_sp[
            np.nansum(shift_sp, axis=1) > 0
        ]  # select trials with at least one spike
        conv = (
            np.convolve(
                shift_sp[:, : fix_t + end_m + 100].mean(axis=0), kernel, mode="same"
            )
            * fs_ds
        )
        m_mean = (
            conv[fix_t + st_m : fix_t + end_m].mean()
            - conv[fix_t : fix_t + end_m].min()
        )
        v_mean = conv[fix_t : fix_t + dur_v].mean() - conv[fix_t : fix_t + end_m].min()
        vm_index = (m_mean - v_mean) / (v_mean + m_mean)
        # save results
        test_vm["code"] += [code]
        test_vm["array_position"] += [i_neuron]
        test_vm["vm_index"] += [vm_index]
        test_vm["sig_type"] += [row["type"]]
        test_vm["cluster"] += [row["cluster"]]
        test_vm["group"] += [row["group"]]
    test_vm = pd.DataFrame(test_vm)
    return test_vm


def get_max_fr(
    target_codes,
    sp_samples,
    code_samples,
    code_numbers,
    i_n,
    kernel,
    win_size,
    dur_v,
    e_code_align,
    test_vm,
    fs_ds,
):
    (
        fr_max_visual,
        fr_max_motor,
        fr_angle,
        fr_max_trial,
        v_significant,
        m_significant,
    ) = ([], [], [], [], [], [])
    for code in target_codes.keys():
        target_t_idx = target_codes[code][
            "trial_idx"
        ]  # select trials with the same stimulus
        trials_s_on = code_samples[
            target_t_idx, np.where(code_numbers[target_t_idx] == e_code_align)[1]
        ]
        shift_sp = TrialsData.indep_roll(
            sp_samples[target_t_idx, i_n], -(trials_s_on - win_size).astype(int), axis=1
        )  # align trials on event
        # select trials with at least one spike
        shift_sp = shift_sp[np.nansum(shift_sp, axis=1) > 0]
        mean_sp = shift_sp.mean(axis=0)  # mean of all trials
        conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
        fr_max_visual.append(max(conv[win_size : win_size + dur_v]))
        fr_angle.append(target_codes[code]["angle_codes"])
        fr_max_motor.append(max(conv[win_size + 800 : win_size + 1100]))
        fr_max_trial.append(max(conv[win_size : win_size + 1100]))
        if (
            code
            in test_vm[
                (test_vm["array_position"] == i_n) & (test_vm["sig_type"] == "visual")
            ]["code"].values
        ):
            v_significant.append(True)
        else:
            v_significant.append(False)
        if (
            code
            in test_vm[
                (test_vm["array_position"] == i_n)
                & (test_vm["sig_type"] == "anticipation")
            ]["code"].values
        ):
            m_significant.append(True)
        else:
            m_significant.append(False)
    return (
        fr_max_visual,
        fr_max_motor,
        fr_angle,
        fr_max_trial,
        v_significant,
        m_significant,
    )
