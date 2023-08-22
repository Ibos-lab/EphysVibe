import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Dict, Optional
from ephysvibe.structures.spike_data import SpikeData
from ephysvibe.structures.bhv_data import BhvData
from collections import defaultdict
from scipy import stats

seed = 2023


def get_trials(code, target_codes):

    code_order = np.array(["127", "126", "125", "124", "123", "122", "121", "120"])
    code_pos = np.where(code_order == code)[0][0]
    next = code_pos + 1
    if next == len(code_order):
        next = 0
    next = code_order[next]
    prev = code_pos - 1
    prev = code_order[prev]
    next_idx = np.array(target_codes[next]["trial_idx"])
    prev_idx = np.array(target_codes[prev]["trial_idx"])
    return next_idx, prev_idx


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
        shift_sp = SpikeData.indep_roll(
            sp_samples[target_t_idx, i_n],
            -(trials_s_on - t_before).astype(int),
            axis=1,
        )  # align trials on event
        # select trials with at least 5 spikes/sec
        shift_sp = shift_sp[
            np.nansum(shift_sp[:, t_before : t_before + 1100], axis=1) > 5 * 1100 / 1000
        ][:, : t_before + 1100]
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
    sp_samples: np.ndarray,
    fix_t: int,
    neuron_type: np.ndarray,
    target_codes: Dict,
    ipsi: np.ndarray,
    contra: np.ndarray,
    dur_v,
    st_m,
    end_m,
    neuron_idx: np.ndarray = None,
    min_trials: int = 5,
    n_spikes_sec: int = 5,
) -> pd.DataFrame:
    trial_dur = sp_samples.shape[2]
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
            trial_idx = np.array(target_codes[code]["trial_idx"])
            trial_idx = trial_idx[
                (
                    sp_samples[trial_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * trial_dur / 1000
                )
            ]
            n_tr = len(trial_idx)
            larger = False
            p = None
            if n_tr <= min_trials:  # if less than x tr, use tr from adjacent locations
                next_idx, prev_idx = get_trials(code, target_codes)
                next_idx = next_idx[
                    (
                        sp_samples[next_idx, i_neuron].sum(axis=1)
                        > n_spikes_sec * trial_dur / 1000
                    )
                ]
                prev_idx = prev_idx[
                    (
                        sp_samples[prev_idx, i_neuron].sum(axis=1)
                        > n_spikes_sec * trial_dur / 1000
                    )
                ]
                n_tr_min = np.min([len(next_idx), len(prev_idx)])
                rng = np.random.default_rng(seed=seed)
                next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
                prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
                trial_idx = np.concatenate([trial_idx, next_idx, prev_idx])
            n_tr = len(trial_idx)
            if n_tr >= min_trials:  # if enough tr, compute p value
                mean_visual = sp_samples[
                    trial_idx, i_neuron, fix_t : fix_t + dur_v
                ].mean(axis=1)
                mean_prep = sp_samples[
                    trial_idx, i_neuron, fix_t + st_m : fix_t + end_m
                ].mean(axis=1)
                mean_bl = sp_samples[trial_idx, i_neuron, :fix_t].mean(axis=1)
                v_larger = mean_bl.mean() < mean_visual.mean()
                p_larger = mean_bl.mean() < mean_prep.mean()
                larger = v_larger or p_larger
                p_v = stats.ttest_rel(mean_bl, mean_visual)[1]
                p_p = stats.ttest_rel(mean_bl, mean_prep)[1]
                p = np.min([p_v, p_p])
            if code in ipsi:
                laterality = "ipsi"
            else:
                laterality = "contra"
            neurons_info["code"] += [code]
            neurons_info["p"] += [p]
            neurons_info["larger"] += [larger]
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
            shift_sp = SpikeData.indep_roll(
                sp_samples[target_t_idx, i_neuron],
                -(trials_event_time - before_trial).astype(int),
                axis=1,
            )  # align trials with (target_on - before_trial)
            # select trials with at least  5sp/sec
            shift_sp = shift_sp[
                np.nansum(shift_sp[:, before_trial : before_trial + 1100], axis=1)
                > n_spikes_sec * 1100 / 1000
            ]
            # check number of trials
            if (
                shift_sp.shape[0] > 1
            ):  # if at least 2 trials, compute and save t-test in pd.DataFrame
                # mean fr during event
                mean_sp = (
                    shift_sp[:, i_st:i_end].sum(axis=0)
                    / shift_sp.shape[0]
                    * (i_end - i_st)
                )  # Average fr of all trials
                # mean fr during fixation
                mean_sp_fix = (
                    shift_sp[:, :before_trial].sum(axis=0)
                    / shift_sp.shape[0]
                    * before_trial
                )  # Average fr of all trials
                p = stats.ttest_ind(mean_sp, mean_sp_fix, equal_var=False)[1]
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


def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    d_shape = data.shape
    d_avg = np.zeros((d_shape[0], int(np.floor(d_shape[1] / step))))
    count = 0
    for i_step in np.arange(0, d_shape[1] - step, step):
        d_avg[:, count] = np.mean(data[:, i_step : i_step + win], axis=1)
        count += 1
    return d_avg


def get_rf(
    th_involved: pd.DataFrame,
    sp_samples: np.ndarray,
    ipsi: np.ndarray,
    contra: np.ndarray,
    target_codes: Dict,
    dur_v: int,
    st_m: int,
    end_m: int,
    n_spikes_sec: np.ndarray = 5,
    min_trials: int = 5,
) -> pd.DataFrame:

    test_rf: Dict[str, list] = defaultdict(list)
    for _, row in th_involved.iterrows():
        p = np.nan
        larger = False
        i_neuron = row["array_position"]
        code = row["code"]

        if code in ipsi:
            idx = np.where(ipsi == code)[0]
            opposite_code = contra[idx][0]
        else:
            idx = np.where(contra == code)[0]
            opposite_code = ipsi[idx][0]
        # code
        code_t_idx = np.array(
            target_codes[code]["trial_idx"]
        )  # select trials with the same stimulus
        code_t_idx = code_t_idx[
            (
                sp_samples[code_t_idx, i_neuron].sum(axis=1)
                > n_spikes_sec * (1100) / 1000
            )
        ]
        if code_t_idx.shape[0] < min_trials:
            next_idx, prev_idx = get_trials(code, target_codes)
            next_idx = next_idx[
                (
                    sp_samples[next_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]
            prev_idx = prev_idx[
                (
                    sp_samples[prev_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]
            n_tr_min = np.min([len(next_idx), len(prev_idx)])

            rng = np.random.default_rng(seed=seed)
            next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
            prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
            code_t_idx = np.concatenate([code_t_idx, next_idx, prev_idx])
            code_t_idx = code_t_idx[
                (
                    sp_samples[code_t_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]

        # opposite_code
        oppos_t_idx = np.array(
            target_codes[opposite_code]["trial_idx"]
        )  # select trials with the same stimulus
        oppos_t_idx = oppos_t_idx[
            (
                sp_samples[oppos_t_idx, i_neuron].sum(axis=1)
                > n_spikes_sec * (1100) / 1000
            )
        ]
        if oppos_t_idx.shape[0] < min_trials:
            next_idx, prev_idx = get_trials(opposite_code, target_codes)
            next_idx = next_idx[
                (
                    sp_samples[next_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]
            prev_idx = prev_idx[
                (
                    sp_samples[prev_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]
            n_tr_min = np.min([len(next_idx), len(prev_idx)])
            rng = np.random.default_rng(seed=seed)
            next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
            prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
            oppos_t_idx = np.concatenate([oppos_t_idx, next_idx, prev_idx])
            oppos_t_idx = oppos_t_idx[
                (
                    sp_samples[oppos_t_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]

        if code_t_idx.shape[0] >= min_trials and oppos_t_idx.shape[0] >= min_trials:
            sp_code = sp_samples[code_t_idx, i_neuron]
            sp_oppos = sp_samples[oppos_t_idx, i_neuron]
            # visual
            mean_sp_code = sp_code[:, :dur_v].mean(axis=1)
            mean_sp_opposite = sp_oppos[:, :dur_v].mean(axis=1)
            p_v = stats.ttest_ind(mean_sp_code, mean_sp_opposite)[1]
            v_larger = mean_sp_code.mean() > mean_sp_opposite.mean()
            # preparatory
            mean_sp_code = sp_code[:, st_m:end_m].mean(axis=1)
            mean_sp_opposite = sp_oppos[:, st_m:end_m].mean(axis=1)
            p_p = stats.ttest_ind(mean_sp_code, mean_sp_opposite)[1]
            p_larger = mean_sp_code.mean() > mean_sp_opposite.mean()
            p = np.min([p_v, p_p])
            larger = v_larger or p_larger

        test_rf["code"] += [code]
        test_rf["array_position"] += [i_neuron]
        test_rf["p"] += [p]
        test_rf["larger"] += [larger]

        test_rf["cluster"] += [row["cluster"]]
        test_rf["group"] += [row["group"]]
    test_rf = pd.DataFrame(test_rf)
    return test_rf


def get_vm_index(
    th_rf,
    target_codes,
    sp_samples,
    sp_bl,
    align_event,
    fix_t,
    dur_v,
    st_m,
    end_m,
    n_spikes_sec,
    min_trials,
):
    test_vm: Dict[str, list] = defaultdict(list)
    for _, row in th_rf.iterrows():
        i_neuron = row["array_position"]
        code = row["code"]
        vm_index = np.nan
        sig_type = np.nan
        # select trials with the same stimulus
        target_t_idx = np.array(target_codes[code]["trial_idx"])
        target_t_idx = target_t_idx[
            (
                sp_samples[target_t_idx, i_neuron].sum(axis=1)
                > n_spikes_sec * (1100) / 1000
            )
        ]
        all_trials_sp = []
        for i_code in target_codes.keys():
            # select trials with the same stimulus
            all_trials = np.array(target_codes[i_code]["trial_idx"])
            all_trials = all_trials[
                (
                    sp_samples[all_trials, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]
            all_trials = sp_samples[all_trials, i_neuron]
            all_trials_sp.append(
                [all_trials[:, 50:dur_v].mean(), all_trials[:, st_m:end_m].mean()]
            )

        all_trials_sp = np.concatenate(all_trials_sp)
        min_sp = np.nanmin(all_trials_sp)
        max_sp = np.nanmax(all_trials_sp) - min_sp

        if target_t_idx.shape[0] <= min_trials:
            next_idx, prev_idx = get_trials(code, target_codes)
            n_tr_min = np.min([len(next_idx), len(prev_idx)])
            rng = np.random.default_rng(seed=seed)
            next_idx = rng.choice(next_idx, size=n_tr_min, replace=False)
            prev_idx = rng.choice(prev_idx, size=n_tr_min, replace=False)
            target_t_idx = np.concatenate([target_t_idx, next_idx, prev_idx])
            target_t_idx = target_t_idx[
                (
                    sp_samples[target_t_idx, i_neuron].sum(axis=1)
                    > n_spikes_sec * (1100) / 1000
                )
            ]
        if target_t_idx.shape[0] >= min_trials:
            sp_code = sp_samples[
                target_t_idx, i_neuron
            ]  # Select trials with at least  5 spikes/sec

            sp_trial_avg = sp_code.mean(axis=0)
            v_mean = (sp_trial_avg[50:dur_v].mean() - min_sp) / max_sp
            m_mean = (sp_trial_avg[st_m:end_m].mean() - min_sp) / max_sp

            vm_index = (m_mean - v_mean) / (v_mean + m_mean)
            if vm_index <= 0:
                sig_type = "visual"
            else:
                sig_type = "anticipation"
        # save results
        test_vm["code"] += [code]
        test_vm["array_position"] += [i_neuron]
        test_vm["vm_index"] += [vm_index]
        test_vm["sig_type"] += [sig_type]
        test_vm["cluster"] += [row["cluster"]]
        test_vm["group"] += [row["group"]]
    test_vm = pd.DataFrame(test_vm)
    return test_vm


def get_laterality_idx(
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
    kernel: np.ndarray,
    fs_ds: int,
) -> pd.DataFrame:
    lat_index_df: Dict[str, list] = defaultdict(list)
    n_involved = th_involved["array_position"].unique()
    contra_idx = np.concatenate(pd.DataFrame(target_codes).iloc[1][contra].values)
    ipsi_idx = np.concatenate(pd.DataFrame(target_codes).iloc[1][ipsi].values)
    for i_neuron in n_involved:
        # contra
        trials_event_time = code_samples[
            contra_idx, np.where(code_numbers[contra_idx] == align_event)[1]
        ]  # moment when the target_on ocurrs in each trial
        sp_contra = SpikeData.indep_roll(
            sp_samples[contra_idx, i_neuron], -(trials_event_time).astype(int), axis=1
        )  # align trials with event onset
        sp_contra = sp_contra[
            np.nansum(sp_contra, axis=1) > 0
        ]  # Select trials with at least one spike
        # ipsi
        trials_event_time = code_samples[
            ipsi_idx, np.where(code_numbers[ipsi_idx] == align_event)[1]
        ]  # moment when the target_on ocurrs in each trial
        sp_ipsi = SpikeData.indep_roll(
            sp_samples[ipsi_idx, i_neuron], -(trials_event_time).astype(int), axis=1
        )  # align trials with event onset
        sp_ipsi = sp_ipsi[
            np.nansum(sp_ipsi, axis=1) > 0
        ]  # Select trials with at least one spike
        # Average fr of all trials
        mean_sp_contra = np.nanmean(sp_contra[:, :1100], axis=0)
        mean_sp_ipsi = np.nanmean(sp_ipsi[:, :1100], axis=0)
        # convolution
        conv_contra = np.convolve(mean_sp_contra, kernel, mode="same") * fs_ds
        conv_ipsi = np.convolve(mean_sp_ipsi, kernel, mode="same") * fs_ds
        min_value = np.concatenate([conv_contra, conv_ipsi]).min()
        conv_contra = np.nanmean(conv_contra - min_value)
        conv_ipsi = np.nanmean(conv_ipsi - min_value)
        p = stats.ttest_ind(mean_sp_contra, mean_sp_ipsi)[1]
        lat_index = (conv_contra - conv_ipsi) / (conv_contra + conv_ipsi)

        lat_index_df["array_position"] += [i_neuron]
        lat_index_df["lat_index"] += [lat_index]
        lat_index_df["p_lat"] += [p]

    lat_index_df = pd.DataFrame(lat_index_df)
    return lat_index_df


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
    fr_max_visual, fr_max_motor, fr_angle, fr_max_codes = [], [], [], []
    v_significant, m_significant = [], []
    for code in target_codes.keys():
        target_t_idx = target_codes[code][
            "trial_idx"
        ]  # select trials with the same stimulus
        trials_s_on = code_samples[
            target_t_idx, np.where(code_numbers[target_t_idx] == e_code_align)[1]
        ]
        shift_sp = SpikeData.indep_roll(
            sp_samples[target_t_idx, i_n], -(trials_s_on).astype(int), axis=1
        )  # align trials on event
        # select trials with at least one spike
        shift_sp = shift_sp[np.nansum(shift_sp[:, :1100], axis=1) > 5 * 1100 / 1000]
        mean_sp = np.nanmean(shift_sp, axis=0)  # mean of all trials
        if shift_sp.shape[0] == 0:
            conv = np.zeros((1100))
        else:
            conv = np.convolve(mean_sp, kernel, mode="same") * fs_ds
        fr_max_visual.append(np.nanmax(conv[:dur_v]))
        fr_angle.append(target_codes[code]["angle_codes"])
        fr_max_motor.append(np.nanmax(conv[700:1100]))
        fr_max_codes.append(np.nanmax(conv[:1100]))
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
        np.array(fr_max_visual),
        np.array(fr_max_motor),
        np.array(fr_angle),
        np.array(fr_max_codes),
        np.array(v_significant),
        np.array(m_significant),
    )
