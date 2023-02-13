import os
import logging
from pathlib import Path
import numpy as np
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple
from ...spike_sorting import config
from ...task import task_constants


def select_events_timestamps(sp, trials_idx, events):
    events_timestamps = []
    for i_t in trials_idx:
        e_timestamps = []
        for _, event in events.items():
            idx_event = np.where(sp["code_numbers"][i_t] == event)[0]
            if len(idx_event) == 0:
                sample_event = [np.nan]
            else:
                sample_event = sp["code_samples"][i_t][idx_event]
            e_timestamps.append(sample_event)
        events_timestamps.append(np.concatenate(e_timestamps))
    return np.array(events_timestamps, dtype="object")


def align_neuron_spikes(trials_idx, sp, neuron, event_timestamps):
    # create list of neurons containing the spikes timestamps aligned with the event
    neuron_trials = []
    for i, i_t in enumerate(trials_idx):
        neuron_trials.append(sp["sp_samples"][i_t][neuron] - event_timestamps[i])
    return np.array(neuron_trials, dtype="object")


def trial_average_fr(neuron_trials):
    # Compute the Average firing rate
    sorted_sp_neuron = np.sort(np.concatenate(neuron_trials))
    sum_sp = np.zeros(int(sorted_sp_neuron[-1] - sorted_sp_neuron[0] + 1))
    sorted_sp_shift = np.array(sorted_sp_neuron - sorted_sp_neuron[0], dtype=int)
    for i in sorted_sp_shift:
        sum_sp[i] += 1
    trial_average_sp = sum_sp / len(neuron_trials)
    return trial_average_sp, sorted_sp_neuron


def define_kernel(w_size, w_std, fs):
    kernel = signal.gaussian(M=w_size * fs, std=w_std * fs)
    kernel = kernel / sum(kernel)  # area of the kernel must be one
    return kernel


def compute_conv_fr(neuron_trials, kernel, fs, downsample):
    trials_conv = []
    for i_trial in range(len(neuron_trials)):
        if len(neuron_trials[i_trial]) != 0:
            arr_timestamps = np.zeros(
                neuron_trials[i_trial][-1] + 1
            )  # array with n timestamps
            for sp in neuron_trials[i_trial]:
                arr_timestamps[sp] += 1
            # Downsample to 1ms
            arr_timestamps = np.sum(
                np.concatenate(
                    (
                        arr_timestamps,
                        np.zeros(downsample - len(arr_timestamps) % downsample),
                    )
                ).reshape(-1, downsample),
                axis=1,
            )
            conv = np.convolve(arr_timestamps, kernel, mode="same") * fs
        else:
            conv = [0]
        trials_conv.append(conv)
    return trials_conv


def fr_in_window(x, start, end):
    fr = np.zeros(len(x))
    for i, i_x in enumerate(x):
        fr[i] = np.nan_to_num(np.mean(i_x[start[i] : end[i]]), nan=0)
    return fr


def aligne_conv_fr(conv, event_timestamps, align_event):
    max_shift = np.max(event_timestamps[:, align_event])
    max_duration = np.max(event_timestamps[:, -1])
    conv_shift = []
    events_shift = []
    for i, i_conv in enumerate(conv):
        diff_before = max_shift - event_timestamps[i, align_event]
        diff_after = max_duration - (diff_before + len(i_conv))
        if diff_after < 0:  # (sp between trials)
            conv_shift.append(
                np.concatenate((np.zeros(diff_before), i_conv[:diff_after]))
            )
        else:
            conv_shift.append(
                np.concatenate((np.zeros(diff_before), i_conv, np.zeros(diff_after)))
            )
        events_shift.append(event_timestamps[i] + diff_before)
    return np.array(conv_shift), max_shift, events_shift


def sp_from_timestamp_to_binary(neuron_trials, downsample):
    trials_sp = []
    for i_trial in range(len(neuron_trials)):
        if len(neuron_trials[i_trial]) != 0:
            arr_timestamps = np.zeros(
                neuron_trials[i_trial][-1] + 1
            )  # array with n timestamps
            for sp in neuron_trials[i_trial]:
                arr_timestamps[sp] += 1
            # Downsample to 1ms
            arr_timestamps = np.sum(
                np.concatenate(
                    (
                        arr_timestamps,
                        np.zeros(downsample - len(arr_timestamps) % downsample),
                    )
                ).reshape(-1, downsample),
                axis=1,
            )
        else:
            arr_timestamps = [0]
        trials_sp.append(arr_timestamps)
    return trials_sp


def reshape_sp_list(
    trials_sp: list, event_timestamps: np.ndarray, align_event: int
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Fill with zeros so all arrays have the same shape and align in align_event.

    Args:
        trials_sp (list): array of len n trials containing 1 if sp or 0 if no sp at each timestamp.
        event_timestamps (np.ndarray): array containing the events timestamps for each trial.
        align_event (int): event on which align the spikes.

    Returns:
        Tuple[np.ndarray, int, np.ndarray]:
            - sp_shift (np.ndarray): array of shape (n trials, max trial duration) containig if spike 1, else 0.
            - max_shift (int): max shift applied to the right.
            - events_shift (np.ndarray): array of shape (n trials, n events) containig the shifted timestamps of events.
    """
    max_shift = np.max(event_timestamps[:, align_event])
    max_duration = np.max(event_timestamps[:, -1])
    sp_shift = []
    events_shift = []
    for i, i_tr in enumerate(trials_sp):
        diff_before = max_shift - event_timestamps[i, align_event]
        diff_after = max_duration - (diff_before + len(i_tr))
        if diff_after < 0:  # (sp between trials)
            sp_shift.append(np.concatenate((np.zeros(diff_before), i_tr[:diff_after])))
        else:
            sp_shift.append(
                np.concatenate((np.zeros(diff_before), i_tr, np.zeros(diff_after)))
            )
        events_shift.append(event_timestamps[i] + diff_before)
    return np.array(sp_shift), max_shift, np.array(events_shift)


def plot_raster_fr(
    all_trials_fr,
    max_shift,
    fs,
    neuron_trials,
    code,
    ax,
    fig,
    i,
    x_lim_max,
    x_lim_min,
    events,
):
    ax2 = ax.twinx()
    # fr
    ax.plot((np.arange(len(all_trials_fr)) - max_shift) / fs, all_trials_fr)
    # raster
    conv_max = int(np.floor(max(all_trials_fr)) + 2)
    num_trials = len(neuron_trials)
    lineoffsets = np.arange(conv_max, num_trials + conv_max)
    ax2.eventplot(neuron_trials / fs, color=".2", lineoffsets=1, linewidths=0.8)
    # events
    ax.vlines(
        events[1] / fs, 0, lineoffsets[-1], color="b", linestyles="dashed"
    )  # target_on
    ax.vlines(
        events[2] / fs, 0, lineoffsets[-1], color="k", linestyles="dashed"
    )  # target_off
    ax.vlines(
        events[3] / fs, 0, lineoffsets[-1], color="k", linestyles="dashed"
    )  # fix_spot_off
    ax.vlines(
        events[4] / fs, 0, lineoffsets[-1], color="k", linestyles="dashed"
    )  # response
    # figure setings
    ax.set(xlabel="Time (s)", ylabel="Average firing rate")
    ax2.set(xlabel="Time (s)", ylabel="trials")
    ax2.set_yticks(range(-conv_max, num_trials))
    ax.set_title("Code %s" % (code), fontsize=8)
    ax.set_xlim(x_lim_min, x_lim_max)
    plt.setp(ax2.get_yticklabels(), visible=False)
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    fig.suptitle("Neuron %d" % (i + 1), x=0)
    return fig


def plot_b1(
    ax,
    samples,
    trials_conv_fr,
    trials_time,
    trials_sp,
    events,
    in_out,
    x_lim_min,
    x_lim_max,
    trials_mask,
    line_len=0.5,
):
    num_trials = np.sum(trials_mask)
    conv_max = int(np.floor(np.max(trials_conv_fr))) + 1
    # fig, ax = plt.subplots(figsize=(10, 6), sharex=True, sharey=True)
    ax2 = ax.twinx()
    color, trials_sorted = [], []
    for i_s, i_sample in enumerate(samples):
        ax.plot(
            trials_time,
            trials_conv_fr[i_s],
            color=task_constants.PALETTE_B1[i_sample],
            label="Sample %s" % i_sample,
        )
        trials_sorted.append(trials_sp[trials_mask[i_s]])
        color.append([task_constants.PALETTE_B1[i_sample]] * sum(trials_mask[i_s]))
    ax2.eventplot(
        np.concatenate(trials_sorted, axis=0),
        color=np.concatenate(color, axis=0),
        lineoffsets=np.arange(conv_max, (num_trials * line_len) + conv_max, line_len),
        linewidths=0.8,
        linelengths=line_len,
    )
    # events
    for n_event, n_color in zip(range(2, 6), ["k", "k", "k", "k"]):
        ax.vlines(
            events[n_event],
            0,
            conv_max + (num_trials * line_len),
            color=n_color,
            linestyles="dashed",
        )
    ax.set(xlabel="Time (s)", ylabel="Average firing rate")
    ax2.set(xlabel="Time (s)", ylabel="trials")

    condition = {-1: "out", 1: "in"}
    ax.set_title(condition[in_out])
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(0)
    ax2.set_ylim(0)
    ax2.set_yticks(np.arange(conv_max, (num_trials * line_len) + conv_max, line_len))
    plt.setp(ax2.get_yticklabels(), visible=False)
    # fig.legend(fontsize=9)
    # fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    # fig.text(
    #     0.10,
    #     0.01,
    #     s="Aligned with %s" % list(task_constants.EVENTS_B1.keys())[e_align],
    #     horizontalalignment="center",
    #     verticalalignment="center",
    # )
    # fig.suptitle("Neuron (%s) %d" % (cgroup, i_neuron + 1), x=0.10)
    # return fig


def fr_by_sample_neuron(
    sp: Dict,
    neurons: np.ndarray,
    task: pd.DataFrame,
    in_out: int,
    e_align: int,
    plot: bool = True,
    kernel: np.ndarray = None,
    output_dir: Path = None,
    filename: str = "",
    x_lim_max: float = 1.5,
    x_lim_min: float = -0.7,
    fs: int = config.FS,
    downsample: int = config.DOWNSAMPLE,
) -> pd.DataFrame:
    """Compute and plot the fr for each neuron, trial and sample.

    Args:
        sp (Dict): dictionary with the spike sorting data.
        neurons (np.ndarray): number of the neurons.
        task (pd.DataFrame): info of the task for each trial.
        in_out (int): 1 for trials with stimuli in, -1 out.
        e_align (int): event to which align the spikes.
        plot (bool, optional): whether to plot. Defaults to True.
        kernel (np.ndarray, optional): kernel to compute the mean fr. Defaults to None.
        output_dir (Path, optional): output directory. Defaults to None.
        filename (str, optional): Name for the output file. Defaults to "".
        x_lim_max (float, optional): x max limit for the plot. Defaults to 1.5.
        x_lim_min (float, optional): x min limit for the plot. Defaults to -0.7.
        fs (int, optional): sampling frequency. Defaults to config.FS.
        downsample (int, optional): int to downsample the frequecy. Defaults to config.DOWNSAMPLE.

    Returns:
        pd.DataFrame: contains the fr at each timestamp and the events timestamps.
    """
    sample_id = task[(task["in_out"] == in_out)]["sample_id"].values
    samples = np.sort(np.unique(sample_id))
    target_trials_idx = task[(task["in_out"] == in_out)]["idx_trial"].values
    fr_samples: Dict[str, list] = defaultdict(list)
    for i_neuron, neuron in enumerate(neurons):
        ev_ts = select_events_timestamps(
            sp, target_trials_idx, task_constants.EVENTS_B1
        )  # select events timestamps for all trials
        neuron_trials = align_neuron_spikes(
            target_trials_idx, sp, neuron, ev_ts[:, 0]
        )  # align sp with start trial
        shift_ev_ts = np.round(
            (((ev_ts.T - ev_ts.T[0]).T) / downsample).astype(float)
        ).astype(
            int
        )  # aling events with start trial
        trials_sp = sp_from_timestamp_to_binary(neuron_trials, downsample)
        sp_shift, max_shift, events_shift = reshape_sp_list(
            trials_sp, event_timestamps=shift_ev_ts, align_event=e_align
        )  # add zeros so each trial array has the same shape
        if plot == True:
            conv = compute_conv_fr(neuron_trials, kernel, (fs / downsample), downsample)
            all_trials_fr, _, _ = aligne_conv_fr(
                conv=conv, event_timestamps=shift_ev_ts, align_event=e_align
            )
        trials_conv_fr, all_mask = [], []
        # Iterate by sample and save results in df
        for i_sample in samples:
            mask_sample = sample_id == i_sample
            all_mask.append(mask_sample)
            sp_sample = sp_shift[mask_sample, :]
            for t in np.arange(sp_sample.shape[1]):
                fr_samples["t" + str(t)] += sp_sample[:, t].tolist()
            fr_samples["neuron"] += [i_neuron + 1] * len(sp_sample)
            fr_samples["sample"] += [i_sample] * len(sp_sample)
            fr_samples["trial_idx"] += target_trials_idx[mask_sample].tolist()
            for n_ev, n_event in enumerate(list(task_constants.EVENTS_B1.keys())):
                fr_samples[n_event] += events_shift[mask_sample][:, n_ev].tolist()
            if plot == True:
                trials_conv_fr.append(np.mean(all_trials_fr[mask_sample, :], axis=0))
        # plot
        if plot == True:
            trials_conv_fr = np.array(trials_conv_fr)
            trials_time = (np.arange(len(trials_conv_fr[0])) - max_shift) / (
                fs / downsample
            )
            events_shift = (np.mean(events_shift, axis=0) - max_shift) / (
                fs / downsample
            )
            num_trials = len(neuron_trials)
            neuron_trials_shift = (
                align_neuron_spikes(target_trials_idx, sp, neuron, ev_ts[:, 2])
                / config.FS
            )  # align sp with stim onset
            fig = plot_b1(
                samples,
                trials_conv_fr,
                trials_time,
                neuron_trials_shift,
                events_shift,
                num_trials,
                in_out,
                x_lim_min,
                x_lim_max,
                i_neuron,
                all_mask,
                e_align=e_align,
            )
            if output_dir:
                logging.info("Saving figure, neuron: %d" % (i_neuron + 1))
                fig.savefig(
                    "/".join(
                        [os.path.normpath(output_dir)]
                        + [filename + "_n" + str(i_neuron + 1) + "_b1.jpg"]
                    )
                )
    fr_samples = pd.DataFrame(fr_samples)
    return fr_samples
