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


# def select_events_timestamps(sp, trials_idx, events):
#     events_timestamps = []
#     for i_t in trials_idx:
#         e_timestamps = []
#         for _, event in events.items():
#             idx_event = np.where(sp["code_numbers"][i_t] == event)[0]
#             if len(idx_event) == 0:
#                 sample_event = [np.nan]
#             else:
#                 sample_event = sp["code_samples"][i_t][idx_event]
#             e_timestamps.append(sample_event)
#         events_timestamps.append(np.concatenate(e_timestamps))
#     return np.array(events_timestamps, dtype="object")


def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    d_shape = data.shape
    count = 0
    if len(d_shape) == 3:
        d_avg = np.zeros((d_shape[0], d_shape[1], int(np.floor(d_shape[2] / step))))
        for i_step in np.arange(0, d_shape[2] - win, step):
            d_avg[:, :, count] = np.mean(data[:, :, i_step : i_step + win], axis=2)
            count += 1
    if len(d_shape) == 2:
        d_avg = np.zeros((d_shape[0], int(np.floor(d_shape[1] / step))))
        for i_step in np.arange(0, d_shape[1] - win, step):
            d_avg[:, count] = np.mean(data[:, i_step : i_step + win], axis=1)
            count += 1
    if len(d_shape) == 1:
        d_avg = np.zeros((int(np.floor(d_shape[0] / step))))
        for i_step in np.arange(0, d_shape[0] - win, step):
            d_avg[count] = np.mean(data[i_step : i_step + win], axis=0)
            count += 1
    return d_avg


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


def convolve_signal(
    arr: np.ndarray,
    fs: int = 1000,
    w_size: float = 0.1,
    w_std: float = 0.015,
    axis: int = 1,
):
    # define kernel for convolution
    kernel = define_kernel(w_size, w_std, fs=fs)
    conv = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=axis, arr=arr
    )
    return conv * fs


def convolve_signal(
    arr: np.ndarray,
    fs: int = 1000,
    w_size: float = 0.1,
    w_std: float = 0.015,
    axis: int = 1,
):
    # define kernel for convolution
    kernel = define_kernel(w_size, w_std, fs=fs)
    conv = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=axis, arr=arr
    )
    return conv * fs


# def compute_conv_fr(neuron_trials, kernel, fs, downsample):
#     trials_conv = []
#     for i_trial in range(len(neuron_trials)):
#         if len(neuron_trials[i_trial]) != 0:
#             arr_timestamps = np.zeros(
#                 neuron_trials[i_trial][-1] + 1
#             )  # array with n timestamps
#             for sp in neuron_trials[i_trial]:
#                 arr_timestamps[sp] += 1
#             # Downsample to 1ms
#             arr_timestamps = np.sum(
#                 np.concatenate(
#                     (
#                         arr_timestamps,
#                         np.zeros(downsample - len(arr_timestamps) % downsample),
#                     )
#                 ).reshape(-1, downsample),
#                 axis=1,
#             )
#             conv = np.convolve(arr_timestamps, kernel, mode="same") * fs
#         else:
#             conv = [0]
#         trials_conv.append(conv)
#     return trials_conv


def fr_in_window(x, start, end):
    fr = np.zeros(len(x))
    for i, i_x in enumerate(x):
        fr[i] = np.nan_to_num(np.mean(i_x[start[i] : end[i]]), nan=0)
    return fr


# def sp_from_timestamp_to_binary(neuron_trials, downsample):
#     trials_sp = []
#     for i_trial in range(len(neuron_trials)):
#         if len(neuron_trials[i_trial]) != 0:
#             arr_timestamps = np.zeros(
#                 neuron_trials[i_trial][-1] + 1
#             )  # array with n timestamps
#             for sp in neuron_trials[i_trial]:
#                 arr_timestamps[sp] += 1
#             # Downsample to 1ms
#             arr_timestamps = np.sum(
#                 np.concatenate(
#                     (
#                         arr_timestamps,
#                         np.zeros(downsample - len(arr_timestamps) % downsample),
#                     )
#                 ).reshape(-1, downsample),
#                 axis=1,
#             )
#         else:
#             arr_timestamps = [0]
#         trials_sp.append(arr_timestamps)
#     return trials_sp


# def reshape_sp_list(
#     trials_sp: list, event_timestamps: np.ndarray, align_event: int
# ) -> Tuple[np.ndarray, int, np.ndarray]:
#     """Fill with zeros so all arrays have the same shape and align in align_event.

#     Args:
#         trials_sp (list): array of len n trials containing 1 if sp or 0 if no sp at each timestamp.
#         event_timestamps (np.ndarray): array containing the events timestamps for each trial.
#         align_event (int): event on which align the spikes.

#     Returns:
#         Tuple[np.ndarray, int, np.ndarray]:
#             - sp_shift (np.ndarray): array of shape (n trials, max trial duration) containig if spike 1, else 0.
#             - max_shift (int): max shift applied to the right.
#             - events_shift (np.ndarray): array of shape (n trials, n events) containig the shifted timestamps of events.
#     """
#     max_shift = np.max(event_timestamps[:, align_event])
#     max_duration = np.max(event_timestamps[:, -1])
#     sp_shift = []
#     events_shift = []
#     for i, i_tr in enumerate(trials_sp):
#         diff_before = max_shift - event_timestamps[i, align_event]
#         diff_after = max_duration - (diff_before + len(i_tr))
#         if diff_after < 0:  # (sp between trials)
#             sp_shift.append(np.concatenate((np.zeros(diff_before), i_tr[:diff_after])))
#         else:
#             sp_shift.append(
#                 np.concatenate((np.zeros(diff_before), i_tr, np.zeros(diff_after)))
#             )
#         events_shift.append(event_timestamps[i] + diff_before)
#     return np.array(sp_shift), max_shift, np.array(events_shift)


def plot_raster_fr(conv, shift_sp, time, ax, fig, t_before: int = 0):
    ax2 = ax.twinx()
    # fr
    ax.plot(time, conv)
    # raster
    conv_max = int(np.floor(max(conv)) + 2)
    num_trials = shift_sp.shape[0]
    # lineoffsets = np.arange(conv_max, num_trials + conv_max)
    rows, cols = np.where(shift_sp >= 1)
    ax2.scatter(
        cols - t_before,
        rows + num_trials,
        marker="|",
        alpha=1,
        edgecolors="none",
        color="k",
    )

    return fig, ax, ax2


# def plot_b1(
#     ax,
#     samples,
#     trials_conv_fr,
#     trials_time,
#     trials_sp,
#     events,
#     in_out,
#     x_lim_min,
#     x_lim_max,
#     trials_mask,
#     line_len=0.5,
# ):
#     num_trials = np.sum(trials_mask)
#     conv_max = int(np.floor(np.max(trials_conv_fr))) + 1
#     # fig, ax = plt.subplots(figsize=(10, 6), sharex=True, sharey=True)
#     ax2 = ax.twinx()
#     color, trials_sorted = [], []
#     for i_s, i_sample in enumerate(samples):
#         ax.plot(
#             trials_time,
#             trials_conv_fr[i_s],
#             color=task_constants.PALETTE_B1[i_sample],
#             label="Sample %s" % i_sample,
#         )
#         trials_sorted.append(trials_sp[trials_mask[i_s]])
#         color.append([task_constants.PALETTE_B1[i_sample]] * sum(trials_mask[i_s]))
#     ax2.eventplot(
#         np.concatenate(trials_sorted, axis=0),
#         color=np.concatenate(color, axis=0),
#         lineoffsets=np.arange(conv_max, (num_trials * line_len) + conv_max, line_len),
#         linewidths=0.8,
#         linelengths=line_len,
#     )
#     # events
#     for n_event, n_color in zip(range(2, 6), ["k", "k", "k", "k"]):
#         ax.vlines(
#             events[n_event],
#             0,
#             conv_max + (num_trials * line_len),
#             color=n_color,
#             linestyles="dashed",
#         )
#     ax.set(xlabel="Time (s)", ylabel="Average firing rate")
#     ax2.set(xlabel="Time (s)", ylabel="trials")

#     condition = {-1: "out", 1: "in"}
#     ax.set_title(condition[in_out])
#     ax.set_xlim(x_lim_min, x_lim_max)
#     ax.set_ylim(0)
#     ax2.set_ylim(0)
#     ax2.set_yticks(np.arange(conv_max, (num_trials * line_len) + conv_max, line_len))
#     plt.setp(ax2.get_yticklabels(), visible=False)
