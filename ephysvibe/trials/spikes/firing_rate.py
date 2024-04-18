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


def fr_in_window(x, start, end):
    fr = np.zeros(len(x))
    for i, i_x in enumerate(x):
        fr[i] = np.nan_to_num(np.mean(i_x[start[i] : end[i]]), nan=0)
    return fr


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
