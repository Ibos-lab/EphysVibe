import glob
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal, stats
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import select_trials
from ephysvibe.spike_sorting import config
from ephysvibe.task import def_task, task_constants
from collections import defaultdict
from typing import Dict
import logging
from scipy import fft, signal

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn import metrics
from multiprocessing import Pool

seed = 2023


def get_fr_df(filepath, in_out, cgroup, e_align):
    py_f = np.load(filepath, allow_pickle=True).item(0)
    sp = py_f["sp_data"]
    bhv = py_f["bhv"]
    trial_idx = select_trials.select_trials_block(sp, n_block=1)
    trial_idx = select_trials.select_correct_trials(bhv, trial_idx)
    task = def_task.create_task_frame(trial_idx, bhv, task_constants.SAMPLES_COND)
    neurons = np.where((sp["clustersgroup"] == cgroup))[0]
    fr_samples = firing_rate.fr_by_sample_neuron(
        sp=sp,
        neurons=neurons,
        task=task,
        in_out=in_out,
        kernel=0,
        e_align=e_align,
        plot=False,
    )
    return fr_samples
