# Tools for pre-processing OpenEphis data
import os
import glob
import h5py
from h5py import Group
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from open_ephys.analysis import Session
import numpy as np
import pandas as pd
import re
from scipy.signal import butter, sosfilt
from . import data_structure, config


def load_oe_data(directory: Path) -> Tuple[Session, str, str, list]:
    """Load OpenEphis data.
    Args:
        directory (Path): path to the directory
    Returns:
        Tuple[Session, str, str, list]:
            - session (Session): session object
            - subject (str): name of the subject
            - date_time (str): date and time of the recording
            - areas (List): name of the recorded areas
    """
    logging.info("Loading OE data")
    # directory = bhv_path.parents[3]
    session = Session(directory)
    split_dir = os.path.normpath(directory).split(os.sep)
    date_time = split_dir[-1]
    subject = split_dir[-2]
    # check nodes
    areas = []
    for node in session.recordnodes:
        node_path = node.directory
        n_area = re.split(r"[ ]", node_path.split(os.sep)[-1])[-1]
        areas.append(n_area)

    return session, subject, date_time, areas


def load_dat_file(dat_path: Path, shape_0: int, shape_1: int) -> np.memmap:
    """load .dat file.

    Args:
        dat_path (Path): path to the .dat file
        shape_0 (int): number of rows
        shape_1 (int): number of columns

    Returns:
        np.memmap: .dat file
    """
    dat_file = np.memmap(dat_path, mode="r", dtype="int16", shape=(shape_0, shape_1)).T
    return dat_file


def load_event_files(event_path: Path) -> Dict:
    """Load files in the event folder from OE.

    Args:
        event_path (Path): path to the events folder

    Returns:
        Dict[np.array, np.array, np.array]:
            - samples (np.array): n sample at wich each event acurres
            - channel (np.array): channel that change of state at each sample
            - state (np.array): state of the channel:
                - 1 switched to on
                - 0 switched to off
    """
    samples = np.load(glob.glob("/".join([event_path] + ["sample_numbers.npy"]))[0])
    channel = np.load(glob.glob("/".join([event_path] + ["states.npy"]))[0])
    state = np.where(channel > 0, 1, 0)
    channel = abs(channel)
    events = {"samples": samples, "channel": channel, "state": state}
    return events


def load_bhv_data(directory: Path, subject: str) -> Group:
    """Load behavioral data.

    Args:
        directory (Path): path to the directory
        subject (str): name of the subject

    Returns:
        Group: bhv file
    """
    # Load behavioral data
    bhv_path = os.path.normpath(str(directory) + "/*" + subject + ".h5")
    bhv_path = glob.glob(bhv_path, recursive=True)
    logging.info(directory)
    if len(bhv_path) == 0:
        logging.info("Bhv file not found")
    logging.info("Loading bhv data")
    bhv = h5py.File(bhv_path[0], "r")["ML"]

    return bhv


def load_eyes(
    s_path: List, shape_0: int, shape_1: int, start_time: int = 0
) -> np.array:
    """Load eyes .dat file.

    Args:
        s_path (List): list containing the splited continuous path
        shape_0 (int): number of rows
        shape_1 (int): number of columns
        start_time (int, optional): sample where to start taking the signal. Defaults to 0.

    Returns:
        np.array: array containing the downsampled eyes values
    """
    # load eyes data
    eyes_path = "/".join(s_path[:-1] + ["Record Node eyes"] + ["eyes.dat"])
    continuous = load_dat_file(eyes_path, shape_0=shape_0, shape_1=shape_1)
    # downsample signal
    eyes_ds = signal_downsample(
        continuous[:, start_time:], config.DOWNSAMPLE, idx_start=0, axis=1
    )

    return eyes_ds


def load_spike_data(spike_path: str) -> Tuple[np.array, np.memmap, pd.DataFrame]:
    """Load spikes data.

    Args:
        spike_path (str): path to the kilosort folder

    Returns:
        Tuple[np.array, np.memmap, pd.DataFrame]:
            - idx_spiketimes (np.array): array containing the spike times
            - spiketimes_clusters_id (memmap): array containing to which neuron the spike times belongs to
            - cluster_info (pd.Dataframe): info about the clusters
    """
    # search kilosort folder
    logging.info("Loading spikes data")
    idx_sp_ksamples = np.load(spike_path + "/spike_times.npy", "r").reshape(-1) - 1
    sp_ksamples_clusters_id = np.load(spike_path + "/spike_clusters.npy", "r")  #
    cluster_info = pd.read_csv(
        spike_path + "/cluster_info.tsv", sep="\t"
    )  # info of each cluster
    # ignore noisy groups
    cluster_info = cluster_info[cluster_info["group"] != "noise"]
    return idx_sp_ksamples, sp_ksamples_clusters_id, cluster_info


def signal_downsample(
    x: np.array, downsample: int, idx_start: int = 0, axis: int = 1
) -> np.array:
    """Downsample signal.

    Args:
        x (np.array): signal to downsample
        downsample (int): amount to downsample
        idx_start (int, optional): sample where to start taking the signal. Defaults to 0.
        axis (int, optional): axis where to apply the downsample. Defaults to 1.

    Returns:
        np.array: downsample signal.
    """
    idx_ds = np.arange(idx_start, x.shape[axis], downsample)
    if axis == 1:
        return x[:, idx_ds]
    return x[idx_ds]


def select_samples(c_samples, e_samples, fs, t_before_event=10, downsample=30):
    # Select the samples of continuous data from t sec before the first event occurs
    # This is done to reduce the data
    start_time = np.where(c_samples == e_samples[0])[0]
    start_time = (
        start_time[0] if start_time.shape[0] > 0 else 0
    )  # check if not empty, else we select all data
    start_time = (
        start_time - fs * t_before_event if start_time - fs * t_before_event > 0 else 0
    )  # check if start_time - fs*t >0, else we select all data
    # select samples from start_time and donwsample
    ds_samples = signal_downsample(c_samples, downsample, idx_start=start_time, axis=0)

    return ds_samples, start_time


def reconstruct_8bits_words(real_strobes, e_channel, e_state):
    idx_old = 0
    current_8code = np.zeros(8, dtype=np.int64)
    full_word = np.zeros(len(real_strobes))

    for n_strobe, idx_strobe in enumerate(real_strobes):

        for ch in np.arange(0, 7):

            idx_ch = np.where(e_channel[idx_old:idx_strobe] == ch + 1)[0]

            current_8code[7 - ch] = (
                e_state[idx_ch[-1]] if idx_ch.size != 0 else current_8code[7 - ch]
            )

        full_word[n_strobe] = int("".join([str(item) for item in current_8code]), 2)

    return full_word


def check_strobes(bhv, full_word, real_strobes):
    # Check if strobe and codes number match
    bhv_codes = []
    trials = list(bhv.keys())[1:-1]
    for i_trial in trials:
        bhv_codes.append(list(bhv[i_trial]["BehavioralCodes"]["CodeNumbers"])[0])
    bhv_codes = np.concatenate(bhv_codes)

    if full_word.shape[0] != real_strobes.shape[0]:
        logging.info("Warning, Strobe and codes number do not match")
        logging.info("Strobes =", real_strobes.shape[0])
        logging.info("codes number =", full_word.shape[0])
    else:
        logging.info("Strobe and codes number do match")
        logging.info("Strobes = %d", real_strobes.shape[0])
        logging.info("codes number = %d", full_word.shape[0])

    if full_word.shape[0] != bhv_codes.shape[0]:
        logging.info("Warning, ML and OE code numbers do not match")
    else:
        logging.info("ML and OE code numbers do match")
        if np.sum(bhv_codes - full_word) != 0:
            logging.info("Warning, ML and OE codes are different")
        else:
            logging.info("ML and OE codes are the same")


def find_events_codes(events, bhv):

    # Reconstruct 8 bit words
    logging.info("Reconstructing 8 bit words")
    idx_real_strobes = np.where(
        np.logical_and(
            np.logical_and(events["channel"] == 8, events["state"] == 1),
            events["samples"] > 0,
        )
    )[
        0
    ]  # state 1: ON, state 0: OFF
    full_word = reconstruct_8bits_words(
        idx_real_strobes, e_channel=events["channel"], e_state=events["state"]
    )
    # Check if strobe and codes number match
    check_strobes(bhv, full_word, idx_real_strobes)

    real_strobes = events["samples"][idx_real_strobes]
    start_trials = real_strobes[
        full_word == config.START_CODE
    ]  # samples where trials starts

    # search number of blocks
    trial_keys = list(bhv.keys())[1:-1]
    n_trials = len(trial_keys)
    blocks = []
    # iterate over trials
    for trial_i in range(n_trials):
        blocks.append(bhv[trial_keys[trial_i]]["Block"][0][0])

    # change bhv structure
    bhv = np.array(data_structure.bhv_to_dictionary(bhv))

    return (full_word, real_strobes, start_trials, blocks, bhv)


def compute_lfp(c_values: np.array) -> np.array:
    """Compute lfp and downsample.

    Args:
        c_values (np.array): signal from which compute lfps

    Returns:
        np.array: lfps
    """
    # define lowpass and high pass butterworth filter
    hp_sos = butter(config.HP_ORDER, config.HP_FC, "hp", fs=config.FS, output="sos")
    lp_sos = butter(config.LP_ORDER, config.LP_FC, "lp", fs=config.FS, output="sos")
    lfp_ds = np.zeros(
        (c_values.shape[0], int(np.floor(c_values.shape[1] / config.DOWNSAMPLE)) + 1)
    )
    for i_data in range(c_values.shape[0]):
        data_f = sosfilt(hp_sos, c_values[i_data])
        data_f = sosfilt(lp_sos, data_f)
        lfp_ds[i_data] = signal_downsample(
            data_f, config.DOWNSAMPLE, idx_start=0, axis=0
        )
    return lfp_ds
