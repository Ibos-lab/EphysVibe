# Tools for pre-processing OpenEphis data
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from open_ephys.analysis import Session
import numpy as np
import pandas as pd
import re
from . import config
from ..structures.bhv_data import BhvData
import mne


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


def load_event_files(event_path: Path) -> Dict:
    """Load files in the event folder from OE.

    Args:
        event_path (Path): path to the events folder

    Returns:
        Dict[np.ndarray, np.ndarray, np.ndarray]:
            - samples (np.ndarray): n sample at wich each event acurres
            - channel (np.ndarray): channel that change state at each sample
            - state (np.ndarray): state of the channel:
                - 1 switched to on
                - 0 switched to off
    """
    samples = np.load(glob.glob("/".join([event_path] + ["sample_numbers.npy"]))[0])
    channel = np.load(glob.glob("/".join([event_path] + ["states.npy"]))[0])
    state = np.where(channel > 0, 1, 0)
    channel = abs(channel)
    events = {"samples": samples, "channel": channel, "state": state}
    return events


def load_eyes(
    continuous_path: List,
    shape_0: int,
    shape_1: int,
    start_ch: int,
    n_eyes: int,
    idx_start_time: int = 0,
) -> np.ndarray:
    """Load eyes .dat file.

    Args:
        continuous_path (List): list containing the splited continuous path
        shape_0 (int): number of rows
        shape_1 (int): number of columns
        idx_start_time (int, optional): sample where to start taking the signal. Defaults to 0.

    Returns:
        np.ndarray: array containing the downsampled eyes values
    """
    # load eyes data
    cont = np.memmap(
        continuous_path,
        mode="r",
        dtype="int16",
        shape=(shape_1, shape_0),
        order="F",
    )
    # downsample signal
    eyes_ds = np.zeros(
        (
            n_eyes,
            int(np.floor((cont.shape[1] - idx_start_time) / config.DOWNSAMPLE) + 1),
        )
    )
    for i, i_data in enumerate(range(start_ch, start_ch + n_eyes)):
        logging.info("Downsampling eyes")
        dat = np.array(np.asarray(cont[i_data, idx_start_time:]), order="C")
        eyes_ds[i] = signal_downsample(
            dat,
            config.DOWNSAMPLE,
            idx_start=0,
            axis=0,
        )
        del dat
    del cont
    return eyes_ds


def load_spike_data(spike_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load spikes data.

    Args:
        spike_path (str): path to the kilosort folder

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            - spike_times_idx (np.ndarray): array containing the spike times
            - spike_clusters (np.ndarray): array containing to which neuron the spike belongs to
            - cluster_info (pd.Dataframe): info about the clusters
    """
    # search kilosort folder
    logging.info("Loading spikes data")
    spike_times_idx = (
        np.load(spike_path + "/spike_times.npy", "r").reshape(-1) - 1
    )  # -1 to have the first idx = 0 and not 1
    spike_clusters = np.load(spike_path + "/spike_clusters.npy", "r")  #
    cluster_info = pd.read_csv(
        spike_path + "/cluster_info.tsv", sep="\t"
    )  # info of each cluster

    return spike_times_idx, spike_clusters, cluster_info


def check_clusters(
    spike_times_idx: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_info: pd.DataFrame,
    len_samples: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Check whether spikes are detected during the recording and if there are good or mua.

    Args:
        spike_times_idx (np.ndarray): array containing the spike times
        spike_clusters (np.ndarray): array containing to which neuron the spike belongs to
        cluster_info (pd.DataFrame): info about the clusters
        len_samples (int):

    Raises:
        IndexError: spikes of mua or good units are detected after the end of the recording
        ValueError: there are not good or mua units

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            - spike_times_idx (np.ndarray): array containing the spike times during the recording
            - spike_clusters (np.ndarray): array containing to which neuron the spike belongs to during the recording
            - cluster_info (pd.Dataframe): info about good and mua clusters
    """
    # check if all the spikes are detected inside the time of recording
    idx_sp_out = np.where(spike_times_idx >= len_samples)[0]
    if len(idx_sp_out) > 0:
        sp_out_id = spike_clusters[idx_sp_out]
        cluster_info[cluster_info["cluster_id"] == sp_out_id[0]]["group"].values
        clusters_out = cluster_info[cluster_info["cluster_id"].isin(sp_out_id)][
            "group"
        ].values
        if ~np.all(clusters_out == "noise"):
            raise IndexError
        else:
            spike_times_idx = spike_times_idx[: min(idx_sp_out)]
            spike_clusters = spike_clusters[: min(idx_sp_out)]
    nan_values = (
        cluster_info[["cluster_id", "ch", "depth", "fr", "group", "n_spikes"]]
        .isnull()
        .sum()
        .sum()
    )
    if nan_values > 0:  # check if nan values in relevant columns
        logging.warning("/cluster_info.tsv has %d nan values" % nan_values)
        cluster_info = cluster_info.dropna(
            axis=0, subset=["cluster_id", "ch", "depth", "fr", "group", "n_spikes"]
        )
        logging.warning("Rows with nan values deleted")
    cluster_info = cluster_info[cluster_info["group"] != "noise"]  # ignore noisy groups
    if cluster_info.shape[0] == 0:
        raise ValueError

    return spike_times_idx, spike_clusters, cluster_info


def signal_downsample(
    x: np.ndarray, downsample: int, idx_start: int = 0, axis: int = 1
) -> np.ndarray:
    """Downsample signal.

    Args:
        x (np.ndarray): signal to downsample
        downsample (int): amount to downsample
        idx_start (int, optional): sample where to start taking the signal. Defaults to 0.
        axis (int, optional): axis where to apply the downsample. Defaults to 1.

    Returns:
        np.ndarray: downsample signal.
    """
    idx_ds = np.arange(idx_start, x.shape[axis], downsample)
    if axis == 1:
        return x[:, idx_ds]
    x = x[idx_ds]
    return x


def select_samples(
    c_samples, e_samples, fs, t_before_event=10, downsample=30
) -> Tuple[np.ndarray, np.ndarray]:
    # Select the samples of continuous data from t sec before the first event occurs
    # This is done to reduce the data
    idx_start_time = np.where(c_samples == e_samples[0])[0]
    idx_start_time = (
        idx_start_time[0] if idx_start_time.shape[0] > 0 else 0
    )  # check if not empty, else we select all data
    idx_start_time = (
        idx_start_time - fs * t_before_event
        if idx_start_time - fs * t_before_event > 0
        else 0
    )  # check if idx_start_time - fs*t >0, else we select all data
    # select samples from idx_start_time and donwsample
    ds_samples = signal_downsample(
        c_samples, downsample, idx_start=idx_start_time, axis=0
    )
    return ds_samples, idx_start_time


def reconstruct_8bits_words(
    real_strobes: np.ndarray, e_channel: np.ndarray, e_state: np.ndarray
) -> np.ndarray:
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


def select_trials_bhv(bhv: BhvData, n_trials: int) -> BhvData:
    bhv_trials = bhv.block.shape[0]
    new_bhv = vars(bhv).copy()
    for key, val in new_bhv.items():
        if val.shape[0] == bhv_trials:
            new_bhv[key] = val[:n_trials]

    new_bhv = BhvData(**new_bhv)
    return new_bhv


def check_strobes(
    code_numbers: np.ndarray, full_word: np.ndarray, real_strobes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    # Check if strobe and codes number match
    len_idx = None
    bhv_codes = code_numbers.reshape(-1)
    bhv_codes = bhv_codes[~np.isnan(bhv_codes)]
    if full_word.shape[0] != real_strobes.shape[0]:
        logging.warning("Strobe and codes number shapes do not match")
        logging.info("Strobes =", real_strobes.shape[0])
        logging.info("codes number =", full_word.shape[0])
    else:
        logging.info("Strobe and codes number shapes do match")
        logging.info("Strobes = %d", real_strobes.shape[0])
        logging.info("codes number = %d", full_word.shape[0])
    if full_word.shape[0] != bhv_codes.shape[0]:
        logging.warning("ML and OE shapes do not match")
        logging.info("ML = %d", bhv_codes.shape[0])
        logging.info("OE = %d", full_word.shape[0])
        if full_word.shape[0] > bhv_codes.shape[0]:
            logging.error(
                "OE has %d more codes than ML",
                (full_word.shape[0] - bhv_codes.shape[0]),
            )
            raise IndexError
        elif np.sum(full_word - bhv_codes[: full_word.shape[0]]) != 0:
            logging.error("Strobe and codes number do not match")
            raise IndexError
        else:  # np.sum(full_word-bhv_codes[:full_word.shape[0]]) == 0

            # find the last 18 in bhv_codes (last complete trial)
            logging.info(
                "ML has %d more codes than OE but the existing ones match",
                (bhv_codes.shape[0] - full_word.shape[0]),
            )
            bhv_codes = bhv_codes[: full_word.shape[0]]
            idx = np.where(bhv_codes == 18)[0]
            # bhv = select_trials_bhv(bhv, len(idx))
            len_idx = len(idx)
            full_word = full_word[: idx[-1] + 1]
            real_strobes = real_strobes[: idx[-1] + 1]
    else:
        logging.info("ML and OE code numbers do match")
        if np.sum(bhv_codes - full_word) != 0:
            logging.info("Warning, ML and OE codes are different")
        else:
            logging.info("ML and OE codes are the same")

    return full_word, real_strobes, len_idx


def find_events_codes(
    events: Dict, code_numbers: BhvData
) -> Tuple[np.ndarray, np.ndarray, int]:
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
    full_word, idx_real_strobes, len_idx = check_strobes(
        code_numbers, full_word, idx_real_strobes
    )
    real_strobes = events["samples"][idx_real_strobes]

    idx_start = np.where(full_word == config.START_CODE)[0]
    idx_end = np.where(full_word == config.END_CODE)[0]

    return (
        full_word,
        real_strobes,
        len_idx,
        idx_start,
        idx_end,
    )


def compute_lfp(
    continuous_path: Path,
    shape_0: int,
    shape_1: int,
    start_time: int = 0,
    start_ch: int = 0,
    n_ch: int = 0,
    f_lp: int = None,
    f_hp: int = None,
    filt: bool = True,
) -> np.ndarray:
    """Filter and downsample lfp.

    Args:
        continuous_path (Path): path to the continuous file.
        shape_0 (int): number of timestamps.
        shape_1 (int): number of channels.
        start_time (int, optional): starting timestamp. Defaults to 0.
        start_ch (int, optional): first channel. Defaults to 0.
        n_ch (int, optional): number of channels. Defaults to 0.
        f_lp (int, optional): low pass frequency. Defaults to None.
        f_hp (int, optional): high pass frequency. Defaults to None.
        filt (bool, optional): whether to filter lfps. Defaults to True.

    Returns:
        np.ndarray: preprocessed signal.
    """
    logging.info("Computing LFPs")

    cont = np.memmap(
        continuous_path,
        mode="r",
        dtype="int16",
        shape=(shape_1, shape_0),
        order="F",
    )
    lfp_ds = np.zeros(
        (n_ch, int(np.floor((cont.shape[1] - start_time) / config.DOWNSAMPLE) + 1))
    )
    for i, i_data in enumerate(range(start_ch, start_ch + n_ch)):
        dat = np.asarray(cont[i_data, start_time:])
        if filt:
            dat = mne.filter.filter_data(
                dat.astype(float),
                sfreq=config.FS,
                l_freq=f_lp,
                h_freq=f_hp,
                method="fir",
                verbose=False,
            )
        lfp_ds[i] = signal_downsample(dat, config.DOWNSAMPLE, idx_start=0, axis=0)
        del dat
    del cont
    return lfp_ds
