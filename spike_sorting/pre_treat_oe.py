import numpy as np
import pandas as pd
import h5py
import pandas as pd
import logging

from spike_sorting import config
from spike_sorting import utils_oe
from spike_sorting import data_structure


def pre_treat_oe(directory, bhv_filepath, spike_dir, start_code=9, end_code=18):

    # Load behavioral data
    bhv = h5py.File(directory + bhv_filepath, "r")["ML"]

    # Load OpenEphis data
    logging.info("Loading OE data")
    session, recordnode, continuous, events = utils_oe.load_op_data(
        directory, config.N_NODE, config.RECORDING_NUM
    )

    # Select the timestamps of continuous data from 10 ms before the first event
    # This is done to reduce the data
    logging.info("Selecting OE timestamps")
    filtered_timestamps, start_time = utils_oe.select_timestamps(
        continuous.timestamps, events.timestamp, config.FS
    )

    # Compute LFP
    logging.info("Computing LFPs")
    selected_channels = np.arange(0, 33)
    n_channels = len(selected_channels)
    LFP_ds = utils_oe.butter_lowpass_filter(
        continuous.samples[start_time:, : n_channels - 2],
        fc=config.FC,
        fs=config.FS,
        order=config.ORDER,
        downsample=config.DOWNSAMPLE,
    )
    eyes_ds = utils_oe.signal_downsample(
        continuous.samples[start_time:, n_channels - 2 :],
        config.DOWNSAMPLE,
        idx_start=0,
        axis=0,
    )

    # Reconstruct 8 bit words
    logging.info("Reconstructing 8 bit words")
    idx_real_strobes = np.where(
        np.logical_and(
            np.logical_and(events.channel == 8, events.state == 1), events.timestamp > 0
        )
    )[
        0
    ]  # state 1: ON, state 0: OFF
    full_word = utils_oe.reconstruct_8bits_words(
        idx_real_strobes, e_channel=events.channel, e_state=events.state
    )

    # Check if strobe and codes number match
    utils_oe.check_strobes(bhv, full_word, idx_real_strobes)

    # Load data
    print("getting spikes...")

    idx_spiketimes = np.load(directory + spike_dir + "/spike_times.npy", "r").reshape(
        -1
    )
    spiketimes_clusters_id = np.load(
        directory + spike_dir + "/spike_clusters.npy", "r"
    )  # to which neuron the spike times belongs to
    cluster_info = pd.read_csv(
        directory + spike_dir + "/cluster_info.tsv", sep="\t"
    )  # info of each cluster

    # codes of the events

    n_trials = np.sum(full_word == start_code)
    real_strobes = events.timestamp[idx_real_strobes].values
    start_trials = real_strobes[
        full_word == start_code
    ]  # timestamps where trials starts
    end_trials = real_strobes[full_word == end_code]
    spiketimes = continuous.timestamps[idx_spiketimes]  # timestamps of all the spikes
    valid_clusters = cluster_info[
        cluster_info["group"] != "noise"
    ]  # we only keep good and mua groups

    (
        times,
        code_numbers,
        code_times,
        eyes_sample,
        lfp_sample,
        timestamps,
    ) = data_structure.sort_data_trial(
        clusters=valid_clusters,
        spiketimes=spiketimes,
        start_trials=start_trials,
        end_trial=end_trials,
        real_strobes=real_strobes,
        filtered_timestamps=filtered_timestamps,
        spiketimes_clusters_id=spiketimes_clusters_id,
        full_word=full_word,
        LFP_ds=LFP_ds,
        eyes_ds=eyes_ds,
    )

    data = data_structure.build_data_structure(
        clusters=cluster_info,
        times=times,
        code_numbers=code_numbers,
        code_times=code_times,
        eyes_sample=eyes_sample,
        lfp_sample=lfp_sample,
        timestamps=timestamps,
    )

    print("pre_treat_oe successfully run")
