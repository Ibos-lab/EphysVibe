import numpy as np
import os
import logging


def sort_data_trial(
    clusters,
    spiketimes,
    start_trials,
    end_trial,
    real_strobes,
    filtered_timestamps,
    spiketimes_clusters_id,
    full_word,
    LFP_ds,
    eyes_ds,
):
    clusters_id = clusters["cluster_id"].values
    n_trials = len(start_trials)

    times = []  #  n_trials x n_neurons x n_times
    code_numbers = []
    code_times = []
    lfp_sample = []
    timestamps = []
    eyes_sample = []

    for trial_i in range(n_trials):  # iterate over trials

        # define trial masks
        sp_mask = np.logical_and(
            spiketimes >= start_trials[trial_i], spiketimes <= end_trial[trial_i]
        )
        events_mask = np.logical_and(
            real_strobes >= start_trials[trial_i], real_strobes <= end_trial[trial_i]
        )
        lfp_mask = np.logical_and(
            filtered_timestamps >= start_trials[trial_i],
            filtered_timestamps <= end_trial[trial_i],
        )
        # select spikes
        sp_trial = spiketimes[sp_mask]
        id_clusters = spiketimes_clusters_id[
            sp_mask
        ]  # to which neuron correspond each spike

        # select code numbers
        code_numbers.append(
            full_word[events_mask]
        )  # all trials have to start & end with the same codes
        # select code times
        code_times.append(real_strobes[events_mask])
        # select lfp
        lfp_sample.append(LFP_ds[:, lfp_mask].tolist())
        # select timestamps
        timestamps.append(filtered_timestamps[lfp_mask])
        # select eyes
        eyes_sample.append(eyes_ds[:, lfp_mask].tolist())

        spiketimes_trial = []  # n_neurons x n_times
        for i_cluster in clusters_id:  # iterate over clusters
            # sort spikes in neurons (spiketimestamp)
            idx_cluster = np.where(id_clusters == i_cluster)[0]
            spiketimes_trial.append(sp_trial[idx_cluster])

        times.append(spiketimes_trial)

    return (
        np.array(times, dtype=object),
        np.array(code_numbers, dtype=object),
        np.array(code_times, dtype=object),
        np.array(eyes_sample, dtype=object),
        np.array(lfp_sample, dtype=object),
        np.array(timestamps, dtype=object),
    )


def save_data(data, save_dir, subject, date_time, area):
    n_block = data["block"]
    path = (
        str(save_dir)
        + "/"
        + subject
        + "/"
        + area
        + "/"
        + date_time
        + "/"
        + str(n_block)
    )
    file_name = "/" + subject + "_" + area + "_" + date_time + "_" + str(n_block)
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path + file_name, data)
    logging.info("Data successfully saved")


def build_data_structure(
    clusters,
    times,
    code_numbers,
    code_times,
    eyes_sample,
    lfp_sample,
    timestamps,
    block,
):
    data = {
        "times": times,
        "block": int(block),
        "clusters_id": clusters["cluster_id"].values,
        "clustersch": clusters["ch"].values,
        "clustersgroup": clusters["group"].values,
        "clusterdepth": clusters["depth"].values,
        "clusterdepth": clusters["depth"].values,
        "code_numbers": code_numbers,
        "code_times": code_times,
        "eyes_sample": eyes_sample,
        "lfp_sample": lfp_sample,
        "timestamps": timestamps,
    }
    return data
