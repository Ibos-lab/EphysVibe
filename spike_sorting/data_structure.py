import numpy as np


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
    times = []  #  n_trials x n_neurons x n_times
    clusters_id = clusters["cluster_id"].values
    code_numbers = []
    code_times = []
    lfp_sample = []
    timestamps = []
    eyes_sample = []

    for trial_i in range(len(start_trials)):  # iterate over trials

        # define trial masks
        sp_mask = np.logical_and(
            spiketimes >= start_trials[trial_i], spiketimes < end_trial[trial_i]
        )
        events_mask = np.logical_and(
            real_strobes >= start_trials[trial_i], real_strobes <= end_trial[trial_i]
        )
        lfp_mask = np.logical_and(
            filtered_timestamps >= start_trials[trial_i],
            filtered_timestamps <= end_trial[trial_i],
        )
        # split spikes
        sp_trial = spiketimes[sp_mask]
        id_clusters = spiketimes_clusters_id[
            sp_mask
        ]  # to which neuron correspond each spike

        # split code numbers
        code_numbers.append(
            full_word[events_mask]
        )  # all trials have to start and end with the same codes
        # split code times
        code_times.append(real_strobes[events_mask])
        # split lfp
        lfp_sample.append(LFP_ds[:, lfp_mask])
        # timestamps
        timestamps.append(filtered_timestamps[lfp_mask])
        # eyes
        eyes_sample.append(eyes_ds[lfp_mask])
        spiketimes_trial = []  # n_neurons x n_times

        for n, i_cluster in enumerate(clusters_id):  # iterate over clusters

            # sort spikes in neurons (spiketimestamp)
            idx_cluster = np.where(id_clusters == i_cluster)[0]
            spiketimes_trial.append(sp_trial[idx_cluster])

        times.append(spiketimes_trial)

    return times, code_numbers, code_times, eyes_sample, lfp_sample, timestamps


def build_data_structure(
    clusters, times, code_numbers, code_times, eyes_sample, lfp_sample, timestamps
):

    data = {
        "times": times,
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
