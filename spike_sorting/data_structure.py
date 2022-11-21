import numpy as np
import os
import logging
import re
import h5py


def bhv_to_dictionary(bhv):
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            # split node name
            split_name = re.split(r"[/]", node.name)
            if split_name[2] != "MLConfig" and split_name[2] != "TrialRecord":
                n_trial = int(re.split(r"[Trial]", split_name[2])[-1])
                node_name = split_name[-1]
                node_data = np.array(node)
                if len(split_name) > 4 and split_name[4] == "Attribute":
                    node_name = split_name[4] + split_name[5] + split_name[6]
                    if node_name == "Attribute11" or node_name == "Attribute21":
                        # cases where data is saved in utf-8
                        node_data = np.array(node).item().decode("utf-8")
                if len(split_name) > 4 and split_name[4] == "Position":
                    node_name = split_name[4] + split_name[5]

                bhv_res[n_trial - 1][node_name] = node_data

    trials_keys = list(bhv.keys())[1:-1]
    # create a list of dicts, where each dict is a trial
    bhv_res = []
    for trial in trials_keys:
        bhv_res.append({"trial": re.split(r"[Trial]", trial)[-1]})

    bhv.visititems(visitor_func)

    return bhv_res


def sort_data_trial(
    clusters,
    spiketimes,
    start_trials,
    real_strobes,
    filtered_timestamps,
    spiketimes_clusters_id,
    full_word,
    LFP_ds,
    eyes_ds,
):
    clusters_id = clusters["cluster_id"].values
    start_trials = np.append(start_trials, [filtered_timestamps[-1]])
    n_trials = len(start_trials) - 1

    times = []  #  n_trials x n_neurons x n_times
    code_numbers = []
    code_times = []
    lfp_sample = []
    timestamps = []
    eyes_sample = []

    for trial_i in range(n_trials):  # iterate over trials

        # define trial masks
        sp_mask = np.logical_and(
            spiketimes >= start_trials[trial_i], spiketimes < start_trials[trial_i + 1]
        )
        events_mask = np.logical_and(
            real_strobes >= start_trials[trial_i],
            real_strobes < start_trials[trial_i + 1],
        )
        lfp_mask = np.logical_and(
            filtered_timestamps >= start_trials[trial_i],
            filtered_timestamps < start_trials[trial_i + 1],
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
    n_block = data[0]["block"]
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
    bhv_trial,
):
    sp_data = {
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

    bhv_trial = bhv_trial
    return [sp_data, bhv_trial]


def build_bhv_structure(
    block,
):
    data = {
        "block": int(block),
    }
    return data
