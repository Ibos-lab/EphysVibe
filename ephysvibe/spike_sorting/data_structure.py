import numpy as np
import pandas as pd
import os
import logging
import re
import h5py

from collections import defaultdict


def bhv_to_dictionary(bhv):
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            # split node name
            split_name = re.split(r"[/]", node.name)
            if split_name[2] != "MLConfig" and split_name[2] != "TrialRecord":
                n_trial = int(re.split(r"[Trial]", split_name[2])[-1])
                node_name = split_name[-1]

                node_data = np.array(node)
                if len(np.array(node).shape) != 0 and np.array(node).shape[0] == 1:
                    node_data = np.squeeze(node_data, axis=0)

                if len(split_name) > 4 and split_name[4] == "Attribute":
                    node_name = split_name[4] + split_name[5] + split_name[6]

                    if (
                        node_name == "Attribute11"
                        or node_name == "Attribute21"
                        or node_name == "Attribute12"
                    ):
                        # cases where data is saved in utf-8
                        if node_data.item() != 0:
                            node_data = np.array(
                                node_data.item().decode("utf-8")
                            )  # np.array(node)
                if len(split_name) > 4 and split_name[4] == "Position":
                    node_name = split_name[4] + split_name[5]

                bhv_res[n_trial - 1][node_name] = node_data

    trials_keys = list(bhv.keys())[1:-1]
    # create a list of dicts, where each dict is a trial
    bhv_res = []
    for trial in trials_keys:
        bhv_res.append({"trial": re.split(r"[Trial]", trial)[-1]})
    bhv.visititems(visitor_func)
    result = defaultdict(list)
    for i in range(len(bhv_res)):
        current = bhv_res[i]
        for key, value in current.items():
            for j in range(len(value)):
                result[key].append(value[j])
    return bhv_res


# TODO: adapt this function to create the neuron structures
def sort_data_trial(
    clusters,
    spike_sample,
    start_trials,
    end_trials,
    # code_samples,
    # ds_samples,
    spike_clusters,
    # full_word,
    # lfp_ds,
    # eyes_ds,
):
    clusters_id = clusters["cluster_id"].values

    n_trials, n_neurons, n_ts = (
        len(start_trials),
        len(clusters),
        np.max(end_trials - start_trials) + 1,
    )
    #  Define arrays
    sp_samples = np.full((n_trials, n_neurons, n_ts), np.nan)
    logging.info("Sorting data by trial")
    for trial_i in range(n_trials):  # iterate over trials
        # define trial masks
        sp_mask = np.logical_and(
            spike_sample >= start_trials[trial_i],
            spike_sample <= end_trials[trial_i],
        )
        # select spikes
        sp_trial = spike_sample[sp_mask] - start_trials[trial_i]
        id_clusters = spike_clusters[sp_mask]  # to which neuron correspond each spike
        # fill with zeros
        len_trial = int(end_trials[trial_i] - start_trials[trial_i])
        sp_samples[trial_i, :, :len_trial] = np.zeros((sp_samples.shape[1], len_trial))

        for i_c, i_cluster in enumerate(clusters_id):  # iterate over clusters
            # sort spikes in neurons (spiketimestamp)
            idx_cluster = np.where(id_clusters == i_cluster)[0]
            sp_samples[trial_i, i_c, sp_trial[idx_cluster]] = 1

    return sp_samples


def get_clusters_spikes(
    clusters: pd.DataFrame,
    spike_sample: np.ndarray,
    spike_clusters: np.ndarray,
) -> np.ndarray:
    """_summary_

    Args:
        clusters (pd.DataFrame): _description_
        spike_sample (np.ndarray): _description_
        spike_clusters (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    # remap cluster ID to numbers from 0 to n clusters
    i_cluster = clusters["i_cluster"].values
    clusters_id = clusters["cluster_id"].values
    spike_clusters = pd.Series(spike_clusters).replace(clusters_id, i_cluster).values
    # create sp matrix
    n_neurons = len(i_cluster)
    time = max(spike_sample)
    sp_samples = np.zeros((n_neurons, time + 1))
    sp_samples[spike_clusters, spike_sample] = 1

    return sp_samples


def save_data(data, output_dir, subject, date_time, area, n_exp, n_record):
    output_dir = os.path.normpath(output_dir)
    path = "/".join([output_dir] + ["session_struct"] + [subject] + [area])
    file_name = date_time + "_" + subject + "_" + area + "_e" + n_exp + "_r" + n_record
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info("Saving data")
    np.save("/".join([path] + [file_name]), data)
    logging.info("Data successfully saved")
