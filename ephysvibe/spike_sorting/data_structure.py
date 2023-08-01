import numpy as np
import os
import logging
import re
import h5py
from ..structures.trials_data import TrialsData
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
    # , [ds_samples[-1]])
    n_trials, n_neurons, n_ts = (
        len(start_trials),
        len(clusters),
        np.max(end_trials - start_trials) + 1,
    )
    #  Define arrays
    sp_samples = np.full((n_trials, n_neurons, n_ts), np.nan)
    # code_numbers = []
    # trial_code_samples = []
    # lfp_values = np.full((n_trials, lfp_ds.shape[0], n_ts), np.nan)
    # samples = np.full((n_trials, n_ts), np.nan)
    # eyes_values = np.full((n_trials, eyes_ds.shape[0], n_ts), np.nan)
    logging.info("Sorting data by trial")
    for trial_i in range(n_trials):  # iterate over trials
        # define trial masks
        sp_mask = np.logical_and(
            spike_sample >= start_trials[trial_i],
            spike_sample <= end_trials[trial_i],
        )
        # events_mask = np.logical_and(
        #     code_samples >= start_trials[trial_i],
        #     code_samples <= end_trials[trial_i],
        # )
        # lfp_mask = np.logical_and(
        #     ds_samples >= start_trials[trial_i],
        #     ds_samples <= end_trials[trial_i],
        # )
        # select spikes
        sp_trial = spike_sample[sp_mask] - start_trials[trial_i]
        id_clusters = spike_clusters[sp_mask]  # to which neuron correspond each spike
        # fill with zeros
        len_trial = int(end_trials[trial_i] - start_trials[trial_i])
        sp_samples[trial_i, :, :len_trial] = np.zeros((sp_samples.shape[1], len_trial))
        # # select code numbers
        # code_numbers.append(
        #     full_word[events_mask].tolist()
        # )  # all trials have to start & end with the same codes
        # # select code times
        # trial_code_samples.append(
        #     (code_samples[events_mask] - start_trials[trial_i]).tolist()
        # )
        # select lfp
        # lfp_values[trial_i, :, : np.sum(lfp_mask)] = lfp_ds[:, lfp_mask]
        # # select timestamps
        # # samples[trial_i, : np.sum(lfp_mask)] = ds_samples[lfp_mask]
        # # select eyes
        # eyes_values[trial_i, :, : np.sum(lfp_mask)] = eyes_ds[:, lfp_mask]
        for i_c, i_cluster in enumerate(clusters_id):  # iterate over clusters
            # sort spikes in neurons (spiketimestamp)
            idx_cluster = np.where(id_clusters == i_cluster)[0]
            sp_samples[trial_i, i_c, sp_trial[idx_cluster]] = 1
    # # complete with nan
    # length = max(map(len, code_numbers))
    # code_numbers = np.array(
    #     [cn_i + [np.nan] * (length - len(cn_i)) for cn_i in code_numbers]
    # )
    # trial_code_samples = np.array(
    #     [cs_i + [np.nan] * (length - len(cs_i)) for cs_i in trial_code_samples]
    # )

    return sp_samples
    # code_numbers,
    # trial_code_samples,
    # eyes_values,
    # lfp_values,


def save_data(data, output_dir, subject, date_time, area, n_exp, n_record):
    output_dir = os.path.normpath(output_dir)
    path = "/".join([output_dir] + ["session_struct"] + [subject] + [area])
    file_name = date_time + "_" + subject + "_" + area + "_e" + n_exp + "_r" + n_record
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info("Saving data")
    np.save("/".join([path] + [file_name]), data)
    logging.info("Data successfully saved")


# def build_data_structure(
#     clusters,
#     sp_samples,
#     code_numbers,
#     code_samples,
#     eyes_values,
#     lfp_values,
#     samples,
#     blocks,
#     bhv_trial,
# ):
#     sp_data = {
#         "sp_samples": sp_samples,
#         "blocks": blocks,
#         "code_numbers": code_numbers,
#         "code_samples": code_samples,
#         "eyes_values": eyes_values,
#         "lfp_values": lfp_values,
#         "samples": samples,
#         "clusters_id": clusters["cluster_id"].values,
#         "clusters_ch": clusters["ch"].values,
#         "clustersgroup": clusters["group"].values,
#         "clusterdepth": clusters["depth"].values,
#     }
#     data = {"sp_data": sp_data, "bhv": bhv_trial}

#     return data


def restructure(
    start_trials,
    end_trials,
    cluster_info,
    spike_sample,
    real_strobes,
    spike_clusters,
    full_word,
    bhv,
):

    (
        sp_samples,
        # code_numbers,
        # code_samples,
        # eyes_values,
        # lfp_values,
    ) = sort_data_trial(
        clusters=cluster_info,
        spike_sample=spike_sample,
        start_trials=start_trials,
        end_trials=end_trials,
        # real_strobes=real_strobes,
        # ds_samples=ds_samples,
        spike_clusters=spike_clusters,
        # full_word=full_word,
        # lfp_ds=lfp_ds,
        # eyes_ds=eyes_ds,
    )
    # esto se checkea cuando se genera real_strobes
    # # check if code_numbers == bhv.code_numbers
    # if np.nansum(code_numbers - bhv.code_numbers[: code_numbers.shape[0]]) != 0:
    #     logging.error("bhv.code_numbers != code_numbers")
    #     raise ValueError

    data = TrialsData(
        sp_samples=sp_samples,
        # eyes_values=eyes_values,
        # lfp_values=lfp_values,
        code_samples=code_samples,
        clusters_id=cluster_info["cluster_id"].values,
        clusters_ch=cluster_info["ch"].values,
        clustersgroup=cluster_info["group"].values,
        clusterdepth=cluster_info["depth"].values,
        start_trials=start_trials,
    )
    return data
