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
    spike_sample,
    start_trials,
    real_strobes,
    ds_samples,
    sp_ksamples_clusters_id,
    full_word,
    lfp_ds,
    eyes_ds,
):
    clusters_id = clusters["cluster_id"].values
    start_trials = np.append(start_trials, [ds_samples[-1]])
    n_trials = len(start_trials) - 1

    sp_samples = []  #  n_trials x n_neurons x n_times
    code_numbers = []
    code_samples = []
    lfp_values = []
    samples = []
    eyes_values = []
    logging.info("Sorting data by trial")
    for trial_i in range(n_trials):  # iterate over trials

        # define trial masks
        sp_mask = np.logical_and(
            spike_sample >= start_trials[trial_i],
            spike_sample < start_trials[trial_i + 1],
        )
        events_mask = np.logical_and(
            real_strobes >= start_trials[trial_i],
            real_strobes < start_trials[trial_i + 1],
        )
        lfp_mask = np.logical_and(
            ds_samples >= start_trials[trial_i],
            ds_samples < start_trials[trial_i + 1],
        )
        # select spikes
        sp_trial = spike_sample[sp_mask]
        id_clusters = sp_ksamples_clusters_id[
            sp_mask
        ]  # to which neuron correspond each spike

        # select code numbers
        code_numbers.append(
            full_word[events_mask]
        )  # all trials have to start & end with the same codes
        # select code times
        code_samples.append(real_strobes[events_mask])
        # select lfp
        lfp_values.append(lfp_ds[:, lfp_mask])
        # select timestamps
        samples.append(ds_samples[lfp_mask])
        # select eyes
        eyes_values.append(eyes_ds[:, lfp_mask])

        spiketimes_trial = []  # n_neurons x n_times
        for i_cluster in clusters_id:  # iterate over clusters
            # sort spikes in neurons (spiketimestamp)
            idx_cluster = np.where(id_clusters == i_cluster)[0]
            spiketimes_trial.append(sp_trial[idx_cluster])

        sp_samples.append(spiketimes_trial)

    return (
        np.array(sp_samples, dtype=object),
        np.array(code_numbers, dtype=object),
        np.array(code_samples, dtype=object),
        eyes_values,
        lfp_values,
        np.array(samples, dtype=object),
    )


def save_data(data, output_dir, subject, date_time, area, n_exp, n_record):
    output_dir = os.path.normpath(output_dir)
    path = "/".join([output_dir] + ["session_struct"] + [subject] + [area])
    file_name = date_time + "_" + subject + "_" + area + "_e" + n_exp + "_r" + n_record
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info("Saving data")
    np.save("/".join([path] + [file_name]), data)
    logging.info("Data successfully saved")


def build_data_structure(
    clusters,
    sp_samples,
    code_numbers,
    code_samples,
    eyes_values,
    lfp_values,
    samples,
    blocks,
    bhv_trial,
):
    sp_data = {
        "sp_samples": sp_samples,
        "blocks": np.array(blocks, dtype=int),
        "code_numbers": code_numbers,
        "code_samples": code_samples,
        "eyes_values": eyes_values,
        "lfp_values": lfp_values,
        "samples": samples,
        "clusters_id": clusters["cluster_id"].values,
        "clusters_ch": clusters["ch"].values,
        "clustersgroup": clusters["group"].values,
        "clusterdepth": clusters["depth"].values,
    }
    data = {"sp_data": sp_data, "bhv": bhv_trial}

    return data


def restructure(
    start_trials,
    blocks,
    cluster_info,
    spike_sample,
    real_strobes,
    ds_samples,
    sp_ksamples_clusters_id,
    full_word,
    lfp_ds,
    eyes_ds,
    dict_bhv,
):

    (
        sp_samples,
        code_numbers,
        code_samples,
        eyes_values,
        lfp_values,
        samples,
    ) = sort_data_trial(
        clusters=cluster_info,
        spike_sample=spike_sample,
        start_trials=start_trials,
        real_strobes=real_strobes,
        ds_samples=ds_samples,
        sp_ksamples_clusters_id=sp_ksamples_clusters_id,
        full_word=full_word,
        lfp_ds=lfp_ds,
        eyes_ds=eyes_ds,
    )

    data = build_data_structure(
        clusters=cluster_info,
        sp_samples=sp_samples,
        code_numbers=code_numbers,
        code_samples=code_samples,
        eyes_values=eyes_values,
        lfp_values=lfp_values,
        samples=samples,
        blocks=blocks,
        bhv_trial=dict_bhv,
    )
    logging.info(" ")
    return data
