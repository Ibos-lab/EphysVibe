import numpy as np
import os


def sort_data_trial(
    bhv,
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
    trial_keys = list(bhv.keys())[1:-1]
    n_trials = len(trial_keys)

    block = np.zeros(n_trials)
    times = []  #  n_trials x n_neurons x n_times
    code_numbers = []
    code_times = []
    lfp_sample = []
    timestamps = []
    eyes_sample = []

    for trial_i in range(n_trials):  # iterate over trials

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
        )  # all trials have to start & end with the same codes
        # split code times
        code_times.append(real_strobes[events_mask])
        # split lfp
        lfp_sample.append(LFP_ds[:, lfp_mask].tolist())
        # timestamps
        timestamps.append(filtered_timestamps[lfp_mask])
        # eyes
        eyes_sample.append(eyes_ds[lfp_mask].tolist())
        # blocks
        block[trial_i] = bhv[trial_keys[trial_i]]["Block"][0][0]

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
        block.astype(int),
    )


def save_data(data, save_dir, info):
    n_block = data["block"][0]

    path = (
        save_dir
        + info["subject"]
        + "/"
        + info["area"]
        + "/"
        + info["date"]
        + "/"
        + str(n_block)
        + "/"
        + "res"
    )
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path, data)


def build_data_structure(
    clusters,
    times,
    code_numbers,
    code_times,
    eyes_sample,
    lfp_sample,
    timestamps,
    block,
    save_dir,
    info,
):
    # Do a for loop where it creates and saves a data structure for each block
    n_blocks = np.unique(block)

    for n, i_bk in enumerate(n_blocks):
        trials_bk = np.where(block == i_bk)[0].tolist()

        data = {
            "times": times[trials_bk],
            "block": block[trials_bk],
            "clusters_id": clusters["cluster_id"].values,
            "clustersch": clusters["ch"].values,
            "clustersgroup": clusters["group"].values,
            "clusterdepth": clusters["depth"].values,
            "clusterdepth": clusters["depth"].values,
            "code_numbers": code_numbers[trials_bk],
            "code_times": code_times[trials_bk],
            "eyes_sample": eyes_sample[trials_bk],
            "lfp_sample": lfp_sample[trials_bk],
            "timestamps": timestamps[trials_bk],
        }
        save_data(data, save_dir, info)
