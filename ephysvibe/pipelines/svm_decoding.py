import numpy as np
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from ..task import def_task, task_constants
from ..structures.spike_data import SpikeData
from ..structures.bhv_data import BhvData
from collections import defaultdict
from typing import Dict
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import metrics
from multiprocessing import Pool

seed = 2023

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    d_shape = data.shape
    d_avg = np.zeros((d_shape[0], d_shape[1], int(np.floor(d_shape[2] / step))))
    count = 0
    for i_step in np.arange(0, d_shape[2] - step, step):
        d_avg[:, :, count] = np.mean(data[:, :, i_step : i_step + win], axis=2)
        count += 1
    return d_avg


def load_fr_samples(
    sp_path,
    bhv_path,
    cgroup,
    win,
    step,
    in_out="in",
    e_align="sample_on",
    t_before=200,
    to_decode="samples",
):
    logging.info(sp_path)
    logging.info(bhv_path)
    data = SpikeData.from_python_hdf5(sp_path)
    bhv = BhvData.from_python_hdf5(bhv_path)

    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == 1))[0]
    if np.any(np.isnan(data.neuron_cond)):
        neuron_cond = np.ones(len(data.clustersgroup))
    else:
        neuron_cond = data.neuron_cond
    task = def_task.create_task_frame(
        condition=bhv.condition[trial_idx],
        test_stimuli=bhv.test_stimuli[trial_idx],
        samples_cond=task_constants.SAMPLES_COND,
        neuron_cond=neuron_cond,
    )
    if cgroup == "all":
        neurons = np.where(data.clustersgroup != cgroup)[0]
    else:
        neurons = np.where(data.clustersgroup == cgroup)[0]

    task = task[
        np.logical_and(
            np.in1d(task["i_neuron"].values, neurons), task["in_out"] == in_out
        )
    ]

    if to_decode == "samples":
        task = task[task["sample"] != "o0_c0"]
    elif to_decode == "neutral":
        task["sample"].replace(
            ["o1_c1", "o1_c5", "o5_c1", "o5_c5"], "no_neutral", inplace=True
        )
    elif to_decode == "colors":
        task = task[task["sample"] != "o0_c0"]
        task["sample"].replace(["o1_c1", "o5_c1"], "c1", inplace=True)
        task["sample"].replace(["o1_c5", "o5_c5"], "c5", inplace=True)
    elif to_decode == "orientation":
        task = task[task["sample"] != "o0_c0"]
        task["sample"].replace(["o1_c1", "o1_c5"], "o1", inplace=True)
        task["sample"].replace(["o5_c1", "o5_c5"], "o5", inplace=True)
    else:
        logging.error(
            'to_decode must be "samples", "neutral", "colors" or "orientation"'
        )
        raise ValueError
    # split in two groups where the neurons in each have the same trials in in or out
    task_1 = task[task["i_neuron"] == neurons[0]].copy()
    trials_neuron = task_1["trial_idx"].values
    task_1["trial_idx"] = task_1["trial_idx"].replace(
        trials_neuron, np.arange(0, len(trials_neuron))
    )
    t_neurons = task[np.in1d(task["trial_idx"].values, trials_neuron)][
        "i_neuron"
    ].unique()
    t_neurons_2 = neurons[~np.in1d(neurons, t_neurons)]
    task_all = [task_1]
    if len(t_neurons_2) != 0:
        trials_neuron_2 = task[task["i_neuron"] == t_neurons_2[0]]["trial_idx"].values
        trials_neuron = [trials_neuron, trials_neuron_2]
        t_neurons = [t_neurons, t_neurons_2]
        task_2 = task[task["i_neuron"] == t_neurons_2[0]].copy()
        task_2["trial_idx"] = task_2["trial_idx"].replace(
            trials_neuron_2, np.arange(0, len(trials_neuron_2))
        )
        task_all = [task_1, task_2]
    else:
        trials_neuron = [trials_neuron]
        t_neurons = [t_neurons]

    sp_avg_all = []
    for i_task, (trial_idx_n, neurons, task) in enumerate(
        zip(trials_neuron, t_neurons, task_all)
    ):
        min_task = task.groupby(["sample"]).count().min().min()
        # check number of trials
        if min_task >= 30:
            trials_s_on = data.code_samples[
                trial_idx[trial_idx_n],
                np.where(
                    data.code_numbers[trial_idx[trial_idx_n]]
                    == task_constants.EVENTS_B1[e_align]
                )[1],
            ]
            shifts = -(trials_s_on - t_before).astype(int)
            shifts = shifts[:, np.newaxis]
            shift_sp = SpikeData.indep_roll(
                data.sp_samples[trial_idx[trial_idx_n]][:, neurons], shifts, axis=2
            )[:, :, :1600]
            sp_avg = moving_average(shift_sp, win=win, step=step)
            sp_avg_all.append(sp_avg)
        else:
            task_all.pop(i_task)
    return task_all, sp_avg_all


def sample_df(frs_avg, tasks, min_trials, seed):
    sample_dict: Dict[str, list] = defaultdict(list)
    for i_sample in tasks[0]["sample"].unique():
        all_sample_fr = []
        for fr_s, n_task in zip(frs_avg, tasks):  # days
            t_idx = (
                n_task[n_task["sample"] == i_sample]
                .sample(min_trials, random_state=seed)["trial_idx"]
                .values
            )
            sample_fr = fr_s[t_idx]
            all_sample_fr.append(sample_fr)
        all_sample_fr = np.concatenate(all_sample_fr, axis=1)
        sample_dict[i_sample] = all_sample_fr
    return sample_dict


def compute_window_matrix(all_df, n_win):
    y, all_samples = [], []
    for i_sample in all_df.keys():
        n_df = all_df[i_sample]
        data = n_df[:, :, n_win]

        all_samples.append(data)
        y.append([i_sample] * data.shape[0])
    return np.concatenate(all_samples, axis=0), np.concatenate(y)


def run_svm_decoder(model, frs_avg, tasks, windows, min_trials, it_seed, n_it, le):
    scores = np.zeros((windows))
    all_df = sample_df(frs_avg, tasks, min_trials, it_seed[n_it])
    for n_win in np.arange(0, windows):
        #  select trials randomly
        X, y = compute_window_matrix(all_df, n_win)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        y = le.transform(y)
        # split in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=it_seed[n_it], stratify=y
        )
        np.random.seed(it_seed[n_it])
        idx_swr = np.random.choice(
            X_train.shape[0], size=X_train.shape[0], replace=True, p=None
        )
        X_train = X_train[idx_swr]
        y_train = y_train[idx_swr]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        scores[n_win] = metrics.accuracy_score(y_test, y_predict)
    return scores


# plot results
def plot_accuracy(scores, win_steps, neuron_max_shift, x_lim_min, x_lim_max, n_neuron):
    fig, ax = plt.subplots()
    ax.plot(
        ((np.arange(0, len(scores[0])) * win_steps) - neuron_max_shift[n_neuron - 1])
        / 1000,
        scores[:13].mean(axis=0),
    )
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.vlines(0, 0.3, 1, color="k", linestyles="dashed")  # sample on
    ax.hlines(0.5, x_lim_min, x_lim_max, color="gray", linestyles="solid")
    ax.set_title("Is neuron %d engaged in the task?" % (n_neuron))
    ax.set(xlabel="Time (s)", ylabel="SVM classifier accuracy")
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    fig.legend(["Accuracy", "Sample on"], fontsize=9)


def main(
    sp_paths: Path,
    bhv_paths: Path,
    output_dir: Path,
    in_out: int,
    cgroup: str,
    jobs_load: int = 1,
    jobs_svm: int = 1,
    to_decode: str = "samples",
    win_size: int = 50,
):
    logging.info("--- Start ---")
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if paths exist
    if not os.path.exists(sp_paths):
        raise FileExistsError
    if not os.path.exists(bhv_paths):
        raise FileExistsError
    file1 = open(bhv_paths, "r")
    lines_bhv = file1.readlines()
    file1 = open(sp_paths, "r")
    lines_sp = file1.readlines()
    # load all files
    paths_bhv, paths_sp = [], []
    for line in lines_bhv:
        paths_bhv.append(line.strip())
    for line in lines_sp:
        paths_sp.append(line.strip())

    step = 10
    fix_duration = 200
    t_before = 200
    e_align = "sample_on"
    tasks_all, frs_avg_all = [], []
    with Pool(jobs_load) as pool:
        async_fr = [
            pool.apply_async(
                load_fr_samples,
                args=(
                    paths_sp[n],
                    paths_bhv[n],
                    cgroup,
                    win_size,
                    step,
                    in_out,
                    e_align,
                    t_before,
                    to_decode,
                ),
            )
            for n in np.arange(len(paths_sp))
        ]
        for asc in async_fr:
            tasks_all.append(asc.get()[0])
            frs_avg_all.append(asc.get()[1])
    logging.info("load_fr_samples finished")
    tasks, frs_avg = [], []
    for i in range(len(frs_avg_all)):
        if len(frs_avg_all[i]) > 0:
            frs_avg.append(frs_avg_all[i][0])
            tasks.append(tasks_all[i][0])
            if len(frs_avg_all[i]) > 1:
                frs_avg.append(frs_avg_all[i][1])
                tasks.append(tasks_all[i][1])
    # max number of trials that can be used
    min_trials = frs_avg[0].shape[0]
    for rec in range(len(tasks)):
        min_n_trials = tasks[rec].groupby(["sample"]).count().min()[0]
        min_trials = min_n_trials if min_n_trials < min_trials else min_trials
    # define model
    model = SVC(kernel="linear", C=20, decision_function_shape="ovr", gamma=0.001)
    le = LabelEncoder()
    le.fit(tasks[0]["sample"].unique())
    n_iterations = 1000
    rng = np.random.default_rng(seed=seed)
    it_seed = rng.integers(low=1, high=2023, size=n_iterations, dtype=int)
    windows = frs_avg[0].shape[2]
    logging.info("Runing decoder")
    with Pool(jobs_svm) as pool:
        async_scores = [
            pool.apply_async(
                run_svm_decoder,
                args=(model, frs_avg, tasks, windows, min_trials, it_seed, n, le),
            )
            for n in np.arange(n_iterations)
        ]
        scores = [asc.get() for asc in async_scores]

    s_path = os.path.normpath(paths_sp[0]).split(os.sep)
    n_neurons = 0
    for rec in range(len(frs_avg)):
        n_neurons += frs_avg[rec].shape[1]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = ((np.arange(0, len(scores[0]))) - fix_duration / step) * 10
    ax.plot(x, np.array(scores).mean(axis=0), label="Accuracy")
    if to_decode == "samples":
        threshole = 0.25
    else:
        threshole = 0.5
    ss = np.sum(np.array(scores) <= threshole, axis=0) / np.array(scores).shape[0]
    mask_inf = ss <= 0.01
    mask_inf_5 = ss <= 0.05
    # stars
    ax.scatter(
        x[mask_inf],
        [threshole - 0.1] * len(x[mask_inf]),
        color="k",
        marker="*",
        label="Below 1%",
        s=6,
    )
    ax.scatter(
        x[mask_inf_5],
        [threshole - 0.12] * len(x[mask_inf_5]),
        color="r",
        marker="*",
        label="Below 5%",
        s=6,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # delete boundaries
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.legend(fontsize=9)
    fig.suptitle(
        "%s, condition: %s, group: %s, %d neurons,window: %d, steps: %d, trials: %d"
        % (s_path[-3], in_out, cgroup, n_neurons, win_size, step, min_trials)
    )

    fig.savefig(
        "/".join(
            [os.path.normpath(output_dir)]
            + [
                s_path[-3]
                + "_"
                + to_decode
                + "_"
                + str(n_iterations)
                + "it_"
                + cgroup
                + "_"
                + in_out
                + "_win"
                + str(win_size)
                + ".jpg"
            ],
        )
    )

    scores_path = "/".join(
        [os.path.normpath(output_dir)]
        + [
            s_path[-3]
            + "_"
            + to_decode
            + "_"
            + str(n_iterations)
            + "it_"
            + cgroup
            + "_"
            + in_out
            + "_win"
            + str(win_size)
        ],
    )
    # save accuracy
    np.save(scores_path, arr=np.array(scores))

    logging.info("--- End ---")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sp_paths", help="Path to txt file", type=Path)
    parser.add_argument("bhv_paths", help="Path to txt file", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--in_out", default="in", help="in or out of the rf", type=str)

    parser.add_argument(
        "--cgroup", "-g", default="good", help="cluster goup, good, mua, all", type=str
    )
    parser.add_argument("--jobs_load", "-l", default=2, help="", type=int)
    parser.add_argument("--jobs_svm", "-s", default=8, help="", type=int)
    parser.add_argument(
        "--to_decode", "-c", default="samples", help="samples or neutral", type=str
    )
    parser.add_argument("--win_size", "-w", default=50, help="", type=int)

    args = parser.parse_args()
    try:
        main(
            args.sp_paths,
            args.bhv_paths,
            args.output_dir,
            args.in_out,
            args.cgroup,
            args.jobs_load,
            args.jobs_svm,
            args.to_decode,
            args.win_size,
        )
    except FileExistsError:
        logging.error("filepath does not exist")
