import glob
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import select_trials
from ephysvibe.spike_sorting import config
from ephysvibe.task import def_task, task_constants
from collections import defaultdict
from typing import Dict
import logging


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from multiprocessing import Pool

seed = 2023


def get_fr_df(filepath, in_out, cgroup, e_align):
    py_f = np.load(filepath, allow_pickle=True).item(0)
    sp = py_f["sp_data"]
    bhv = py_f["bhv"]
    trial_idx = select_trials.select_trials_block(sp, n_block=1)
    trial_idx = select_trials.select_correct_trials(bhv, trial_idx)
    task = def_task.create_task_frame(trial_idx, bhv, task_constants.SAMPLES_COND)
    neurons = np.where((sp["clustersgroup"] == cgroup))[0]
    fr_samples = firing_rate.fr_by_sample_neuron(
        sp=sp,
        neurons=neurons,
        task=task,
        in_out=in_out,
        kernel=0,
        e_align=e_align,
        plot=False,
    )
    return fr_samples


def load_fr_samples(
    path: str,
    in_out: int,
    cgroup: str,
    win_size: int,
    step: int,
    fix_duration: int,
    sample_duration: int,
):
    df = get_fr_df(path, in_out, cgroup, e_align=2)
    rolling_df = (
        df.loc[:, :"neuron"]
        .iloc[:, :-1]
        .rolling(window=win_size, axis=1, step=step, min_periods=1)
        .mean()
    )
    rolling_df_sample = rolling_df.iloc[
        :,
        int((df["sample_on"][0] - fix_duration) / step) : int(
            (df["sample_on"][0] + sample_duration) / step
        ),
    ]
    df = get_fr_df(path, in_out, cgroup, e_align=4)
    rolling_df = (
        df.loc[:, :"neuron"]
        .iloc[:, :-1]
        .rolling(window=win_size, axis=1, step=step, min_periods=1)
        .mean()
    )
    rolling_df_test = rolling_df.iloc[
        :, int((df["test_on_1"][0] - 400) / step) :
    ]  # int((df['test_on_1'][0]+2500)/step)
    rolling_df = pd.concat([rolling_df_sample, rolling_df_test], axis=1)
    rolling_df.columns = np.arange(rolling_df.shape[1])
    rolling_df = pd.concat([rolling_df, df[["neuron", "sample", "trial_idx"]]], axis=1)
    return rolling_df


## svm
def sample_df(fr_samples, seed, max_n_trials):
    all_df = []
    for fr_s in fr_samples:  # days
        for i_sample in fr_s["sample"].unique():
            n_df = fr_s[fr_s["sample"] == i_sample]
            sam_df = (
                n_df[n_df["neuron"] == n_df["neuron"].iloc[0]]
                .sample(min(max_n_trials), random_state=seed)
                .reset_index(drop=True)[["sample", "trial_idx"]]
            )
            sam_df = pd.merge(fr_s, sam_df, on=["sample", "trial_idx"], how="inner")
            sam_df["trial_idx"].replace(
                sam_df["trial_idx"].unique(),
                np.arange(0, min(max_n_trials)),
                inplace=True,
            )
            all_df.append(sam_df)
    all_df = pd.concat(all_df)
    all_df = all_df.replace(np.nan, 0)
    return all_df


def compute_window_matrix(all_df, n_win, max_n_trials):
    y, all_samples = [], []
    for i_sample in all_df["sample"].unique():
        n_df = all_df[all_df["sample"] == i_sample]
        data = n_df[[n_win, "neuron", "trial_idx"]]
        n_pivot = pd.pivot_table(
            data, values=n_win, index="trial_idx", columns="neuron"
        ).reset_index(
            drop=True
        )  # .loc[:,0:]
        all_samples.append(n_pivot)
        y.append([i_sample] * min(max_n_trials))
    return pd.concat(all_samples).reset_index(drop=True), np.concatenate(y)


def run_svm_decoder(model, fr_samples, windows, it_seed, n_it, le):
    scores = np.zeros((windows))
    all_df = sample_df(fr_samples, it_seed[n_it])
    all_df["sample"] = le.transform(all_df["sample"])
    for n_win in np.arange(0, windows):
        #  select trials randomly
        X, y = compute_window_matrix(all_df, n_win)
        # split in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=it_seed[n_it], stratify=y
        )
        X_train["label_encoder"] = y_train
        # Sample with replacement (only train set)
        X_train = X_train.sample(len(X_train), random_state=it_seed[n_it], replace=True)
        y_train = X_train["label_encoder"]
        X_train = X_train.iloc[:, :-1]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        scores[n_win] = metrics.accuracy_score(y_test, y_predict)
    return scores


def main(
    fr_paths: Path,
    area: str,
    output_dir: Path,
    cgroup: str,
    in_out: int,
):

    file1 = open(fr_paths, "r")
    Lines = file1.readlines()

    # load all pfc files
    paths = []
    for line in Lines:
        paths.append(line.strip())

    e_align = 2
    win_size = 100
    step = 10
    fix_duration = 200
    sample_duration = 450
    fr_samples = []
    max_n_trials = []
    num_neurons = 0

    with Pool(5) as pool:
        async_fr = [
            pool.apply_async(
                load_fr_samples,
                args=(
                    paths[n],
                    in_out,
                    cgroup,
                    win_size,
                    step,
                    fix_duration,
                    sample_duration,
                ),
            )
            for n in np.arange(len(paths))
        ]
        fr_roll = [asc.get() for asc in async_fr]

    for rolling_df in fr_roll:
        # max number of trials that can be used
        max_n_trials.append(
            rolling_df[["neuron", "sample"]][rolling_df["neuron"] == 1]
            .groupby(["sample"])
            .count()
            .min()[0]
        )
        # rename neurons
        unique_neurons = rolling_df["neuron"].unique()
        rolling_df["neuron"].replace(
            unique_neurons,
            np.arange(num_neurons, num_neurons + len(unique_neurons)),
            inplace=True,
        )
        num_neurons += len(unique_neurons)
        # rename trials
        unique_trials = rolling_df["trial_idx"].unique()
        rolling_df["trial_idx"].replace(
            unique_trials, np.arange(len(unique_trials)), inplace=True
        )
        fr_samples.append(rolling_df)

    model = SVC(kernel="linear", C=20, decision_function_shape="ovr", gamma=0.001)

    le = LabelEncoder()
    le.fit(pd.concat(fr_samples)["sample"].unique())
    # all_df['sample']=le.transform(all_df['sample'])
    n_iterations = 100
    rng = np.random.default_rng(seed=seed)
    it_seed = rng.integers(low=1, high=2023, size=n_iterations, dtype=int)
    windows = 354  # 0
    with Pool(10) as pool:
        async_scores = [
            pool.apply_async(
                run_svm_decoder, args=(model, fr_samples, windows, it_seed, n, le)
            )
            for n in np.arange(n_iterations)
        ]
        scores2 = [asc.get() for asc in async_scores]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = ((np.arange(0, len(scores2[0]))) - fix_duration / 10) / 100
    ax.plot(x, np.array(scores2).mean(axis=0), label="Accuracy")
    ss = np.sum(np.array(scores2) < 0.2, axis=0) / np.array(scores2).shape[0]
    mask_inf = ss <= 0.05
    mask_sup = ss > 0.05
    ax.fill_between(
        x,
        y1=min(np.array(scores2).mean(axis=0)),
        y2=max(np.array(scores2).mean(axis=0)),
        where=ss <= 0.05,
        color="grey",
        alpha=0.5,
        label="Below 5%",
    )
    fig.legend(fontsize=9)
    condition = {-1: "out", 1: "in"}
    fig.suptitle(
        "%s, condition: %s, group: %s, %d neurons"
        % (
            area,
            condition[in_out],
            cgroup,
            pd.concat(fr_samples)["neuron"].unique()[-1],
        )
    )
    fig.savefig(
        output_dir
        + "/"
        + area
        + "_"
        + n_iterations
        + "it_"
        + cgroup
        + "_"
        + condition[in_out]
        + ".jpg"
    )


if __name__ == "__main__":
    "/home/INT/losada.c/Documents/codes/run_pipelines/paths_decoding.txt"
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fr_paths", help="Path to txt file", type=Path)
    parser.add_argument("area", help="area", type=str)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--in_out", default=1, help="1 in, -1 out of the rf", type=int)

    parser.add_argument(
        "--cgroup", "-g", default="good", help="cluster goup, good, mua, all", type=str
    )
    args = parser.parse_args()
    try:
        main(args.fr_paths, args.area, args.output_dir, args.in_out, args.cgroup)
    except FileExistsError:
        logging.error("filepath does not exist")
