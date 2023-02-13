import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict


def test_stim(task, bhv_idx):
    for key, value in bhv_idx.items():
        if "Stim_Filename_" in key:
            val = value.item(0).decode("utf-8")
            if key[-1] == "d":  # distractor
                task["test_stimuli_" + key[-2:]] += [
                    val[-11:-10] + val[-9:-6] + val[-5:-4]
                ]
            else:
                task["test_stimuli_" + key[-1]] += [
                    val[-11:-10] + val[-9:-6] + val[-5:-4]
                ]
                n_test = int(key[-1])

    for n in range(n_test + 1, 6):
        task["test_stimuli_" + str(n)] += [""]
        task["test_stimuli_" + str(n) + "d"] += [""]
    return task, n_test


def create_task_frame(condition, test_stimuli, samples_cond):
    task: Dict[str, list] = defaultdict(list)
    for key_cond, n_cond in samples_cond.items():
        idx = np.where(np.in1d(condition, n_cond))[0]
        n_test_stimuli = np.sum(~np.isnan(test_stimuli[idx]), axis=1)
        code = 7 - (n_cond[-1] - condition[idx])
        sample = np.where(
            np.logical_and(code == 7, n_test_stimuli == 5), "o0_c0", key_cond[:5]
        )

        task["trial_idx"] += idx.tolist()
        task["sample"] += sample.tolist()
        task["in_out"] += [key_cond[6:]] * len(idx)
        task["n_test_stimuli"] += n_test_stimuli.tolist()
        task["code"] += code.astype(int).tolist()

    return pd.DataFrame(task)


def info_task(task):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    data = (
        task[["sample_id", "in_out"]]
        .groupby(["sample_id", "in_out"], as_index=False)
        .size()
    )
    sns.barplot(data=data, x="sample_id", y="size", hue="in_out", ax=ax[0])
    data = (
        task[(task["in_out"] == 1) & (task["sample_id"] != "o0_c0")][
            ["sample_id", "n_test_stimuli"]
        ]
        .groupby(["sample_id", "n_test_stimuli"], as_index=False)
        .size()
    )
    sns.barplot(data=data, x="sample_id", y="size", hue="n_test_stimuli", ax=ax[1])
    data = task.groupby(
        task.drop(
            [
                "idx_trial",
                "test_stimuli_1d",
                "test_stimuli_2d",
                "test_stimuli_3d",
                "test_stimuli_4d",
                "test_stimuli_5d",
            ],
            inplace=False,
            axis=1,
        ).columns.to_list(),
        as_index=False,
    ).size()
    data = data[data["size"] > 1].replace("", float(np.nan))
    data.dropna(how="all", axis=1, inplace=True)
    data.sort_values(by=["sample_id", "code", "size"])
    return fig, data
