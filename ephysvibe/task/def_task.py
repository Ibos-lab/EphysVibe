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


def create_task_frame(trial_idx, bhv, samples_cond):
    task: Dict[str, list] = defaultdict(list)  # {}
    for idx in trial_idx:
        task["idx_trial"] += [idx]
        cond = int(bhv[idx]["Condition"][0][0])
        o1_c1_out = samples_cond["o1_c1_in"] + 28
        o1_c5_out = samples_cond["o1_c5_in"] + 28
        o5_c1_out = samples_cond["o5_c1_in"] + 28
        o5_c5_out = samples_cond["o5_c5_in"] + 28
        # IN
        if cond in samples_cond["o1_c1_in"]:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o1_c1"
            code = 7 - (samples_cond["o1_c1_in"][-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        elif cond in samples_cond["o1_c5_in"]:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o1_c5"
            code = 7 - (samples_cond["o1_c5_in"][-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        elif cond in samples_cond["o5_c1_in"]:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o5_c1"
            code = 7 - (samples_cond["o5_c1_in"][-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        elif cond in samples_cond["o5_c5_in"]:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o5_c5"
            code = 7 - (samples_cond["o5_c5_in"][-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        # OUT
        elif cond in o1_c1_out:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o1_c1"
            code = 7 - (o1_c1_out[-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [-1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        elif cond in o1_c5_out:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o1_c5"
            code = 7 - (o1_c5_out[-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [-1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        elif cond in o5_c1_out:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o5_c1"
            code = 7 - (o5_c1_out[-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [-1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]
        elif cond in o5_c5_out:
            task, n_test = test_stim(task, bhv[idx])
            sample_id = "o5_c5"
            code = 7 - (o5_c5_out[-1] - cond)
            if code == 7:
                sample_id = "o0_c0"
            task["sample_id"] += [sample_id]
            task["in_out"] += [-1]
            task["n_test_stimuli"] += [n_test]
            task["code"] += [code]

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
