# plot spiking activity task b1
import os
import argparse
from pathlib import Path
import logging
import numpy as np
from ..trials.spikes import firing_rate, sp_constants
from ..spike_sorting import config
from ..task import task_constants, def_task
from ..trials.spikes import plot_raster
from ..analysis import circular_stats
import warnings
from matplotlib import pyplot as plt
from ..structures.spike_data import SpikeData
from ephysvibe.structures.bhv_data import BhvData
from typing import Dict
from collections import defaultdict
import pandas as pd
from scipy import stats
import gc

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def get_neurons_info(
    sp_samples: np.ndarray,
    fix_t: int,
    neuron_type: np.ndarray,
    task: pd.DataFrame,
    dur_v: int,
    st_m: int,
    end_m: int,
    st_t: int,
    end_t: int,
    clusterdepth: np.ndarray,
    date: str,
    min_trials: int = 5,
    n_spikes_sec: int = 5,
) -> pd.DataFrame:
    trial_dur = sp_samples.shape[2]
    neurons_info: Dict[str, list] = defaultdict(list)
    i_good, i_mua, n_type = 0, 0, 0
    for i_neuron, type_neuron in enumerate(neuron_type):
        # check and count type of unit
        if type_neuron == "good":
            i_good += 1
            n_type = i_good
        elif type_neuron == "mua":
            i_mua += 1
            n_type = i_mua
        max_in_out = np.array([0, 0])
        larger, v_larger, m_larger, idx_p_min = (
            np.array([False, False]),
            np.array([False, False]),
            np.array([False, False]),
            np.array([0, 0]),
        )
        t_larger = np.nan
        p_in_out, p_v, p_m, vm_index = (
            np.array([None, None]),
            np.array([None, None]),
            np.array([None, None]),
            np.nan,
        )
        tr_min, tr_max, mean_mem_all, mean_visual_all = (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        )
        for i, in_out in enumerate(["in", "out"]):  # iterate by code'
            trial_idx = task[
                (task["i_neuron"] == i_neuron)
                & (task["in_out"] == in_out)
                & (task["sample"] != "o0_c0")
            ]["trial_idx"]
            trial_idx = trial_idx[(sp_samples[trial_idx, i_neuron].sum(axis=1) > 0)]
            n_tr = len(trial_idx)
            true_in_out = "in"
            if n_tr > 3:
                mean_visual = sp_samples[
                    trial_idx, i_neuron, fix_t + 50 : fix_t + dur_v
                ].mean(axis=1)
                mean_mem = sp_samples[
                    trial_idx, i_neuron, fix_t + st_m : fix_t + end_m
                ].mean(axis=1)
                mean_test = sp_samples[
                    trial_idx, i_neuron, fix_t + st_t : fix_t + end_t
                ].mean(axis=1)
                mean_bl = sp_samples[trial_idx, i_neuron, :fix_t].mean(axis=1)
                v_larger[i] = mean_bl.mean() < mean_visual.mean()
                m_larger[i] = mean_bl.mean() < mean_mem.mean()
                t_larger = mean_bl.mean() < mean_test.mean()
                larger[i] = v_larger[i] or m_larger[i] or t_larger

                p_v[i] = stats.ttest_rel(mean_bl, mean_visual)[1]
                p_m[i] = stats.ttest_rel(mean_bl, mean_mem)[1]
                p_t = stats.ttest_rel(mean_bl, mean_test)[1]

                p_in_out[i] = np.min([p_v[i], p_m[i], p_t])
                max_in_out[i] = np.max([v_larger[i], m_larger[i], t_larger])
                idx_p_min[i] = np.argmin([p_v[i], p_m[i], p_t])

                all_mean = sp_samples[trial_idx, i_neuron, : fix_t + end_m].mean(axis=0)
                tr_min[i], tr_max[i] = np.min(all_mean), np.max(all_mean)
                mean_mem_all[i], mean_visual_all[i] = (
                    mean_mem.mean(),
                    mean_visual.mean(),
                )
        if not np.all(np.isnan(tr_min)):
            tr_min_all = np.nanmin(tr_min)
            tr_max_all = np.nanmax(tr_max)  # - tr_min_all
            if not (p_v[0] is None):
                p_and_large_in = np.any(
                    np.logical_and(
                        np.array([p_v[0], p_m[0]]) < 0.05,
                        np.array([v_larger[0], m_larger[0]]),
                    )
                )
            if not (p_v[0] is None) and p_and_large_in:
                mean_mem = (mean_mem_all[0]) / tr_max_all  # - tr_min_all
                mean_visual = (mean_visual_all[0]) / tr_max_all  # - tr_min_all
                vm_index = (mean_mem - mean_visual) / (mean_mem + mean_visual)
            else:
                if not (p_v[1] is None):
                    p_and_large_out = np.any(
                        np.logical_and(
                            np.array([p_v[1], p_m[1]]) < 0.05,
                            np.array([v_larger[1], m_larger[1]]),
                        )
                    )
                if not (p_v[1] is None) and p_and_large_out:
                    true_in_out = "out"
                    mean_mem = (mean_mem_all[1]) / tr_max_all  # - tr_min_all
                    mean_visual = (mean_visual_all[1]) / tr_max_all  # - tr_min_all
                    vm_index = (mean_mem - mean_visual) / (mean_mem + mean_visual)

        for i, in_out in enumerate(["in", "out"]):  # iterate by code'
            neurons_info["array_position"] += [i_neuron]
            neurons_info["cluster"] += [n_type]
            neurons_info["group"] += [type_neuron]
            neurons_info["in_out"] += [in_out]
            neurons_info["true_in_out"] += [true_in_out]
            neurons_info["vm_index"] += [vm_index]
            neurons_info["p"] += [p_in_out[1]]
            neurons_info["larger"] += [larger[i]]
            neurons_info["v_larger"] += [v_larger[i]]
            neurons_info["p_v"] += [p_v[i]]
            neurons_info["m_larger"] += [m_larger[i]]
            neurons_info["p_m"] += [p_m[i]]
            neurons_info["depth"] += [clusterdepth[i_neuron]]
            neurons_info["date"] += [date]
    neurons_info = pd.DataFrame(neurons_info)
    return neurons_info


def main(filepath: Path, bhv_path: Path, output_dir: Path, e_align: str, t_before: int):
    s_path = os.path.normpath(filepath).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-3]] + ["b1"])
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if filepath exist
    if not os.path.exists(filepath):
        raise FileExistsError
    logging.info("-- Start --")
    data = SpikeData.from_python_hdf5(filepath)
    bhv = BhvData.from_python_hdf5(bhv_path)
    # Select trials and create task frame
    block = 1
    trials_block = np.where(np.logical_and(data.trial_error == 0, data.block == block))[
        0
    ]
    logging.info("Number of clusters: %d" % len(data.clustersgroup))
    if np.any(np.isnan(data.neuron_cond)):
        neuron_cond = np.ones(len(data.clustersgroup))
    else:
        neuron_cond = data.neuron_cond
    task = def_task.create_task_frame(
        condition=bhv.condition[trials_block],
        test_stimuli=bhv.test_stimuli[trials_block],
        samples_cond=task_constants.SAMPLES_COND,
        neuron_cond=neuron_cond,
    )
    sp_samples = data.sp_samples[trials_block]
    neuron_type = data.clustersgroup
    code_samples = data.code_samples[trials_block]
    code_numbers = data.code_numbers[trials_block]
    sp_samples = data.sp_samples[trials_block]

    fix_t = 200
    dur_v = 250
    st_m = 450  # 800
    end_m = 900
    st_t = 900
    end_t = 900 + 450
    min_trials = 3
    n_spikes_sec = 5
    p_threshold = 0.05

    shifts_on = code_samples[:, 4]
    align_event = task_constants.EVENTS_B1["sample_on"]
    if np.sum(code_numbers[:, 4] - align_event) != 0:
        raise KeyError
    shifts_on = shifts_on[:, np.newaxis]
    shifts_on = np.where(np.isnan(shifts_on), 0, shifts_on)
    shifts_test = code_samples[:, 6]
    align_event = task_constants.EVENTS_B1["test_on_1"]
    if np.sum(code_numbers[:, 6] - align_event) != 0:
        raise KeyError
    shifts_test = shifts_test[:, np.newaxis]
    shifts_test = np.where(np.isnan(shifts_test), 0, shifts_test)
    # shift data
    sp_shift_on = SpikeData.indep_roll(
        sp_samples, -(shifts_on - fix_t).astype(int), axis=2
    )[:, :, : end_m + fix_t]
    sp_shift_test = SpikeData.indep_roll(
        sp_samples, -(shifts_test).astype(int), axis=2
    )[:, :, :450]
    sp_shift = np.concatenate([sp_shift_on, sp_shift_test], axis=2)

    neurons_info = get_neurons_info(
        sp_shift,
        fix_t,
        neuron_type,
        task,
        dur_v=dur_v,
        st_m=st_m,
        end_m=end_m,
        st_t=st_t,
        end_t=end_t,
        min_trials=min_trials,
        n_spikes_sec=n_spikes_sec,
        clusterdepth=data.clusterdepth,
        date=s_path[-1][:19],
    )
    neurons_info = neurons_info[
        (neurons_info["p"] < p_threshold) & (neurons_info["larger"] == True)
    ]  # responding neurons

    neurons_info = neurons_info[neurons_info["in_out"] == neurons_info["true_in_out"]]
    neurons_info = neurons_info[
        (np.logical_and(neurons_info["p_v"] < 0.05, neurons_info["v_larger"] == True))
        | (np.logical_and(neurons_info["p_m"] < 0.05, neurons_info["m_larger"] == True))
    ]

    neurons_info = pd.DataFrame(neurons_info)

    neurons_info.to_csv(
        "/".join([output_dir] + [ss_path + "_vm_b1.csv"]),
        index=False,
    )
    del neurons_info
    gc.collect()
    logging.info("-- end --")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file_path", help="Path to the continuous file (.dat)", type=Path
    )
    parser.add_argument(
        "bhv_path", help="Path to the continuous file (.dat)", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument(
        "--e_align",
        "-e",
        default="target_on",
        help="Event to aligne the spikes",
        type=str,
    )
    parser.add_argument(
        "--t_before",
        "-t",
        default=500,
        help="Time before e_aligne",
        type=int,
    )
    args = parser.parse_args()
    try:
        main(
            args.file_path, args.bhv_path, args.output_dir, args.e_align, args.t_before
        )
    except FileExistsError:
        logging.error("filepath does not exist")
