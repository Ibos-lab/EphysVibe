# plot spiking activity task b1
import os
import argparse
from pathlib import Path
import logging
import numpy as np
from ..trials.spikes import firing_rate, sp_constants
from ..spike_sorting import config
from ..task import task_constants
from ..trials.spikes import plot_raster
from ..analysis import circular_stats, raw_ch
import warnings
from matplotlib import pyplot as plt
from ..structures.spike_data import SpikeData
from typing import Dict
from collections import defaultdict
import pandas as pd
import gc
from scipy import signal, stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def get_neurons_info(
    sp_target_on: np.ndarray,
    sp_baseline: np.ndarray,
    neuron_type: np.ndarray,
    target_codes: Dict,
    ipsi: np.ndarray,
    contra: np.ndarray,
    neuron_idx: np.ndarray = None,
) -> pd.DataFrame:
    tr_d = sp_target_on.shape[2]
    window = 80
    base_d = sp_baseline.shape[2]
    if neuron_idx is None:
        neuron_idx = np.arange(0, len(neuron_type))
    codes = target_codes.keys()
    neurons_info: Dict[str, list] = defaultdict(list)
    i_good, i_mua, n_type = 0, 0, 0
    for i_neuron, type_neuron in zip(neuron_idx, neuron_type):
        # check and count type of unit
        if type_neuron == "good":
            i_good += 1
            n_type = i_good
        elif type_neuron == "mua":
            i_mua += 1
            n_type = i_mua
        for code in codes:  # iterate by code'
            psth_trial = None
            psth_bl = None
            p = None
            trial_idx = np.array(target_codes[code]["trial_idx"])
            # select trials with at least 5 sp/sec
            trial_idx = trial_idx[
                (sp_target_on[trial_idx, i_neuron].sum(axis=1) > 5 * (tr_d) / 1000)
            ]
            n_tr = len(trial_idx)
            if n_tr > 2:

                psth_trial = sp_target_on[trial_idx, i_neuron].mean(axis=0)
                psth_trial = (
                    raw_ch.rolling_window(x=psth_trial, window=window, step=1)
                ).sum(axis=1) * (1000 / window)
                psth_bl = sp_baseline[trial_idx, i_neuron].mean(axis=0)
                psth_bl = (raw_ch.rolling_window(x=psth_bl, window=window, step=1)).sum(
                    axis=1
                ) * (1000 / window)

                p = stats.ttest_ind(psth_trial, psth_bl, equal_var=True)[1]
                # psth_trial = ((sp_target_on[trial_idx,i_neuron].sum(axis=0)/n_tr)*tr_d)
                # ax.plot(psth_trial,'--')
                # psth_bl = ((sp_baseline[trial_idx,i_neuron].sum(axis=0)/n_tr)*base_d)
            if code in ipsi:
                laterality = "ipsi"
            else:
                laterality = "contra"
            neurons_info["code"] += [code]
            neurons_info["max_psth_trial"] += [np.max(psth_trial)]
            neurons_info["max_psth_bl"] += [np.max(psth_bl)]
            neurons_info["p"] += [p]
            neurons_info["laterality"] += [laterality]
            neurons_info["cluster"] += [n_type]
            neurons_info["group"] += [type_neuron]
            neurons_info["array_position"] += [i_neuron]
    neurons_info = pd.DataFrame(neurons_info)
    return neurons_info


def get_vm_index(
    neu_rf,
    target_codes,
    code_samples,
    code_numbers,
    sp_visual,
    sp_motor,
    sp_target_on,
    window=10,
):
    test_vm: Dict[str, list] = defaultdict(list)

    for _, row in neu_rf.iterrows():
        i_neuron = row["array_position"]
        code = row["code"]
        # event = row['event']
        trial_idx = np.array(
            target_codes[code]["trial_idx"]
        )  # select trials with the same stimulus
        # select trials
        # shift = code_samples[trial_idx, np.where(code_numbers[trial_idx] == align_event)[1]] # moment when the target_on ocurrs in each trial

        trial_idx = trial_idx[
            sp_target_on[trial_idx, i_neuron, 200:].sum(axis=1)
            > 5 * (sp_target_on.shape[2]) / 1000
        ]  # 5sp/1000 ms
        n_tr = len(trial_idx)
        if n_tr > 2:
            m_psth = sp_motor[trial_idx, i_neuron].mean(axis=0)
            m_psth = (raw_ch.rolling_window(x=m_psth, window=window, step=1)).sum(
                axis=1
            ) * (1000 / window)
            v_psth = sp_visual[trial_idx, i_neuron].mean(axis=0)
            v_psth = (raw_ch.rolling_window(x=v_psth, window=window, step=1)).sum(
                axis=1
            ) * (1000 / window)
            bl_psth = sp_target_on[trial_idx, i_neuron, :200].mean(axis=0)
            bl_psth = (raw_ch.rolling_window(x=bl_psth, window=window, step=1)).sum(
                axis=1
            ) * (1000 / window)

            m_mean = m_psth.mean()
            v_mean = v_psth.mean()
            vm_index = (m_mean - v_mean) / (v_mean + m_mean)
            bl_mean = bl_psth.mean()
            m_p = None
            v_p = None
            if m_mean > 2 * bl_mean:
                m_p = stats.ttest_ind(m_psth, bl_psth, equal_var=True)[1]
            if v_mean > 2 * bl_mean:
                v_p = stats.ttest_ind(v_psth, bl_psth, equal_var=True)[1]
            # save results
            test_vm["code"] += [code]
            test_vm["array_position"] += [i_neuron]
            test_vm["vm_index"] += [vm_index]
            test_vm["v_mean"] += [v_mean]
            test_vm["v_p"] += [v_p]
            test_vm["m_mean"] += [m_mean]
            test_vm["m_p"] += [m_p]
            test_vm["bl_mean"] += [bl_mean]
            test_vm["cluster"] += [row["cluster"]]
            test_vm["group"] += [row["group"]]
    test_vm = pd.DataFrame(test_vm)
    return test_vm


def get_max_fr(
    target_codes,
    sp_samples,
    code_samples,
    code_numbers,
    i_n,
    dur_v,
    st_m,
    end_m,
    test_vm,
    trial_idx,
):
    fr_max_visual, fr_max_motor, fr_angle, fr_max_trial = [], [], [], []
    v_significant, m_significant, vm_significant = [], [], []
    window = 100
    target_on = task_constants.EVENTS_B2["target_on"]
    idx_ev = np.where(code_numbers[trial_idx] == target_on)[1]
    idx_ev = idx_ev[0] if np.sum(idx_ev == idx_ev[0]) else False
    shifts = code_samples[:, idx_ev]
    shifts = shifts[:, np.newaxis]
    sp_target_on = SpikeData.indep_roll(sp_samples, -(shifts).astype(int), axis=2)[
        :, :, :end_m
    ]
    sp_visual = sp_target_on[:, :, :dur_v]
    sp_motor = sp_target_on[:, :, st_m:end_m]
    for code in target_codes.keys():

        target_t_idx = np.array(
            target_codes[code]["trial_idx"]
        )  # select trials with the same stimulus
        target_t_idx = target_t_idx[
            sp_target_on[target_t_idx, i_n].sum(axis=1)
            > 5 * (sp_target_on.shape[2]) / 1000
        ]  # 5sp/1000 ms
        n_tr = len(target_t_idx)
        if n_tr > 2:
            m_psth = sp_motor[target_t_idx, i_n].mean(axis=0)
            m_psth = (raw_ch.rolling_window(x=m_psth, window=window, step=1)).sum(
                axis=1
            ) * (1000 / window)
            v_psth = sp_visual[target_t_idx, i_n].mean(axis=0)
            v_psth = (raw_ch.rolling_window(x=v_psth, window=window, step=1)).sum(
                axis=1
            ) * (1000 / window)
            tr_psth = sp_target_on[target_t_idx, i_n].mean(axis=0)
            tr_psth = (raw_ch.rolling_window(x=tr_psth, window=window, step=1)).sum(
                axis=1
            ) * (1000 / window)
        else:
            m_psth = [0]
            v_psth = [0]
            tr_psth = [0]
        fr_max_visual.append(np.nanmax(v_psth))
        fr_angle.append(target_codes[code]["angle_codes"])
        fr_max_motor.append(np.nanmax(m_psth))
        fr_max_trial.append(np.nanmax(tr_psth))
        v_sig = False
        m_sig = False
        vm_sig = False
        if (
            code
            in test_vm[
                (test_vm["array_position"] == i_n) & (test_vm["vm_index"] < -0.4)
            ]["code"].values
        ):
            v_sig = True
        elif (
            code
            in test_vm[
                (test_vm["array_position"] == i_n) & (test_vm["vm_index"] > 0.4)
            ]["code"].values
        ):
            m_sig = True
        elif (
            code
            in test_vm[
                (test_vm["array_position"] == i_n)
                & (test_vm["vm_index"] < 0.4)
                & (test_vm["vm_index"] > -0.4)
            ]["code"].values
        ):
            vm_sig = True
            m_sig = True
            v_sig = True

        v_significant.append(v_sig)
        m_significant.append(m_sig)
        vm_significant.append(vm_sig)
    return (
        np.array(fr_max_visual),
        np.array(fr_max_motor),
        np.array(fr_angle),
        np.array(fr_max_trial),
        np.array(v_significant),
        np.array(m_significant),
        np.array(vm_significant),
    )


def main(filepath: Path, output_dir: Path, e_align: str, t_before: int):
    s_path = os.path.normpath(filepath).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-2]])
    # check if output dir exist, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # check if filepath exist
    if not os.path.exists(filepath):
        raise FileExistsError
    logging.info("-- Start --")
    data = SpikeData.from_python_hdf5(filepath)
    # Select trials and create task frame
    trial_idx = np.where(np.logical_and(data.trial_error == 0, data.block == 2))[0]
    logging.info("Number of clusters: %d" % len(data.clustersgroup))

    # Define target codes
    position_codes = {
        # code: [[MonkeyLogic axis], [plot axis]]
        "127": [[10, 0], [1, 2], [0]],
        "126": [[7, 7], [0, 2], [45]],
        "125": [[0, 10], [0, 1], [90]],
        "124": [[-7, 7], [0, 0], [135]],
        "123": [[-10, 0], [1, 0], [180]],
        "122": [[-7, -7], [2, 0], [225]],
        "121": [[0, -10], [2, 1], [270]],
        "120": [[7, -7], [2, 2], [315]],
    }
    # create dict with the trials having each code
    target_codes = {}
    for key in position_codes.keys():
        trials = []
        code_idx = []
        for i_trial, code in zip(trial_idx, data.code_numbers[trial_idx]):
            idx = np.where(int(key) == code)[0]
            if len(idx) != 0:
                code_idx.append(idx[0])
                trials.append(i_trial)
        target_codes[key] = {
            "code_idx": code_idx,
            "trial_idx": trials,
            "position_codes": position_codes[key][1],
            "angle_codes": position_codes[key][2][0],
        }
    neuron_type = data.clustersgroup
    ipsi = np.array(["124", "123", "122", "121"])
    contra = np.array(["120", "127", "126", "125"])
    neurons_info = plot_raster.get_neurons_info(
        neuron_type, target_codes, ipsi, contra, neuron_idx=None
    )
    # Search neurons responding to the task
    sp_samples = data.sp_samples
    neuron_type = data.clustersgroup
    code_samples = data.code_samples
    code_numbers = data.code_numbers
    sp_samples = data.sp_samples
    fix_t = 200
    len_trial = 1100
    dur_v = 200
    st_m = 600
    end_m = 1100
    target_on = task_constants.EVENTS_B2["target_on"]
    idx_ev = np.where(code_numbers[trial_idx] == target_on)[1]
    idx_ev = idx_ev[0] if np.sum(idx_ev == idx_ev[0]) else False
    shifts = code_samples[:, idx_ev]
    shifts = shifts[:, np.newaxis]
    sp_target_on = SpikeData.indep_roll(sp_samples, -(shifts).astype(int), axis=2)[
        :, :, :len_trial
    ]
    sp_baseline = SpikeData.indep_roll(
        sp_samples, -(shifts - fix_t).astype(int), axis=2
    )[:, :, :fix_t]
    neuron_type = data.clustersgroup
    ipsi = np.array(["124", "123", "122", "121"])
    contra = np.array(["120", "127", "126", "125"])
    neurons_info = get_neurons_info(
        sp_target_on,
        sp_baseline,
        neuron_type,
        target_codes,
        ipsi,
        contra,
        neuron_idx=None,
    )
    neurons_info = neurons_info[~neurons_info["max_psth_trial"].isna()]
    # neurons responding to the task
    resp_neurons = neurons_info[
        neurons_info["max_psth_trial"] > 2 * neurons_info["max_psth_bl"]
    ].reset_index(drop=True)
    # check where the rf is at least half of the max fr of the peak position
    idx_max = (
        resp_neurons.groupby(["array_position"])["max_psth_trial"].transform(max)
        == resp_neurons["max_psth_trial"]
    )
    max_resp_neurons = resp_neurons[idx_max]
    neu_rf = []
    for i_resp in resp_neurons["array_position"].unique():
        i_n_max = max_resp_neurons[max_resp_neurons["array_position"] == i_resp][
            "max_psth_trial"
        ].values[0]
        i_n = resp_neurons[resp_neurons["array_position"] == i_resp]
        i_n = i_n[i_n["max_psth_trial"] >= i_n_max / 2]
        neu_rf.append(i_n)
    neu_rf = pd.concat(neu_rf)
    # vm index
    target_on = task_constants.EVENTS_B2["target_on"]
    idx_ev = np.where(code_numbers[trial_idx] == target_on)[1]
    idx_ev = idx_ev[0] if np.sum(idx_ev == idx_ev[0]) else False
    shifts = code_samples[:, idx_ev]
    shifts = shifts[:, np.newaxis]
    sp_target_on = SpikeData.indep_roll(
        sp_samples, -(shifts - fix_t).astype(int), axis=2
    )
    sp_visual = sp_target_on[:, :, fix_t : fix_t + dur_v]
    sp_motor = sp_target_on[:, :, fix_t + st_m : fix_t + end_m]
    test_vm = get_vm_index(
        neu_rf,
        target_codes,
        code_samples,
        code_numbers,
        sp_visual,
        sp_motor,
        sp_target_on[:, :, : end_m + fix_t],
    )
    # ------- plot ------------
    color = {"visual": ["salmon", "darkred", "--"], "motor": ["royalblue", "navy", ":"]}
    warnings.filterwarnings("ignore")
    # kernel parameters
    t_before = 500
    fs_ds = config.FS / config.DOWNSAMPLE
    kernel = firing_rate.define_kernel(
        sp_constants.W_SIZE, sp_constants.W_STD, fs=fs_ds
    )
    win_size = int(sp_constants.W_SIZE * fs_ds)
    code_samples = data.code_samples
    code_numbers = data.code_numbers
    sp_samples = data.sp_samples
    e_code_align = task_constants.EVENTS_B2["target_on"]
    # select only individual neurons
    rf_coordinates: Dict[str, list] = defaultdict(list)
    i_neuron, i_mua = 1, 1
    for i_n, cluster in enumerate(data.clustersgroup):  # iterate by units
        if cluster == "good":
            i_cluster = i_neuron
            i_neuron += 1
            cluster = "neuron"
        else:
            i_cluster = i_mua
            i_mua += 1
        fig, _ = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)  # define figure
        (
            all_ax,
            all_ax2,
            all_max_conv,
            max_num_trials,
        ) = plot_raster.plot_activity_location(
            target_codes,
            code_samples,
            code_numbers,
            sp_samples,
            i_n,
            e_code_align,
            t_before,
            fs_ds,
            kernel,
            test_vm,
        )
        # ------------------ plot spider in the center
        (
            fr_max_visual,
            fr_max_motor,
            fr_angle,
            fr_max_trial,
            v_significant,
            m_significant,
            vm_significant,
        ) = get_max_fr(
            target_codes,
            sp_samples,
            code_samples,
            code_numbers,
            i_n,
            dur_v,
            st_m,
            end_m,
            test_vm,
            trial_idx,
        )
        codes_sig = np.logical_or(
            m_significant, np.logical_or(v_significant, vm_significant)
        )
        fr_code_max = max(fr_max_trial)
        vm_index = np.nan

        ax = plt.subplot2grid((3, 3), (1, 1), polar=True)
        fr_angle_rad = (np.array(fr_angle) * 2 * np.pi) / 360
        fr_angle_rad = np.concatenate([fr_angle_rad, fr_angle_rad[:1]])
        for fr_max, event, significant in zip(
            [fr_max_visual, fr_max_motor, fr_max_trial],
            ["visual", "motor", "vm"],
            [v_significant, m_significant, vm_significant],
        ):
            norm_fr_max = np.array(fr_max) / fr_code_max
            # compute mean vector only visual or motor
            if np.any(significant):
                rad, ang = circular_stats.mean_vector(
                    radius=norm_fr_max[significant],
                    angle=fr_angle_rad[:-1][significant],
                )
                idx_max = np.nanargmax(fr_max)
                fr_max_n = fr_max[idx_max]
                ang_max_n = fr_angle[idx_max]
                idx_code = np.where(pd.DataFrame(target_codes).iloc[3] == ang_max_n)[0][
                    0
                ]
                code = list(target_codes.keys())[idx_code]
                neu_test_vm = test_vm[test_vm["array_position"] == i_n]
                if code in neu_test_vm["code"].values:
                    vm_index = neu_test_vm[neu_test_vm["code"] == code][
                        "vm_index"
                    ].values[0]
            else:
                rad, ang = np.nan, np.nan
                fr_max_n = np.nan
                ang_max_n = np.nan

            # compute mean vector of all significant positions/codes
            if np.any(codes_sig):
                rad_all, ang_all = circular_stats.mean_vector(
                    radius=fr_max_trial[codes_sig] / fr_code_max,
                    angle=fr_angle_rad[:-1][codes_sig],
                )
            else:
                rad_all, ang_all = np.nan, np.nan
            # plot max fr
            if event != "vm":
                norm_fr_max = np.concatenate([norm_fr_max, norm_fr_max[:1]])
                ax.set_rlabel_position(90)
                plt.yticks(
                    [0.25, 0.5, 0.75, 1],
                    ["0.25", "0.5", "0.75", "1"],
                    color="grey",
                    size=7,
                )
                plt.ylim(0, 1)
                plt.xticks(fr_angle_rad[:-1], target_codes.keys())
                ax.plot(
                    fr_angle_rad,
                    norm_fr_max,
                    linewidth=1,
                    linestyle="solid",
                    color=color[event][0],
                    label=event,
                )
                ax.fill(fr_angle_rad, norm_fr_max, alpha=0.1, color=color[event][0])
                # plot mean vector
                ax.plot(
                    [0, ang],
                    [0, rad],
                    linewidth=1,
                    linestyle=color[event][2],
                    color=color[event][1],
                )
                plt.legend(loc="upper right", bbox_to_anchor=(0, 0), prop={"size": 7})
            # add results to df
            rf_coordinates["array_position"] += [i_n]
            rf_coordinates["neuron_type"] += [cluster]
            rf_coordinates["i_neuron"] += [i_cluster]
            rf_coordinates["event"] += [event]
            rf_coordinates["rad"] += [rad]
            rf_coordinates["ang"] += [ang]
            rf_coordinates["fr_max"] += [fr_max_n]
            rf_coordinates["ang_max"] += [ang_max_n]
            rf_coordinates["rad_all"] += [rad_all]
            rf_coordinates["ang_all"] += [ang_all]
            rf_coordinates["depth"] += [data.clusterdepth[i_n]]
            rf_coordinates["date"] += [s_path[-1][:19]]
            rf_coordinates["vm_index"] += [vm_index]
        ## ------------------ end spider

        avg_events = [-500, 0, 100, 1100, 1500]
        # num_trials = sp_samples.shape[0]
        for ax, ax2 in zip(all_ax, all_ax2):
            for ev in avg_events:
                ax.vlines(
                    ev,
                    0,
                    all_max_conv + max_num_trials * 3,
                    color="k",
                    linestyles="dashed",
                )  # target_on
            ax.set_ylim(0, all_max_conv + max_num_trials * 3)
            ax.set_yticks(np.arange(0, all_max_conv, 10))
            ax2.set_ylim(-all_max_conv, max_num_trials)
            ax2.set_yticks(np.arange(-all_max_conv, max_num_trials * 3, 10))
            ax.set(xlabel="Time (s)", ylabel="Average firing rate")
            ax2.set(xlabel="Time (s)", ylabel="trials")
            plt.setp(ax2.get_yticklabels(), visible=False)
        fig.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
        fig.suptitle("%s: %s %d" % (s_path[-2], cluster, i_cluster), x=0)
        fig.text(
            0,
            0,
            s="Depth: %d" % data.clusterdepth[i_n],
            horizontalalignment="center",
            verticalalignment="center",
        )
        # ----- end plot ----
        logging.info("Saving figure, %s: %d" % (cluster, i_cluster))
        plt.savefig(
            "/".join(
                [os.path.normpath(output_dir)]
                + [ss_path + "_" + cluster + "_" + str(i_cluster) + ".jpg"]
            ),
            bbox_inches="tight",
        )

    rf_coordinates = pd.DataFrame(rf_coordinates)

    rf_coordinates.to_csv(
        "/".join([os.path.normpath(output_dir)] + [ss_path + "_rf_coordinates.csv"]),
        index=False,
    )
    del rf_coordinates
    plt.close(fig)
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
        main(args.file_path, args.output_dir, args.e_align, args.t_before)
    except FileExistsError:
        logging.error("filepath does not exist")
