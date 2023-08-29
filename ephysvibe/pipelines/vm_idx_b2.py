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
from ..analysis import circular_stats
import warnings
from matplotlib import pyplot as plt
from ..structures.spike_data import SpikeData
from typing import Dict
from collections import defaultdict
import pandas as pd
import gc

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(filepath: Path, output_dir: Path, e_align: str, t_before: int):
    s_path = os.path.normpath(filepath).split(os.sep)
    ss_path = s_path[-1][:-3]
    output_dir = "/".join([os.path.normpath(output_dir)] + [s_path[-3]] + ["b2"])
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
                trials.append(int(i_trial))
        target_codes[key] = {
            "code_idx": code_idx,
            "trial_idx": trials,
            "position_codes": position_codes[key][1],
            "angle_codes": position_codes[key][2][0],
        }

    # Search neurons responding to the task
    sp_samples = data.sp_samples
    neuron_type = data.clustersgroup
    code_samples = data.code_samples
    code_numbers = data.code_numbers
    sp_samples = data.sp_samples

    ipsi = np.array(["124", "123", "122", "121"])
    contra = np.array(["120", "127", "126", "125"])

    fix_t = 200
    dur_v = 200
    st_m = 700  # 800
    end_m = 1100
    p_threshold = 0.05
    min_trials = 3

    align_event = task_constants.EVENTS_B2["target_on"]
    shifts = code_samples[:, 3]
    shifts = shifts[:, np.newaxis]
    sp_shift = SpikeData.indep_roll(sp_samples, -(shifts - fix_t).astype(int), axis=2)[
        :, :, : 1100 + fix_t
    ]
    neurons_info = plot_raster.get_neurons_info(
        sp_shift,
        fix_t,
        neuron_type,
        target_codes,
        ipsi,
        contra,
        dur_v=dur_v,
        st_m=st_m,
        end_m=end_m,
        neuron_idx=None,
        min_trials=min_trials,
    )
    neurons_info = neurons_info[
        (neurons_info["p"] < 0.05) & (neurons_info["larger"] == True)
    ]  # responding neurons
    if neurons_info.empty:
        logging.warning("No significant units")
        raise ValueError

    # Search neurons RF
    shifts = code_samples[:, 3]
    shifts = shifts[:, np.newaxis]
    sp_shift = SpikeData.indep_roll(sp_samples, -(shifts).astype(int), axis=2)[
        :, :, :1100
    ]

    rf_test = plot_raster.get_rf(
        neurons_info,
        sp_shift,
        ipsi,
        contra,
        target_codes,
        dur_v,
        st_m,
        end_m,
        min_trials=min_trials,
    )

    th_rf = rf_test[
        np.logical_and(rf_test["p"] < p_threshold, rf_test["larger"] == True)
    ]  # results below threshold
    if th_rf.empty:
        logging.error("No significant units")
        raise ValueError

    shifts = code_samples[:, 3]
    shifts = shifts[:, np.newaxis]
    sp_shift = SpikeData.indep_roll(sp_samples, -(shifts).astype(int), axis=2)[
        :, :, :1100
    ]
    sp_shift_bl = SpikeData.indep_roll(
        sp_samples, -(shifts - fix_t).astype(int), axis=2
    )[:, :, : 1100 + fix_t]
    test_vm = plot_raster.get_vm_index(
        th_rf,
        target_codes,
        sp_shift,
        sp_shift_bl,
        align_event,
        fix_t,
        dur_v,
        st_m,
        end_m,
        min_trials,
    )
    test_vm = test_vm[
        (np.logical_and(test_vm["p_v"] < 0.05, test_vm["v_larger"] == True))
        | (np.logical_and(test_vm["p_p"] < 0.05, test_vm["p_larger"] == True))
    ]
    if test_vm.empty:
        logging.error("No significant units")
        raise ValueError

    no_dup_vm = test_vm  #
    # ------- plot ------------
    color = {
        "visual": ["salmon", "darkred", "--"],
        "anticipation": ["royalblue", "navy", ":"],
    }
    # kernel parameters
    t_before = 200
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
            no_dup_vm,
        )
        # ------------------ plot spider in the center
        (
            fr_max_visual,
            fr_max_motor,
            fr_angle,
            fr_max_codes,
            v_significant,
            m_significant,
        ) = plot_raster.get_max_fr(
            target_codes,
            sp_samples,
            code_samples,
            code_numbers,
            i_n,
            kernel,
            win_size,
            dur_v,
            e_code_align,
            test_vm,
            fs_ds,
        )
        codes_sig = np.logical_or(m_significant, v_significant)
        fr_code_max = max(fr_max_codes)
        vm_index, vm_index_v_fr, vm_index_p_fr = np.nan, np.nan, np.nan
        code_vm_index, code_index_v_fr, code_index_p_fr = np.nan, np.nan, np.nan
        vm_index_v_max, vm_index_p_max, code_v_max, code_p_max = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

        if np.any(~np.isnan(fr_max_codes[codes_sig])):
            neu_test_vm = test_vm[test_vm["array_position"] == i_n]

            # select vm index of the code with the max fr
            idx_max_all = np.nanargmax(fr_max_codes[codes_sig])
            ang_max_all = fr_angle[codes_sig][idx_max_all]
            idx_code = np.where(pd.DataFrame(target_codes).iloc[3] == ang_max_all)[0][0]
            code = list(target_codes.keys())[idx_code]
            if code in neu_test_vm["code"].values:
                vm_index = neu_test_vm[neu_test_vm["code"] == code]["vm_index"].values[
                    0
                ]
                code_vm_index = code
            if np.any(~np.isnan(fr_max_visual[v_significant])):
                # select vm index max visual fr
                idx_max_all = np.nanargmax(fr_max_visual[v_significant])
                ang_max_all = fr_angle[v_significant][idx_max_all]
                idx_code = np.where(pd.DataFrame(target_codes).iloc[3] == ang_max_all)[
                    0
                ][0]
                code = list(target_codes.keys())[idx_code]
                if code in neu_test_vm["code"].values:
                    vm_index_v_fr = neu_test_vm[neu_test_vm["code"] == code][
                        "vm_index"
                    ].values[0]
                    code_index_v_fr = code
                # select max visual vm index
                ang_max_all = fr_angle[v_significant]
                idx_code = np.where(
                    np.isin(pd.DataFrame(target_codes).iloc[3].values, ang_max_all)
                )[0]
                code = np.array(list(target_codes.keys()))[idx_code]
                vm_index_v_max = neu_test_vm[np.isin(neu_test_vm["code"].values, code)]
                idx_max = np.argmax(np.abs(vm_index_v_max["vm_index"]))
                code_v_max = vm_index_v_max["code"].iloc[idx_max]
                vm_index_v_max = vm_index_v_max["vm_index"].iloc[idx_max]
            if np.any(~np.isnan(fr_max_motor[m_significant])):
                # select vm index max anticipation fr
                idx_max_all = np.nanargmax(fr_max_motor[m_significant])
                ang_max_all = fr_angle[m_significant][idx_max_all]
                idx_code = np.where(pd.DataFrame(target_codes).iloc[3] == ang_max_all)[
                    0
                ][0]
                code = list(target_codes.keys())[idx_code]
                if code in neu_test_vm["code"].values:
                    vm_index_p_fr = neu_test_vm[neu_test_vm["code"] == code][
                        "vm_index"
                    ].values[0]
                    code_index_p_fr = code
                # select max visual vm index
                ang_max_all = fr_angle[m_significant]
                idx_code = np.where(
                    np.isin(pd.DataFrame(target_codes).iloc[3].values, ang_max_all)
                )[0]
                code = np.array(list(target_codes.keys()))[idx_code]
                vm_index_p_max = neu_test_vm[np.isin(neu_test_vm["code"].values, code)]
                idx_max = np.argmax(np.abs(vm_index_p_max["vm_index"]))
                code_p_max = vm_index_p_max["code"].iloc[idx_max]
                vm_index_p_max = vm_index_p_max["vm_index"].iloc[idx_max]

        ax = plt.subplot2grid((3, 3), (1, 1), polar=True)
        fr_angle_rad = (np.array(fr_angle) * 2 * np.pi) / 360
        fr_angle_rad = np.concatenate([fr_angle_rad, fr_angle_rad[:1]])

        for fr_max, event, significant in zip(
            [fr_max_visual, fr_max_motor],
            ["visual", "anticipation"],
            [v_significant, m_significant],
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

            else:
                rad, ang = np.nan, np.nan
                fr_max_n = np.nan
                ang_max_n = np.nan

            # compute mean vector of all significant positions/codes
            if np.any(codes_sig):
                rad_all, ang_all = circular_stats.mean_vector(
                    radius=fr_max_codes[codes_sig] / fr_code_max,
                    angle=fr_angle_rad[:-1][codes_sig],
                )
            else:
                rad_all, ang_all = np.nan, np.nan
            # plot max fr
            norm_fr_max = np.concatenate([norm_fr_max, norm_fr_max[:1]])
            ax.set_rlabel_position(90)
            plt.yticks(
                [0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], color="grey", size=7
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
            rf_coordinates["vm_index_frmax"] += [vm_index]
            rf_coordinates["code_vm_index"] += [code_vm_index]
            rf_coordinates["vm_index_v_fr"] += [vm_index_v_fr]
            rf_coordinates["code_index_v_fr"] += [code_index_v_fr]
            rf_coordinates["vm_index_p_fr"] += [vm_index_p_fr]
            rf_coordinates["code_index_p_fr"] += [code_index_p_fr]
            rf_coordinates["vm_index_v_max"] += [vm_index_v_max]
            rf_coordinates["code_v_max"] += [code_v_max]
            rf_coordinates["vm_index_p_max"] += [vm_index_p_max]
            rf_coordinates["code_p_max"] += [code_p_max]
        ## ------------------ end spider
        avg_events = [-200, 0, 100, 1100, 1500]
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
        fig.suptitle("%s: %s %d" % (s_path[-3], cluster, i_cluster), x=0)
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
                [output_dir] + [ss_path + "_" + cluster + "_" + str(i_cluster) + ".jpg"]
            ),
            bbox_inches="tight",
        )

    rf_coordinates = pd.DataFrame(rf_coordinates)
    rf_coordinates.to_csv(
        "/".join([output_dir] + [ss_path + "_vm_b2.csv"]),
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
