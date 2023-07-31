import argparse
from pathlib import Path
import logging
import os
import json
from typing import List, Tuple, Dict
import numpy as np
from ...spike_sorting import utils_oe, config, data_structure, pre_treat_oe
import glob
from ...structures.bhv_data import BhvData
from .. import pipe_config
from collections import defaultdict
from ...structures.spike_data import SpikeData

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def define_paths(continuous_path: Path) -> Tuple[List, str, str, str, str]:
    """Define paths using the input path.

    Args:
        continuous_path (Path): path to the continuous file (.dat) from OE

    Returns:
        Tuple[List, str, str, str, str]:
            - s_path (List): list containing the splited continuous path
            - directory (str): path to the directory (to the date/time of the session)
            - time_path (str): path to the sample_numbers file
            - event_path (str): path to the events folder
            - areas_path (str): path to the json file containing info about the channels in each area
    """
    # define paths
    s_path = os.path.normpath(continuous_path).split(os.sep)
    directory = "/".join(s_path[:-3])
    time_path = "/".join(s_path[:-1] + ["sample_numbers.npy"])
    event_path = "/".join(
        s_path[:-3] + ["events"] + ["Acquisition_Board-100.Rhythm Data"] + ["TTL"]
    )
    # check if paths exist
    if not os.path.exists(directory):
        logging.error("directory: %s does not exist" % directory)
        raise FileExistsError
    if not os.path.exists(time_path):
        logging.error("time_path: %s does not exist" % time_path)
        raise FileExistsError
    if not os.path.exists(event_path):
        logging.error("event_path: %s does not exist" % event_path)
        raise FileExistsError

    return (
        s_path,
        directory,
        time_path,
        event_path,
    )


def main(
    ks_path: Path,
    output_dir: Path,
    areas: list,
) -> None:
    """Compute trials.

    Args:
        continuous_path (Path):  path to the continuous file (.dat) from OE.
        output_dir (Path): output directory.
        areas (list): list containing the areas to which to compute the trials data.
        start_ch (list): list containing the index of the first channel for each area.
        n_ch (list): list containing the number of channels for each area.
    """
    if not os.path.exists(ks_path):
        logging.error("ks_path %s does not exist" % ks_path)
        raise FileExistsError
    logging.info("-- Start --")
    # define paths
    s_path = os.path.normpath(ks_path).split(os.sep)
    time_path = "/".join(s_path + ["continuous/Acquisition_Board-100.Rhythm Data/"])
    bhv_path = "/".join(s_path[:-2] + ["*/*/*_bhv.h5"])
    bhv_path = glob.glob(bhv_path, recursive=True)
    if len(bhv_path) == 0:
        logging.info("Bhv file not found")
        raise ValueError

    # Select info about the recording from the path
    n_exp = s_path[-2][-1]
    n_record = s_path[-1][-1]
    subject = s_path[-5]
    date_time = s_path[-4]
    # check n_areas and n_channels
    if areas == None:
        areas_ch = ["lip", "v4", "pfc"]

    # else:
    #     if n_ch == None or start_ch == None:
    #         raise KeyError("n_ch or start_ch = None")
    #     areas_ch: Dict[str, list] = defaultdict(list)
    # for n, n_area in enumerate(areas):
    #     areas_ch[n_area] = [start_ch[n], n_ch[n]]

    # load bhv data

    logging.info("Loading bhv data")
    bhv = BhvData.from_python_hdf5(bhv_path[0])
    # load timestamps and events
    logging.info("Loading continuous/sample_numbers data")
    c_samples = np.load(time_path + "sample_numbers.npy")
    # events = utils_oe.load_event_files(event_path)
    # shape_0 = len(c_samples)
    # (
    #     full_word,
    #     real_strobes,
    #     start_trials,
    #     end_trials,
    #     ds_samples,
    #     idx_start_time,
    #     eyes_ds,
    #     bhv,
    # ) = pre_treat_oe.pre_treat_oe(
    #     events=events,
    #     bhv=bhv,
    #     c_samples=c_samples,
    # )
    logging.info("Selecting OE samples")
    # ds_samples, idx_start_time = utils_oe.select_samples(
    #     c_samples=c_samples,
    #     e_samples=events["samples"],
    #     fs=config.FS,
    #     t_before_event=config.T_EVENT,
    #     downsample=config.DOWNSAMPLE,
    # )
    # check if eyes
    # start_ch, n_eyes = areas_ch.pop("eyes", False)
    # eyes_ds = np.array([])

    start_trials = bhv.start_trials
    end_trials = bhv.end_trials
    # to ms
    # ds_samples = np.floor(ds_samples / config.DOWNSAMPLE).astype(int)
    # start_trials = np.floor(start_trials / config.DOWNSAMPLE).astype(int)
    # end_trials = np.floor(end_trials / config.DOWNSAMPLE).astype(int)
    # real_strobes = np.floor(real_strobes / config.DOWNSAMPLE).astype(int)
    # Iterate by nodes/areas
    for area in areas_ch:
        # define spikes paths and check if path exist
        spike_path = "/".join([time_path] + ["KS" + area.upper()])
        if not os.path.exists(spike_path):
            logging.error("spike_path: %s does not exist" % spike_path)
            raise FileExistsError
        # load continuous data
        logging.info("Loading %s", area)
        (
            spike_times_idx,
            spike_clusters,
            cluster_info,
        ) = utils_oe.load_spike_data(spike_path)
        try:
            spike_times_idx, spike_clusters, cluster_info = utils_oe.check_clusters(
                spike_times_idx, spike_clusters, cluster_info, len(c_samples)
            )
        except IndexError:
            logging.error(
                "Spikes of valid units are detected after the end of the recording"
            )
            continue
        except ValueError:
            logging.error("There isn't good or mua clusters")
            continue
        # timestamps of all the spikes (in ms)
        spike_sample = np.floor(c_samples[spike_times_idx] / config.DOWNSAMPLE).astype(
            int
        )

        (
            sp_samples,
            # code_numbers,
            # code_samples,
            # eyes_values,
            # lfp_values,
        ) = data_structure.sort_data_trial(
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

        data = SpikeData(
            sp_samples=sp_samples,
            # eyes_values=eyes_values,
            trial_error=bhv.trial_error,
            code_samples=bhv.code_samples,
            code_numbers=bhv.code_numbers,
            block=bhv.block,
            clusters_id=cluster_info["cluster_id"].values,
            clusters_ch=cluster_info["ch"].values,
            clustersgroup=cluster_info["group"].values,
            clusterdepth=cluster_info["depth"].values,
            start_trials=start_trials,
        )

        # data = data_structure.restructure(
        #     start_trials=start_trials,
        #     end_trials=end_trials,
        #     cluster_info=cluster_info,
        #     spike_sample=spike_sample,
        #     real_strobes=bhv.code_samples,
        #     spike_clusters=spike_clusters,
        #     full_word=bhv.code_numbers,
        #     bhv=bhv,
        # )
        output_d = os.path.normpath(output_dir)
        path = "/".join([output_d] + ["session_struct"] + [subject] + [area])
        file_name = (
            date_time
            + "_"
            + subject
            + "_"
            + area
            + "_e"
            + n_exp
            + "_r"
            + n_record
            + "_spike.h5"
        )
        if not os.path.exists(path):
            os.makedirs(path)
        logging.info("Saving data")
        data.to_python_hdf5("/".join([path] + [file_name]))
        logging.info("Data successfully saved")
        del data


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ks_path", help="Path to KS folders", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--areas", "-a", nargs="*", default=None, help="area", type=str)

    args = parser.parse_args()
    try:
        main(args.ks_path, args.output_dir, args.areas)
    except FileExistsError:
        logging.error("path does not exist")
