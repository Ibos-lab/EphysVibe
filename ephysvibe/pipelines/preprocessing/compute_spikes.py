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


def main(
    ks_path: Path,
    bhv_path: Path,
    output_dir: Path = "./output",
    areas: list = None,
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
    ks_path = os.path.normpath(ks_path)
    s_path = ks_path.split(os.sep)
    # time_path = "/".join(s_path + ["continuous/Acquisition_Board-100.Rhythm Data/"])

    # Select info about the recording from the path
    n_exp = s_path[-4][-1]
    n_record = s_path[-3][-1]
    subject = s_path[-7]
    date_time = s_path[-6]
    # load bhv data
    file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_record + "_bhv.h5"
    bhv_path = os.path.normpath(bhv_path) + "/" + file_name
    bhv_path = glob.glob(bhv_path, recursive=True)
    if len(bhv_path) == 0:
        logging.info("Bhv file not found")
        raise ValueError
    # check n_areas and n_channels
    if areas == None:
        areas = ["lip", "v4", "pfc"]
    # load bhv data
    logging.info("Loading bhv data")
    bhv = BhvData.from_python_hdf5(bhv_path[0])
    # load timestamps and events
    logging.info("Loading continuous/sample_numbers data")
    c_samples = np.load("/".join([ks_path] + ["sample_numbers.npy"]))

    logging.info("Selecting OE samples")

    shift_trial = 1000
    start_trials = bhv.start_trials
    start_trials = start_trials - shift_trial
    end_trials = bhv.end_trials

    # Iterate by nodes/areas
    for area in areas:
        # define spikes paths and check if path exist
        spike_path = "/".join([ks_path] + ["KS" + area.upper()])
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
        sp_samples = data_structure.sort_data_trial(
            clusters=cluster_info,
            spike_sample=spike_sample,
            start_trials=start_trials,
            end_trials=end_trials,
            spike_clusters=spike_clusters,
        )
        data = SpikeData(
            sp_samples=sp_samples,
            trial_error=bhv.trial_error,
            code_samples=bhv.code_samples + shift_trial,
            code_numbers=bhv.code_numbers,
            block=bhv.block,
            clusters_id=cluster_info["cluster_id"].values,
            clusters_ch=cluster_info["ch"].values,
            clustersgroup=cluster_info["group"].values,
            clusterdepth=cluster_info["depth"].values,
            start_trials=start_trials,
        )
        output_d = os.path.normpath(output_dir)
        path = "/".join([output_d] + ["session_struct"] + [area] + ["spikes"])
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
            + "_sp.h5"
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
    parser.add_argument("ks_path", help="Path to KS folders location", type=Path)
    parser.add_argument(
        "bhv_path",
        help="Path to session struct folder containing the bhv files",
        type=Path,
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--areas", "-a", nargs="*", default=None, help="area", type=str)
    args = parser.parse_args()
    try:
        main(args.ks_path, args.bhv_path, args.output_dir, args.areas)
    except FileExistsError:
        logging.error("path does not exist")
