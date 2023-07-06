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
    bhv_data_file: Path,
    output_dir: Path,
) -> None:
    """Compute trials.

    Args:
        continuous_path (Path):  path to
        output_dir (Path): output directory.
        areas (list): list containing the areas to which to compute the trials data.
        start_ch (list): list containing the index of the first channel for each area.
        n_ch (list): list containing the number of channels for each area.
    """
    if not os.path.exists(bhv_data_file):
        logging.error("bhv_data_file %s does not exist" % bhv_data_file)
        raise FileExistsError
    logging.info("-- Start --")

    # Select info about the recording from the path
    n_exp = s_path[-5][-1]
    n_record = s_path[-4][-1]
    subject = s_path[-8]
    date_time = s_path[-7]

    # load bhv data
    logging.info("Loading bhv data")
    bhv = BhvData.from_matlab_mat(bhv_data_file)

    # load events
    events = utils_oe.load_event_files(event_path)

    # reconstruct 8 bit words
    real_strobes, len_idx = utils_oe.find_events_codes(
        events, code_numbers=bhv.code_numbers
    )

    bhv = utils_oe.select_trials_bhv(bhv, len_idx)
    # to ms
    real_strobes = np.floor(real_strobes / config.DOWNSAMPLE).astype(int)
    bhv.code_samples = real_strobes

    output_d = os.path.normpath(output_dir)
    path = "/".join([output_d] + ["session_struct"] + [subject])
    file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_record + ".h5"
    if not os.path.exists(path):
        os.makedirs(path)
    logging.info("Saving data")
    bhv.to_python_hdf5("/".join([path] + [file_name]))
    logging.info("Data successfully saved")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bhv_data_file", help="Path to the bhv file (matlab format)", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()
    try:
        main(args.bhv_data_file, args.output_dir)
    except FileExistsError:
        logging.error("path does not exist")
