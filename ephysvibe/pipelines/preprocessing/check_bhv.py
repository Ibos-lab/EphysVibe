import argparse
from pathlib import Path
import logging
import os
import numpy as np
from ...spike_sorting import utils_oe, config
from ...structures.bhv_data import BhvData

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    bhv_data_file: Path,
    output_dir: Path = "./",
    event_path: Path = "events/Acquisition_Board-100.Rhythm Data/TTL",
) -> None:
    """Check bhv and creates bhv structure.

    Args:
        bhv_data_file (Path): path to the bhv file (matlab format).
        output_dir (Path, optional): output directory. Defaults to "./".
        event_path (Path, optional): path to event folder. Defaults to "events/Acquisition_Board-100.Rhythm Data/TTL".
    """
    if not os.path.exists(bhv_data_file):
        logging.error("bhv_data_file %s does not exist" % bhv_data_file)
        raise FileExistsError
    logging.info("-- Start --")
    logging.info("%s" % bhv_data_file)
    # Select info about the recording from the path
    s_path = os.path.normpath(bhv_data_file).split(os.sep)
    event_data_files = "/".join(s_path[:-1] + [event_path])
    n_exp = s_path[-3][-1]
    n_record = s_path[-2][-1]
    subject = s_path[-6]
    date_time = s_path[-5]
    # load bhv data
    logging.info("Loading bhv data")
    bhv = BhvData.from_matlab_mat(bhv_data_file)
    # load events
    events = utils_oe.load_event_files(event_data_files)
    # reconstruct 8 bit words
    _, real_strobes, len_idx, idx_start, idx_end = utils_oe.find_events_codes(
        events, code_numbers=bhv.code_numbers
    )
    start_trials = real_strobes[idx_start]
    end_trials = real_strobes[idx_end]
    if len_idx is not None:
        bhv = utils_oe.select_trials_bhv(bhv, len_idx)
    # to ms
    real_strobes = np.floor(real_strobes / config.DOWNSAMPLE).astype(int)
    start_trials = np.floor(start_trials / config.DOWNSAMPLE).astype(int)
    end_trials = np.floor(end_trials / config.DOWNSAMPLE).astype(int)
    bhv.start_trials = start_trials
    bhv.end_trials = end_trials

    n_trials, n_codes = bhv.code_numbers.shape
    code_samples = np.full((n_trials, n_codes), np.nan)

    for i_trial in range(n_trials):
        events_mask = np.logical_and(
            real_strobes >= start_trials[i_trial],
            real_strobes <= end_trials[i_trial],
        )
        code_samples[i_trial, : np.sum(events_mask)] = (
            real_strobes[events_mask] - start_trials[i_trial]
        )

    bhv.code_samples = code_samples

    file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_record + "_bhv.h5"

    output_dir = "/".join([os.path.normpath(output_dir)] + ["session_struct"] + ["bhv"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = "/".join([os.path.normpath(output_dir)] + [file_name])
    logging.info("Saving data")
    bhv.to_python_hdf5(output_dir)
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
