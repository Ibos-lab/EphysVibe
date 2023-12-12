import argparse
from pathlib import Path
import logging
import os
from typing import List
import numpy as np
from ...spike_sorting import utils_oe
from .. import pipe_config
from ...structures.bhv_data import BhvData
from ...structures.eye_data import EyeData
from ...task import task_constants

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    ks_path: Path,
    bhv_path: Path,
    output_dir: Path = "./output",
) -> None:
    """Compute eye structure.

    Args:
        ks_path (Path): path to Kilosort folders.
        bhv_path (Path): path to bhv file.
        output_dir (Path): output directory.
    """
    logging.info("-- Start --")
    # define paths
    ks_path = os.path.normpath(ks_path)
    time_path = "/".join([ks_path] + ["sample_numbers.npy"])
    continuous_path = "/".join([ks_path] + ["continuous.dat"])
    # load bhv data
    logging.info("Loading Bhv data")
    bhv = BhvData.from_python_hdf5(bhv_path)
    # Select info about the recording from the path
    logging.info("Loading timestamps")
    # --------------------------
    c_samples = np.load(time_path)
    date_time = bhv.date_time
    subject = bhv.subject
    n_exp = bhv.experiment
    n_rec = bhv.recording
    code_samples = bhv.code_samples
    code_numbers = bhv.code_numbers
    before_trial = 1000
    iti = 1500
    next_trial = 6000
    trials_end = code_samples[
        np.where(code_numbers == task_constants.EVENTS_B1["end_trial"], True, False)
    ]
    trials_start = code_samples[:, 0]
    # --------------------------
    trials_max_duration = max(trials_end - trials_start)
    trials_max_duration = int(trials_max_duration + before_trial + iti + next_trial)
    total_ch = pipe_config.TOTAL_CH
    start_ch = pipe_config.area_start_nch["eyes"][0]
    n_eyes = pipe_config.area_start_nch["eyes"][1]
    n_trials = trials_start.shape[0]
    logging.info("load_eyes")
    shape_0 = len(c_samples)
    eyes_ds = utils_oe.load_eyes(
        continuous_path,
        shape_0=shape_0,
        shape_1=total_ch,
        start_ch=start_ch,
        n_eyes=n_eyes,
    )
    # Split eye position by trials
    tr_eye = np.full((n_trials, 3, trials_max_duration), np.nan)
    for i_t in range(n_trials):
        start_trial = (trials_start[i_t] - before_trial).astype(int)
        end_trial = (trials_end[i_t] + iti + next_trial).astype(int)
        if end_trial > eyes_ds.shape[1]:
            end_trial = eyes_ds.shape[1]
        tr_eye[i_t, :, : int(end_trial - start_trial)] = eyes_ds[
            :, start_trial:end_trial
        ]
    code_samples_trial = code_samples - code_samples[:, 0].reshape(-1, 1) + before_trial
    # Save data
    eye_data = EyeData(
        date_time=date_time,
        subject=subject,
        experiment=n_exp,
        recording=n_rec,
        eye=tr_eye,
        block=bhv.block,
        trial_error=bhv.trial_error,
        code_samples=code_samples_trial,
        code_numbers=code_numbers,
        position=bhv.position,
        pos_code=bhv.pos_code,
        sample_id=bhv.sample_id,
        test_stimuli=bhv.test_stimuli,
        test_distractor=bhv.test_distractor,
        eye_ml=bhv.eye_ml,
    )
    output_dir = os.path.normpath(output_dir)
    output_dir = "/".join([output_dir] + ["session_struct"] + ["eye"])
    file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_rec + "_eye.h5"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = "/".join([output_dir] + [file_name])
    logging.info("Saving data")
    logging.info(file_name)
    eye_data.to_python_hdf5(output_dir)
    logging.info("Data successfully saved")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ks_path", help="Path to KS folders location", type=Path)

    parser.add_argument("bhv_path", help="Path to Bhv file", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )

    args = parser.parse_args()
    try:
        main(args.ks_path, args.bhv_path, args.output_dir)
    except FileExistsError:
        logging.error("path does not exist")
