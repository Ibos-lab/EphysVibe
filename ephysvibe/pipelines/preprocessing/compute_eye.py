import argparse
from pathlib import Path
import logging
import os
from typing import List
import numpy as np
from ...spike_sorting import utils_oe
from .. import pipe_config

# from ...spike_sorting import utils_oe, config, data_structure
import glob
from ...structures.bhv_data import BhvData
from ...structures.spike_data import SpikeData
from ...structures.eye_data import EyeData
from ...task import task_constants

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    time_path: Path,
    continuous_path: Path,
    bhv_path: Path,
    output_dir: Path = "./output",
) -> None:
    """Compute trials.

    Args:
        sp_path (Path):  path to the continuous file (.dat) from OE.
        output_dir (Path): output directory.
    """
    # if not os.path.exists(sp_path):
    #     logging.error("sp_path %s does not exist" % sp_path)
    #     raise FileExistsError
    logging.info("-- Start --")

    # define paths
    # sp_path = os.path.normpath(sp_path)
    # s_path = sp_path.split(os.sep)
    # load bhv data
    logging.info("Loading Spike and Bhv data")
    bhv = BhvData.from_python_hdf5(bhv_path)
    # Select info about the recording from the path

    logging.info("Loading timestamps and events")
    c_samples = np.load(time_path)

    # date_time = bhv.date_time
    # subject = bhv.subject
    # n_exp = bhv.experiment
    # n_rec = bhv.recording
    # load bhv data
    # file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_rec + "_bhv.h5"
    # bhv_path = "/".join(s_path[:-3] + ["bhv"] + [file_name])
    # bhv_path = os.path.normpath(bhv_path)

    # --------------------------
    code_samples = bhv.code_samples
    code_numbers = bhv.code_numbers
    # --------------------------
    before_trial = 1000
    iti = 1500
    next_trial = 6000
    trials_end = code_samples[
        np.where(code_numbers == task_constants.EVENTS_B1["end_trial"], True, False)
    ]
    trials_start = code_samples[:, 0]
    trials_max_duration = max(trials_end - trials_start)
    trials_max_duration = int(trials_max_duration + before_trial + iti + next_trial)
    total_ch = pipe_config.TOTAL_CH
    start_ch = pipe_config.area_start_nch["eyes"][0]
    n_eyes = pipe_config.area_start_nch["eyes"][1]
    n_trials = trials_start.shape[0]
    logging.info("load_eyes")
    eyes_ds = utils_oe.load_eyes(
        continuous_path,
        shape_0=len(c_samples),
        shape_1=total_ch,
        start_ch=start_ch,
        n_eyes=n_eyes,
    )
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

    eye_data = EyeData(
        # date_time=date_time,
        # subject=subject,
        # area=area,
        # experiment=n_exp,
        # recording=n_rec,
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
    ####### ------------------------

    # output_d = os.path.normpath(output_dir)
    # path = "/".join([output_d] + ["session_struct"] + [area] + ["neurons"])

    # file_name = (
    #     date_time
    #     + "_"
    #     + subject
    #     + "_"
    #     + area
    #     + "_e"
    #     + n_exp
    #     + "_r"
    #     + n_rec
    #     + "_"
    #     + cluster
    #     + str(i_cluster)
    #     + "_neu.h5"
    # )
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # logging.info("Saving data")
    # logging.info(file_name)
    eye_data.to_python_hdf5(output_dir)
    # logging.info("Data successfully saved")
    # del neuron_data


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("time_path", help="Path to KS folders location", type=Path)
    parser.add_argument(
        "continuous_path", help="Path to KS folders location", type=Path
    )
    parser.add_argument("bhv_path", help="Path to KS folders location", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )

    args = parser.parse_args()
    try:
        main(args.time_path, args.continuous_path, args.bhv_path, args.output_dir)
    except FileExistsError:
        logging.error("path does not exist")