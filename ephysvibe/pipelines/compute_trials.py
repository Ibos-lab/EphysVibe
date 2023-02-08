import argparse
from pathlib import Path
import logging
import os
import json
from typing import List, Tuple
import numpy as np
from ..spike_sorting import utils_oe, config, data_structure, pre_treat_oe
import glob
from ..structures.bhv_data import BhvData
from ..pipelines import pipe_config
from collections import defaultdict
from typing import Dict

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
    continuous_path: Path,
    output_dir: Path,
    areas: list,
    start_ch: list = [0],
    n_ch: list = [0],
) -> None:
    """Compute spike sorting.

    Args:
        continuous_path (Path):  path to the continuous file (.dat) from OE
        output_dir (Path): output directory
    """

    if not os.path.exists(continuous_path):
        logging.error("continuous_path %s does not exist" % continuous_path)
        raise FileExistsError
    logging.info("-- Start --")

    # define paths
    (
        s_path,
        directory,
        time_path,
        event_path,
    ) = define_paths(continuous_path)

    if len(s_path) < 8:
        logging.error("continuous_path should contain at least 8 /")
        raise NotADirectoryError
    # Select info about the recording from the path
    n_exp = s_path[-5][-1]
    n_record = s_path[-4][-1]
    subject = s_path[-8]
    date_time = s_path[-7]
    # Load json channels_file
    # f = open(areas_path)
    # areas_data = json.load(f)
    # f.close()
    if areas == None:
        areas_ch = pipe_config.AREAS  # areas_data["areas"].keys()
        total_ch = pipe_config.TOTAL_CH
    else:
        total_ch = 0
        areas_ch: Dict[str, list] = defaultdict(list)
        for n, n_area in enumerate(areas):
            areas_ch[n_area] = [start_ch[n], n_ch[n]]
            total_ch += n_ch[n]
    # load bhv data
    bhv_path = os.path.normpath(str(directory) + "/*" + subject + ".mat")
    bhv_path = glob.glob(bhv_path, recursive=True)
    if len(bhv_path) == 0:
        logging.info("Bhv file not found")
        raise ValueError
    logging.info("Loading bhv data")
    bhv = BhvData.from_matlab_mat(bhv_path[0])
    # load timestamps (fs=30000)
    c_samples = np.load(time_path)  # np.floor(/ config.DOWNSAMPLE).astype(int)

    # load events (fs=30000)
    events = utils_oe.load_event_files(event_path)
    # events["samples"] = events["samples"]  # np.floor( / config.DOWNSAMPLE).astype(int)
    shape_0 = len(c_samples)  # areas_data["shape_0"]
    (
        full_word,
        real_strobes,
        start_trials,
        end_trials,
        ds_samples,
        start_time,
        eyes_ds,
        areas_ch,
    ) = pre_treat_oe.pre_treat_oe(
        events=events,
        bhv=bhv,
        c_samples=c_samples,
        areas_ch=areas_ch,
        total_ch=total_ch,
        continuous_path=continuous_path,
    )
    # to ms
    ds_samples = np.floor(ds_samples / config.DOWNSAMPLE).astype(int)
    start_trials = np.floor(start_trials / config.DOWNSAMPLE).astype(int)
    end_trials = np.floor(end_trials / config.DOWNSAMPLE).astype(int)
    real_strobes = np.floor(real_strobes / config.DOWNSAMPLE).astype(int)
    # Iterate by nodes/areas
    for area in areas_ch:
        # define spikes paths
        spike_path = "/".join(s_path[:-1] + ["KS" + area.upper()])
        # check if path exist
        if not os.path.exists(spike_path):
            logging.error("spike_path: %s does not exist" % spike_path)
            raise FileExistsError
        # load continuous data
        logging.info("Loading %s", area)

        (
            idx_sp_ksamples,
            sp_ksamples_clusters_id,
            cluster_info,
        ) = utils_oe.load_spike_data(spike_path)
        if cluster_info.shape[0] != 0:  # if there are valid groups
            spike_sample = np.floor(
                c_samples[idx_sp_ksamples] / config.DOWNSAMPLE
            ).astype(
                int
            )  # timestamps of all the spikes (in ms)

            lfp_ds = utils_oe.compute_lfp(
                continuous_path,
                start_time,
                shape_0=shape_0,
                shape_1=total_ch,
                start_ch=areas_ch[area][0],
                n_ch=areas_ch[area][1],
            )
            data = data_structure.restructure(
                start_trials=start_trials,
                end_trials=end_trials,
                cluster_info=cluster_info,
                spike_sample=spike_sample,
                real_strobes=real_strobes,
                ds_samples=ds_samples,
                sp_ksamples_clusters_id=sp_ksamples_clusters_id,
                full_word=full_word,
                lfp_ds=lfp_ds,
                eyes_ds=eyes_ds,
                bhv=bhv,
            )
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
                + ".h5"
            )
            if not os.path.exists(path):
                os.makedirs(path)
            logging.info("Saving data")

            data.to_python_hdf5("/".join([path] + [file_name]))
            # data_structure.save_data(
            #     data, output_dir, subject, date_time, area, n_exp, n_record
            # )
            logging.info("Data successfully saved")
            del data
            del lfp_ds
        else:
            logging.warning("No recordings")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "continuous_path", help="Path to the continuous file (.dat)", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--areas", "-a", nargs="*", default=None, help="area", type=str)
    args = parser.parse_args()
    try:
        main(args.continuous_path, args.output_dir, args.areas)
    except FileExistsError:
        logging.error("path does not exist")
