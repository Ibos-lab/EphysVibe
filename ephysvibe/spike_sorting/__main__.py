import argparse
from pathlib import Path
import logging
import os
import json
from typing import List, Tuple
import numpy as np
from . import utils_oe, config, data_structure, pre_treat_oe
import glob

logging.basicConfig(level=logging.INFO)


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
    areas_path = "/".join(s_path[:-1] + ["channels_info.json"])

    return (
        s_path,
        directory,
        time_path,
        event_path,
        areas_path,
    )


def main(continuous_path: Path, output_dir: Path) -> None:
    """Compute spike sorting.

    Args:
        continuous_path (Path):  path to the continuous file (.dat) from OE
        output_dir (Path): output directory
    """
    # define paths
    (
        s_path,
        directory,
        time_path,
        event_path,
        areas_path,
    ) = define_paths(continuous_path)
    # Select info about the recording from the path
    n_exp = s_path[-5][-1]
    n_record = s_path[-4][-1]
    subject = s_path[-8]
    date_time = s_path[-7]
    # Load json channels_file
    f = open(areas_path)
    areas_data = json.load(f)
    f.close()
    areas = areas_data["areas"].keys()
    shape_0 = areas_data["shape_0"]
    # load bhv data
    bhv = utils_oe.load_bhv_data(directory, subject)
    # load timestamps
    c_samples = np.load(time_path)
    # load events
    events = utils_oe.load_event_files(event_path)

    (
        full_word,
        real_strobes,
        start_trials,
        blocks,
        dict_bhv,
        ds_samples,
        start_time,
        eyes_ds,
        areas_data,
    ) = pre_treat_oe.pre_treat_oe(
        events=events,
        bhv=bhv,
        c_samples=c_samples,
        areas_data=areas_data,
        s_path=s_path,
        shape_0=shape_0,
    )
    # Iterate by nodes/areas
    for area in areas:
        # define dat and spikes paths
        dat_path = "/".join(s_path[:-1] + ["Record Node " + area] + [area + ".dat"])
        spike_path = glob.glob(
            "/".join(
                s_path[:-1] + ["Record Node " + area] + [config.KILOSORT_FOLDER_NAME]
            )
        )[0]

        # load continuous data
        logging.info("Loading %s", area)
        continuous = utils_oe.load_dat_file(
            dat_path, shape_0=shape_0, shape_1=areas_data["areas"][area]
        )
        # load spike data
        (
            idx_sp_ksamples,
            sp_ksamples_clusters_id,
            cluster_info,
        ) = utils_oe.load_spike_data(spike_path)
        if cluster_info.shape[0] != 0:  # if there are valid groups
            spike_sample = c_samples[idx_sp_ksamples]  # timestamps of all the spikes
            logging.info("Computing LFPs")
            lfp_ds = utils_oe.compute_lfp(continuous[:, start_time:])
            data = data_structure.restructure(
                start_trials=start_trials,
                blocks=blocks,
                cluster_info=cluster_info,
                spike_sample=spike_sample,
                real_strobes=real_strobes,
                ds_samples=ds_samples,
                sp_ksamples_clusters_id=sp_ksamples_clusters_id,
                full_word=full_word,
                lfp_ds=lfp_ds,
                eyes_ds=eyes_ds,
                dict_bhv=dict_bhv,
            )
            data_structure.save_data(
                data, output_dir, subject, date_time, area, n_exp, n_record
            )
        else:
            logging.info("No recordings")


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
    args = parser.parse_args()
    main(args.continuous_path, args.output_dir)
