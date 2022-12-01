import argparse
from pathlib import Path
import logging
import os
import json
import numpy as np
import utils_oe, config, data_structure, pre_treat_oe

logging.basicConfig(level=logging.INFO)


def define_paths(continuous_path):
    # define paths
    s_path = os.path.normpath(continuous_path).split(os.sep)
    directory = "/".join(s_path[:-3])
    time_path = "/".join(s_path[:-1] + ["timestamps.npy"])
    event_path = "/".join(s_path[:-3] + ["events"] + ["Rhythm_FPGA-100.0"] + ["TTL_1"])
    areas_path = "/".join(s_path[:-1] + ["channels_info.json"])

    return (
        s_path,
        directory,
        time_path,
        event_path,
        areas_path,
    )


def main(continuous_path, output_dir):
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
    c_timestamps = np.load(time_path)
    # load events
    events = utils_oe.load_event_files(event_path)

    (
        full_word,
        real_strobes,
        start_trials,
        blocks,
        dict_bhv,
        ds_timestamps,
        start_time,
        eyes_ds,
        areas_data,
    ) = pre_treat_oe.pre_treat_oe(
        events, bhv, c_timestamps, areas_data, s_path, shape_0
    )

    # Iterate by nodes/areas
    for area in areas:
        # define dat and spikes paths
        dat_path = "/".join(s_path[:-1] + ["Record Node " + area] + [area + ".dat"])
        spike_path = "/".join(
            s_path[:-1] + ["Record Node " + area] + [config.KILOSORT_FOLDER_NAME]
        )
        # load continuous data
        logging.info("Loading %s", area)
        continuous = utils_oe.load_dat_file(
            dat_path, shape_0=shape_0, shape_1=areas_data["areas"][area]
        )
        # load spike data
        idx_spiketimes, spiketimes_clusters_id, cluster_info = utils_oe.load_spike_data(
            spike_path
        )
        if cluster_info.shape[0] != 0:  # if there are valid groups
            spiketimes = c_timestamps[idx_spiketimes]  # timestamps of all the spikes
            logging.info("Computing LFPs")
            lfp_ds = utils_oe.compute_lfp(continuous[:, start_time:])
            data = data_structure.restructure(
                start_trials,
                blocks,
                cluster_info,
                spiketimes,
                real_strobes,
                ds_timestamps,
                spiketimes_clusters_id,
                full_word,
                lfp_ds,
                eyes_ds,
                dict_bhv,
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
        "continuous_path", help="Path to the continuous file", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()
    main(args.continuous_path, args.output_dir)
