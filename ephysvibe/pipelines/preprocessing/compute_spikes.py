import argparse
from pathlib import Path
import logging
import os
from typing import List
import numpy as np
from ...spike_sorting import utils_oe, config, data_structure
import glob
from ...structures.bhv_data import BhvData
from ...structures.spike_data import SpikeData

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    ks_path: Path,
    output_dir: Path = "./output",
    areas: list = None,
) -> None:
    """Compute trials.
    Args:
        ks_path (Path):  path to the continuous file (.dat) from OE.
        output_dir (Path): output directory.
        areas (list): list containing the areas to which to compute the trials data.
    """
    if not os.path.exists(ks_path):
        logging.error("ks_path %s does not exist" % ks_path)
        raise FileExistsError
    logging.info("-- Start --")
    # define paths
    ks_path = os.path.normpath(ks_path)
    s_path = ks_path.split(os.sep)
    # Select info about the recording from the path
    n_exp = s_path[-4][-1]
    n_record = s_path[-3][-1]
    subject = s_path[-7]
    date_time = s_path[-6]
    # check n_areas and n_channels
    if areas == None:
        areas = ["lip", "v4", "pfc"]
    # load timestamps and events
    logging.info("Loading continuous/sample_numbers data")
    c_samples = np.load("/".join([ks_path] + ["sample_numbers.npy"]))
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
        sp_samples = data_structure.get_clusters_spikes(
            clusters=cluster_info,
            spike_sample=spike_sample,
            spike_clusters=spike_clusters,
        )
        data = SpikeData(
            sp_samples=sp_samples,
            clusters_id=cluster_info["cluster_id"].values,
            clusters_ch=cluster_info["ch"].values,
            clustersgroup=cluster_info["group"].values,
            clusterdepth=cluster_info["depth"].values,
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
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument("--areas", "-a", nargs="*", default=None, help="area", type=str)
    args = parser.parse_args()
    try:
        main(args.ks_path, args.bhv_path, args.output_dir, args.areas)
    except FileExistsError:
        logging.error("path does not exist")
