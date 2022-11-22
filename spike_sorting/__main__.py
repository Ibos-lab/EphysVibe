import argparse
from pathlib import Path
import os
import logging
import re
import pre_treat_oe
import utils_oe


logging.basicConfig(level=logging.INFO)


def main(bhv_path, spike_path, output_dir, config_file):

    # load data
    (
        bhv,
        continuous,
        events,
        idx_spiketimes,
        spiketimes_clusters_id,
        cluster_info,
        config_data,
    ) = utils_oe.load_data(bhv_path, spike_path, config_file)

    split_path = os.path.normpath(bhv_path).split(os.sep)
    subject = re.split(r"[_;.]", split_path[-1])[-2]
    date_time = split_path[-5]

    pre_treat_oe.pre_treat_oe(
        continuous,
        events,
        bhv,
        idx_spiketimes,
        config_data,
        cluster_info,
        output_dir,
        spiketimes_clusters_id,
        subject,
        date_time,
    )


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bhv_path", help="Path to the file", type=Path)
    parser.add_argument("spike_path", help="Path to the file", type=Path)
    parser.add_argument("config_file", help="Contiguration file", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()
    main(args.bhv_path, args.spike_path, args.output_dir, args.config_file)
