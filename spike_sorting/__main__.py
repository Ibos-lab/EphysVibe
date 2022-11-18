import argparse
from pathlib import Path
from spike_sorting import pre_treat_oe
from spike_sorting import utils_oe
import os
import logging
import re

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

    bhv_path = Path(
        "C:/Users/camil/Documents/int/data/openephys/2022-06-09_10-40-34/Record Node 102/experiment1/recording1/220609_TSCM_grid_Riesling.h5"
    )  # I obtain this file using the function convert_format() in matlab
    spike_path = Path(
        "C:/Users/camil/Documents/int/data/openephys/2022-06-09_10-40-34/Record Node 102/experiment1/recording1/continuous/Rhythm_FPGA-100.0/kilosort3"
    )
    output_dir = Path("C:/Users/camil/Documents/int/inVibe/results/")
    config_file = Path("C:/Users/camil/Documents/int/inVibe/spike_sorting/config.json")
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bhv_path", help="Path to the file", type=Path)
    parser.add_argument("spike_path", help="Path to the file", type=Path)
    parser.add_argument(
        "--config_file", "-o", default=config_file, help="Contiguration file", type=Path
    )
    parser.add_argument(
        "--output_dir", "-o", default=output_dir, help="Output directory", type=Path
    )  # "./output"
    args = parser.parse_args()

    main(args.bhv_path, args.spike_path, args.output_dir, args.config_file)
    # main(args.file)
