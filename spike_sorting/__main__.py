import argparse
from pathlib import Path
from spike_sorting import pre_treat_oe
import logging

logging.basicConfig(level=logging.INFO)


def main(directory, bhv_filepath, spike_dir):
    "Spike sorting"
    result = pre_treat_oe.pre_treat_oe(
        directory, bhv_filepath, spike_dir, start_code=9, end_code=18
    )


if __name__ == "__main__":

    directory = "C:/Users/camil/Documents/int/data/openephys/2022-06-09_10-40-34/"
    bhv_filepath = "/Record Node 102/experiment1/recording1/220609_TSCM_grid_Riesling.h5"  # I obtain this file using the function convert_format() in matlab
    spike_dir = (
        "Record Node 102/experiment1/recording1/continuous/Rhythm_FPGA-100.0/kilosort3"
    )

    # Parse arguments
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.RawDescriptionHelpFormatter
    # )
    # parser.add_argument("file", help="Path to the file", type=Path)
    # args = parser.parse_args()

    main(directory, bhv_filepath, spike_dir)
    # main(args.file)
