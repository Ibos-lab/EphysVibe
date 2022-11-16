import argparse
from pathlib import Path
from spike_sorting import pre_treat_oe
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def main(directory, bhv_filepath, spike_dir, save_dir, info):
    "Spike sorting"
    pre_treat_oe.pre_treat_oe(directory, bhv_filepath, spike_dir, save_dir, info)


if __name__ == "__main__":

    directory = "C:/Users/camil/Documents/int/data/openephys/2022-06-09_10-40-34/"
    bhv_filepath = "/Record Node 102/experiment1/recording1/220609_TSCM_grid_Riesling.h5"  # I obtain this file using the function convert_format() in matlab
    spike_dir = (
        "Record Node 102/experiment1/recording1/continuous/Rhythm_FPGA-100.0/kilosort3"
    )
    save_dir = "C:/Users/camil/Documents/int/inVibe/results/"
    info = {"subject": "M", "area": "LIP", "date": "16-11-2022-12-11"}
    # Parse arguments
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.RawDescriptionHelpFormatter
    # )
    # parser.add_argument("file", help="Path to the file", type=Path)
    # args = parser.parse_args()

    main(directory, bhv_filepath, spike_dir, save_dir, info)
    # main(args.file)
