import argparse
from pathlib import Path
import logging
import os
from open_ephys.analysis import Session
import numpy as np
import json
import os


logging.basicConfig(level=logging.INFO)


def main(continuous_path):

    # Split path
    s_path = os.path.normpath(continuous_path).split(os.sep)
    # Select info about the recording
    n_node = int(s_path[-5][-1]) - 1
    n_record = int(s_path[-4][-1]) - 1
    directory = "/".join(s_path[:-6])
    # Load session
    session = Session(directory)
    recordnode = session.recordnodes[n_node]
    # Load continuous data
    continuous = recordnode.recordings[n_record].continuous[0]

    shape_0 = continuous.samples.shape[0]
    areas = ["lip", "v4", "pfc", "eyes"]
    n_channels = [32, 64, 64, 3]
    n_total = 0
    # Define dict to save info about channels
    channels_info = {"shape_0": shape_0, "areas": {}}

    save_path = "/".join(s_path[:-1])
    for n_area, n_ch in zip(areas, n_channels):
        logging.info("Split: %s" % (n_area))
        # Select the channels that correspond to the area
        area_dat = continuous.samples[:, n_total : n_total + n_ch]
        n_total = n_total + n_ch
        # Path where to save the splited data
        node_save_path = "/".join([save_path] + ["Record Node " + n_area])
        # check if path exist
        if not os.path.exists(node_save_path):
            os.makedirs(node_save_path)
        logging.info("Saving %s file" % (n_area))
        # Save file
        memmap_file = np.memmap(
            "/".join([node_save_path] + [n_area + ".dat"]),
            mode="w+",
            dtype="int16",
            shape=(shape_0, n_ch),
        )
        memmap_file[:] = area_dat[:]

        channels_info["areas"][n_area] = n_ch

    # Save json file
    json_path = "/".join([save_path] + ["channels_info.json"])
    with open(json_path, "w") as outfile:
        json.dump(channels_info, outfile)

    logging.info("Successfully run")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "continuous_path", help="Path to the continuous file", type=Path
    )

    args = parser.parse_args()
    main(args.continuous_path)
