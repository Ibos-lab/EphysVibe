import argparse
from pathlib import Path
import logging
import os
from open_ephys.analysis import Session
import numpy as np
import json
import os
import shutil

logging.basicConfig(level=logging.INFO)


def main(continuous_path, output_dir):

    split_path = os.path.normpath(continuous_path).split(os.sep)
    dir = "/".join(split_path[:-6])
    session = Session(dir)
    recordnode = session.recordnodes[0]
    # Load continuous data
    continuous = recordnode.recordings[0].continuous[0]
    shape_0 = continuous.samples.shape[0]

    areas = ["lip", "v4", "pfc"]
    n_channels = [32, 64, 64]
    n_total = 0
    channels_info = {"shape_0": shape_0, "areas": {}}
    for n_area, n_ch in zip(areas, n_channels):

        save_path = (
            str(output_dir)
            + "/"
            + "/".join(split_path[-8:-6])
            + "/"
            + "Record Node "
            + n_area
            + "/"
            + "/".join(split_path[-5:-1])
        )
        if n_area == "lip":
            area_dat = np.concatenate(
                [
                    continuous.samples[:, n_total : n_total + n_ch],
                    continuous.samples[:, -3:],
                ],
                axis=1,
            )
            shape_1 = n_ch + 3
        else:
            area_dat = continuous.samples[:, n_total : n_total + n_ch]
            shape_1 = n_ch

        channels_info["areas"][n_area] = n_ch

        n_total = n_total + n_ch

        source_folder = "/".join(split_path[:-5]) + "/"
        l = len(split_path[:-5])

        destination_folder = (
            str(output_dir)
            + "/"
            + "/".join(split_path[-8:-6])
            + "/"
            + "Record Node "
            + n_area
        )

        # fetch all files
        for (root, _, files) in os.walk(source_folder, topdown=True):

            # construct full file path
            for file_name in files:

                if file_name != "continuous.dat":

                    split_root = os.path.normpath(root).split(os.sep)[l:]

                    source = root + "/" + file_name
                    destination = destination_folder + "/" + "/".join(split_root)
                    # copy only files

                    if not os.path.exists(destination):
                        os.makedirs(destination)
                    shutil.copy(source, destination + "/" + file_name)

        memmap_file = np.memmap(
            save_path + "/" + split_path[-1],
            mode="w+",
            dtype="int16",
            shape=(shape_0, shape_1),
        )
        memmap_file[:] = area_dat[:]

    json_path = (
        str(output_dir) + "/" + "/".join(split_path[-8:-6]) + "/" + "channels_info.json"
    )
    with open(json_path, "w") as outfile:
        json.dump(channels_info, outfile)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("continuous_path", help="Path to the directory", type=Path)

    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()

    main(args.continuous_path, args.output_dir)
