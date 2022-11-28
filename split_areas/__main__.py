import argparse
from pathlib import Path
import logging
import os
from open_ephys.analysis import Session
import numpy as np
import json
import os
import shutil
import glob

logging.basicConfig(level=logging.INFO)


def copy_files(src, dst, l):
    logging.info("Copying files")
    s_dst = os.path.normpath(dst).split(os.sep)
    for (root, _, files) in os.walk(src, topdown=True):
        for file_name in files:
            s_root = os.path.normpath(root).split(os.sep)
            split_root = s_root[l:]

            idx_c = np.where(np.array(split_root) == "continuous")[0]
            idx_e = np.where(np.array(split_root) == "events")[0]
            if (idx_c.shape[0] != 0) and ((idx_c[0] + 1) < len(split_root)):
                split_root.pop(idx_c[0] + 1)

            elif (idx_e.shape[0] != 0) and ((idx_e[0] + 2) < len(split_root)):
                split_root.pop(idx_e[0] + 2)
                split_root.pop(idx_e[0] + 1)

            destination = "/".join(s_dst + split_root)

            source = "/".join(s_root + [file_name])
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.copy(source, "/".join([destination] + [file_name]))


def main(directory, output_dir):

    session = Session(directory)
    recordnode = session.recordnodes[0]
    # Load continuous data
    continuous = recordnode.recordings[0].continuous[0]

    split_path = os.path.normpath(recordnode.recordings[0].directory).split(os.sep)
    split_output_dir = os.path.normpath(output_dir).split(os.sep)
    shape_0 = continuous.samples.shape[0]

    areas = ["lip", "v4", "pfc", "eyes"]
    n_channels = [32, 64, 64, 3]

    n_total = 0
    channels_info = {"shape_0": shape_0, "areas": {}}
    # with the split we add subject and datetime
    destination = "/".join(split_output_dir + split_path[-5:-3])
    l = len(split_path[:-5])

    copy_files(src=directory, dst=output_dir, l=l)

    save_path = "/".join(split_output_dir + split_path[-5:])

    for n_area, n_ch in zip(areas, n_channels):

        logging.info("Split: %s" % (n_area))

        area_dat = continuous.samples[:, n_total : n_total + n_ch]
        n_total = n_total + n_ch

        node_save_path = "/".join([save_path] + ["Record Node " + n_area])
        # check if path exist
        if not os.path.exists(node_save_path):
            os.makedirs(node_save_path)

        logging.info("Saving %s file" % (n_area))
        shape_1 = n_ch
        memmap_file = np.memmap(
            "/".join([node_save_path] + [n_area + ".dat"]),
            mode="w+",
            dtype="int16",
            shape=(shape_0, shape_1),
        )
        memmap_file[:] = area_dat[:]

        channels_info["areas"][n_area] = n_ch

    json_path = "/".join([save_path] + ["channels_info.json"])
    with open(json_path, "w") as outfile:
        json.dump(channels_info, outfile)
    logging.info("Successfully run")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", help="Path to the directory", type=Path)

    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()

    main(args.directory, args.output_dir)
