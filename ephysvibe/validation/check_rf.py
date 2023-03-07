from ephysvibe.structures.trials_data import TrialsData
import numpy as np
from pathlib import Path
import cv2
import os
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(data_path: Path, path_img: Path, output_dir: Path):
    # define paths and read data
    s_path = os.path.normpath(data_path).split(os.sep)
    area = s_path[-2]
    output_dir = os.path.normpath(output_dir) + "/" + s_path[-1]
    path_id = s_path[-1][:-3]
    logging.info("Loading data")
    data = TrialsData.from_python_hdf5(data_path)
    in_out = []
    count_neuron, count_mua = 0, 0
    for cluster in data.clustersgroup:
        if cluster == "good":
            cluster = "neuron"
            count_neuron += 1
            n_group = count_neuron
        else:
            count_mua += 1
            n_group = count_mua

        c_path_img = "/".join(
            [os.path.normpath(path_img)]
            + [path_id + "_" + cluster + "_" + str(n_group) + ".jpg"]
        )
        img = cv2.imread(c_path_img)
        while 1:
            cv2.imshow("ff", img)
            k = cv2.waitKey(30)
            if k == 32:  # space key to stop
                break
            elif k == 49:  # if user press 1 (in)
                in_out.append(1)
                logging.info(1)  # else print its value
            elif k == 50:  # if user press 2
                in_out.append(2)
                logging.info(2)
            else:  # normally -1 is returned
                continue

    cv2.destroyAllWindows()
    new_data = vars(data).copy()
    new_data["neuron_cond"] = np.array(in_out)
    new_data["clustersgroup"] = np.array(data.clustersgroup, dtype=object)
    new_data = TrialsData(**new_data)
    new_data.to_python_hdf5(output_dir)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_path", help="Path to the data (TrialsData)", type=Path)
    parser.add_argument("path_img", help="Path to the img folder", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    args = parser.parse_args()

    try:
        main(args.data_path, args.path_img, args.output_dir)

    except FileExistsError:

        logging.error("path does not exist")
