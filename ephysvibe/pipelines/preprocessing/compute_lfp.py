import argparse
from pathlib import Path
import logging
import os
from typing import List, Dict
import numpy as np
from ...spike_sorting import utils_oe, config
import glob
from ...structures.bhv_data import BhvData
from ...structures.lfp_data import LfpData
from . import preproc_config
from collections import defaultdict
import gc

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    continuous_path: Path,
    bhv_path: Path,
    output_dir: Path = "./output",
    event_path: Path = "events/Acquisition_Board-100.Rhythm Data/TTL",
    areas: list = None,
    start_ch: list = None,
    n_ch: list = None,
    f_lp: int = 250,
    f_hp: int = 3,
    filt: bool = True,
) -> None:
    """Filter and downsample lfps.

    Args:
        continuous_path (Path): path to the continuous file (.dat) from OE.
        bhv_path (Path): path to the folder containing the bhv file (bhv.h5).
        output_dir (Path, optional): output directory. Defaults to "./output".
        event_path (Path, optional): path to event folder. Defaults to "events/Acquisition_Board-100.Rhythm Data/TTL".
        areas (list, optional): list containing the areas to which to compute the trials data. Defaults to None.
        start_ch (list, optional): list containing the index of the first channel for each area. Defaults to None.
        n_ch (list, optional): list containing the number of channels for each area. Defaults to None.
        f_lp (int, optional): low pass frequency. Defaults to 250.
        f_hp (int, optional): high pass frequency. Defaults to 3.
        filt (bool, optional): whether to filter lfps. Defaults to True.
    """
    if not os.path.exists(continuous_path):
        logging.error("continuous_path %s does not exist" % continuous_path)
        raise FileExistsError
    logging.info("-- Start --")
    # define paths
    s_path = os.path.normpath(continuous_path).split(os.sep)
    time_path = "/".join(s_path[:-1] + ["sample_numbers.npy"])
    event_path = os.path.normpath(event_path)
    event_path = "/".join(s_path[:-3] + [event_path])
    # check if paths exist
    if not os.path.exists(time_path):
        logging.error("time_path: %s does not exist" % time_path)
        raise FileExistsError
    if not os.path.exists(event_path):
        logging.error("event_path: %s does not exist" % event_path)
        raise FileExistsError
    if len(s_path) < 8:
        logging.error("continuous_path should contain at least 8 /")
        raise NotADirectoryError
    # Select info about the recording from the path
    n_exp = s_path[-5][-1]
    n_record = s_path[-4][-1]
    subject = s_path[-8]
    date_time = s_path[-7]
    # check n_areas and n_channels
    if areas == None:
        areas_ch = preproc_config.AREAS.copy()
    else:
        if n_ch == None or start_ch == None:
            raise KeyError("n_ch or start_ch = None")
        areas_ch: Dict[str, list] = defaultdict(list)
        for n, n_area in enumerate(areas):
            areas_ch[n_area] = [start_ch[n], n_ch[n]]
    total_ch = preproc_config.TOTAL_CH
    # load bhv data
    file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_record + "_bhv.h5"
    bhv_path = os.path.normpath(bhv_path) + "/" + file_name
    bhv_path = glob.glob(bhv_path, recursive=True)
    if len(bhv_path) == 0:
        logging.info("Bhv file not found")
        raise ValueError
    logging.info("Loading bhv data")
    logging.info(bhv_path[0])
    bhv = BhvData.from_python_hdf5(bhv_path[0])
    # load timestamps and events
    logging.info("Loading timestamps and events")
    c_samples = np.load(time_path)
    events = utils_oe.load_event_files(event_path)
    shape_0 = len(c_samples)
    start_trials = bhv.start_trials
    logging.info("select_samples")
    ds_samples, idx_start_samp = utils_oe.select_samples(
        c_samples=c_samples,
        e_samples=events["samples"],
        fs=config.FS,
        t_before_event=config.T_EVENT,
        downsample=config.DOWNSAMPLE,
    )
    # check if eyes
    start_ch, n_eyes = areas_ch.pop("eyes", False)
    if n_eyes:
        logging.info("load_eyes")
        eyes_ds = utils_oe.load_eyes(
            continuous_path,
            shape_0=len(c_samples),
            shape_1=total_ch,
            start_ch=start_ch,
            n_eyes=n_eyes,
            idx_start_time=idx_start_samp,
        )

    # to ms
    ds_samples = np.floor(ds_samples / config.DOWNSAMPLE).astype(int)
    idx_start = []
    for i_start in start_trials:
        idx_s = np.where(ds_samples == i_start)[0][0]
        idx_start.append(idx_s)
    # Iterate by nodes/areas
    for area in areas_ch:
        # define spikes paths and check if path exist
        logging.info(area)
        logging.info("compute_lfp")
        lfp_ds = utils_oe.compute_lfp(
            continuous_path=continuous_path,
            shape_0=shape_0,
            shape_1=total_ch,
            start_time=idx_start_samp,
            start_ch=areas_ch[area][0],
            n_ch=areas_ch[area][1],
            f_lp=f_lp,
            f_hp=f_hp,
            filt=filt,
        )
        data = LfpData(
            block=bhv.block,
            eyes_values=eyes_ds,
            lfp_values=lfp_ds,
            idx_start=np.array(idx_start),
        )
        output_d = os.path.normpath(output_dir)
        path = "/".join(
            [output_d]
            + ["session_struct"]
            + [area]
            + ["lfp"]
            + [str(f_lp) + "_" + str(f_hp)]
        )
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
            + "_lfp.h5"
        )
        if not os.path.exists(path):
            os.makedirs(path)
        logging.info("Saving data")
        data.to_python_hdf5("/".join([path] + [file_name]))
        logging.info("Data successfully saved")
        del data
        del lfp_ds
        gc.collect()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "continuous_path", help="Path to the continuous file (.dat)", type=Path
    )
    parser.add_argument(
        "bhv_path",
        help="Path to folder containing the bhv files",
        type=Path,
    )
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )
    parser.add_argument(
        "--event_path",
        "-e",
        default="events/Acquisition_Board-100.Rhythm Data/TTL",
        help="Output directory",
        type=Path,
    )
    parser.add_argument("--areas", "-a", nargs="*", default=None, help="area", type=str)
    parser.add_argument(
        "--start_ch", "-s", nargs="*", default=None, help="start_ch", type=int
    )
    parser.add_argument("--n_ch", "-n", nargs="*", default=None, type=int)
    parser.add_argument("--f_lp", "-l", nargs="*", default=250, type=int)
    parser.add_argument("--f_hp", "-t", nargs="*", default=3, type=int)
    parser.add_argument("--filt", "-f", nargs="*", default=True, type=bool)

    args = parser.parse_args()
    try:
        main(
            args.continuous_path,
            args.bhv_path,
            args.output_dir,
            args.event_path,
            args.areas,
            args.start_ch,
            args.n_ch,
            args.f_lp,
            args.f_hp,
            args.filt,
        )
    except FileExistsError:
        logging.error("path does not exist")
