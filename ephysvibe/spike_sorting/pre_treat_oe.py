import logging
from . import utils_oe, config
import numpy as np
from typing import List, Dict, Tuple
from ..structures.bhv_data import BhvData
from pathlib import Path


def pre_treat_oe(
    events: Dict,
    bhv: BhvData,
    c_samples: np.ndarray,
    areas_ch: Dict,
    total_ch: int,
    continuous_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # reconstruct 8 bit words
    (
        full_word,
        real_strobes,
        start_trials,
        end_trials,
    ) = utils_oe.find_events_codes(events, bhv)
    # Select the timestamps of continuous data
    logging.info("Selecting OE samples")
    ds_samples, start_time = utils_oe.select_samples(
        c_samples=c_samples,
        e_samples=events["samples"],
        fs=config.FS,
        t_before_event=config.T_EVENT,
        downsample=config.DOWNSAMPLE,
    )
    # check if eyes
    start_ch, n_eyes = areas_ch.pop("eyes", False)
    if n_eyes:
        eyes_ds = utils_oe.load_eyes(
            continuous_path,
            shape_0=len(c_samples),
            shape_1=total_ch,
            start_ch=start_ch,
            n_eyes=n_eyes,
            start_time=start_time,
        )
    return (
        full_word,
        real_strobes,
        start_trials,
        end_trials,
        ds_samples,
        start_time,
        eyes_ds,
        areas_ch,
    )
