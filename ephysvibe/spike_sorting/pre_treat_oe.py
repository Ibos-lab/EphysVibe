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
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BhvData
]:
    # reconstruct 8 bit words
    (
        full_word,
        real_strobes,
        start_trials,
        end_trials,
        bhv,
    ) = utils_oe.find_events_codes(events, bhv)
    # Select the timestamps of continuous data
    logging.info("Selecting OE samples")
    ds_samples, idx_start_time = utils_oe.select_samples(
        c_samples=c_samples,
        e_samples=events["samples"],
        fs=config.FS,
        t_before_event=config.T_EVENT,
        downsample=config.DOWNSAMPLE,
    )

    return (
        full_word,
        real_strobes,
        start_trials,
        end_trials,
        ds_samples,
        idx_start_time,
        bhv,
    )
