import logging
from . import utils_oe, config


def pre_treat_oe(events, bhv, c_samples, areas_data, s_path, shape_0):
    # reconstruct 8 bit words
    (
        full_word,
        real_strobes,
        start_trials,
        blocks,
        dict_bhv,
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
    n_eyes = areas_data["areas"].pop("eyes", False)
    if n_eyes:
        eyes_ds = utils_oe.load_eyes(
            s_path, shape_0=shape_0, shape_1=n_eyes, start_time=start_time
        )
    return (
        full_word,
        real_strobes,
        start_trials,
        blocks,
        dict_bhv,
        ds_samples,
        start_time,
        eyes_ds,
        areas_data,
    )
