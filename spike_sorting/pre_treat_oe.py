import logging
import data_structure
import utils_oe
import config


def pre_treat_oe(
    continuous,
    events,
    bhv,
    idx_spiketimes,
    area_cluster_info,
    spiketimes_clusters_id,
    eyes=False,
):

    # Select the timestamps of continuous data
    logging.info("Selecting OE timestamps")
    filtered_timestamps, start_time, spiketimes = utils_oe.select_timestamps(
        c_timestamps=continuous.timestamps,
        e_timestamps=events.timestamp,
        idx_spiketimes=idx_spiketimes,
        fs=config.FS,
        t_before_event=config.T_EVENT,
        downsample=config.DOWNSAMPLE,
    )
    # reconstruct 8 bit words
    (
        full_word,
        real_strobes,
        bl_start_trials,
        n_blocks,
        bhv_trials,
    ) = utils_oe.find_events_codes(events, bhv)

    logging.info("Computing LFPs")
    LFP_ds, eyes_ds = utils_oe.compute_lfp(continuous.samples, start_time, eyes=eyes)

    # split in blocks
    for n, (start_trials, i_block) in enumerate(zip(bl_start_trials, n_blocks)):
        logging.info("Block: %d" % (i_block))

        (
            times,
            code_numbers,
            code_times,
            eyes_sample,
            lfp_sample,
            timestamps,
        ) = data_structure.sort_data_trial(
            clusters=area_cluster_info,
            spiketimes=spiketimes,
            start_trials=start_trials,
            real_strobes=real_strobes,
            filtered_timestamps=filtered_timestamps,
            spiketimes_clusters_id=spiketimes_clusters_id,
            full_word=full_word,
            LFP_ds=LFP_ds,
            eyes_ds=eyes_ds,
        )

        data = data_structure.build_data_structure(
            clusters=area_cluster_info,
            times=times,
            code_numbers=code_numbers,
            code_times=code_times,
            eyes_sample=eyes_sample,
            lfp_sample=lfp_sample,
            timestamps=timestamps,
            block=i_block,
            bhv_trial=bhv_trials[n],
        )

        return data

    logging.info("pre_treat_oe successfully run")
