import argparse
from pathlib import Path
import logging
import pre_treat_oe, utils_oe, config, data_structure


logging.basicConfig(level=logging.INFO)


def main(directory, output_dir):

    # load OE data
    session, subject, date_time, areas = utils_oe.load_oe_data(directory)
    # load data
    bhv = utils_oe.load_bhv_data(directory, subject)

    # Iterate by nodes/areas
    for n, n_node in enumerate(session.recordnodes):

        # load spike data
        spike_path = (
            n_node.recordings[config.RECORDING_NUM].directory + config.KILOSORT_PATH
        )
        idx_spiketimes, spiketimes_clusters_id, cluster_info = utils_oe.load_spike_data(
            spike_path
        )
        area_cluster_info = cluster_info[cluster_info["group"] != "noise"]

        if area_cluster_info.shape[0] != 0:

            # Load continuous data and events
            continuous = n_node.recordings[config.RECORDING_NUM].continuous[0]
            events = n_node.recordings[config.RECORDING_NUM].events

            area = areas[n]

            logging.info("Area: %s" % (area))

            data = pre_treat_oe.pre_treat_oe(
                continuous,
                events,
                bhv,
                idx_spiketimes,
                area_cluster_info,
                spiketimes_clusters_id,
            )
            data_structure.save_data(
                data,
                output_dir=output_dir,
                subject=subject,
                date_time=date_time,
                area=area,
            )
        else:
            logging.info("No recordings")


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
