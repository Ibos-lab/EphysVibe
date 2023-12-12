import argparse
from pathlib import Path
import logging
import os
import numpy as np
from ...structures.bhv_data import BhvData
from ...structures.spike_data import SpikeData
from ...structures.neuron_data import NeuronData
from ...task import task_constants

logging.basicConfig(
    format="%(asctime)s | %(message)s ",
    datefmt="%d/%m/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def main(
    sp_path: Path,
    output_dir: Path = "./output",
) -> None:
    """Compute spikes by trial.

    Args:
        sp_path (Path): path to the spike data file (sp.h5).
        output_dir (Path): output directory.
    """
    if not os.path.exists(sp_path):
        logging.error("sp_path %s does not exist" % sp_path)
        raise FileExistsError
    logging.info("-- Start --")

    # define paths
    sp_path = os.path.normpath(sp_path)
    s_path = sp_path.split(os.sep)
    # load spike data
    logging.info("Loading SpikeData")
    logging.info(sp_path)
    sp_data = SpikeData.from_python_hdf5(sp_path)
    # Select info about the recording from the path
    date_time = sp_data.date_time
    area = sp_data.area
    subject = sp_data.subject
    n_exp = sp_data.experiment
    n_rec = sp_data.recording
    # load bhv data
    file_name = date_time + "_" + subject + "_e" + n_exp + "_r" + n_rec + "_bhv.h5"
    bhv_path = "/".join(s_path[:-3] + ["bhv"] + [file_name])
    bhv_path = os.path.normpath(bhv_path)
    # load bhv data
    logging.info("Loading Bhv data")
    bhv = BhvData.from_python_hdf5(bhv_path)
    # --------------------------
    code_samples = bhv.code_samples
    sp_samples = sp_data.sp_samples
    code_numbers = bhv.code_numbers
    before_trial = 1000
    iti = 1500
    next_trial = 6000
    trials_end = code_samples[
        np.where(code_numbers == task_constants.EVENTS_B1["end_trial"], True, False)
    ]
    trials_start = code_samples[:, 0]
    # --------------------------
    trials_max_duration = max(trials_end - trials_start)
    trials_max_duration = int(trials_max_duration + before_trial + iti + next_trial)
    n_trials = trials_start.shape[0]
    n_neurons = sp_samples.shape[0]
    tr_sp_data = np.full((n_trials, n_neurons, trials_max_duration), np.nan)

    for i_t in range(n_trials):
        start_trial = (trials_start[i_t] - before_trial).astype(int)
        end_trial = (trials_end[i_t] + iti + next_trial).astype(int)
        if end_trial > sp_samples.shape[1]:
            end_trial = sp_samples.shape[1]
        tr_sp_data[i_t, :, : int(end_trial - start_trial)] = sp_samples[
            :, start_trial:end_trial
        ]
    code_samples_trial = code_samples - code_samples[:, 0].reshape(-1, 1) + before_trial
    i_neuron = 0
    i_mua = 0
    for i_n, cluster in enumerate(sp_data.clusters_group):
        if cluster == "good":
            i_neuron += 1
            i_cluster = i_neuron
        elif cluster == "mua":
            i_mua += 1
            i_cluster = i_mua
        # Define structure
        neuron_data = NeuronData(
            date_time=date_time,
            subject=subject,
            area=area,
            experiment=n_exp,
            recording=n_rec,
            # -------sp-------
            sp_samples=tr_sp_data[:, i_n],
            cluster_id=np.array(sp_data.clusters_id[i_n], dtype=int),
            cluster_ch=np.array(sp_data.clusters_ch[i_n], dtype=int),
            cluster_group=str(cluster),
            cluster_number=np.array(i_cluster, dtype=int),
            cluster_array_pos=np.array(i_n, dtype=int),
            cluster_depth=sp_data.clusters_depth[i_n],
            # -------bhv-------
            block=bhv.block,
            trial_error=bhv.trial_error,
            code_samples=code_samples_trial,
            code_numbers=code_numbers,
            position=bhv.position,
            pos_code=bhv.pos_code,
            sample_id=bhv.sample_id,
            test_stimuli=bhv.test_stimuli,
            test_distractor=bhv.test_distractor,
        )
        output_d = os.path.normpath(output_dir)
        path = "/".join([output_d] + ["session_struct"] + [area] + ["neurons"])
        file_name = (
            date_time
            + "_"
            + subject
            + "_"
            + area
            + "_e"
            + n_exp
            + "_r"
            + n_rec
            + "_"
            + cluster
            + str(i_cluster)
            + "_neu.h5"
        )
        # Save neuron
        if not os.path.exists(path):
            os.makedirs(path)
        logging.info("Saving data")
        logging.info(file_name)
        neuron_data.to_python_hdf5("/".join([path] + [file_name]))
        logging.info("Data successfully saved")
        del neuron_data


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sp_path", help="Path to KS folders location", type=Path)
    parser.add_argument(
        "--output_dir", "-o", default="./output", help="Output directory", type=Path
    )

    args = parser.parse_args()
    try:
        main(args.sp_path, args.output_dir)
    except FileExistsError:
        logging.error("path does not exist")
