import h5py
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
from ephysvibe.task import task_constants
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import align_trials
from ephysvibe.stats import smetrics


class NeuronData:
    def __init__(
        self,
        date_time: str,
        subject: str,
        area: str,
        experiment: str,
        recording: str,
        # --------sp-------
        sp_samples: np.ndarray,
        cluster_id: int,
        cluster_ch: int,
        cluster_group: str,
        cluster_number: int,
        cluster_array_pos: int,
        cluster_depth: int,
        # -------bhv-------
        block: np.ndarray,
        trial_error: np.ndarray,
        code_samples: np.ndarray,
        code_numbers: np.ndarray,
        position: np.ndarray,
        pos_code: np.ndarray,
        sample_id: np.ndarray,
        test_stimuli: np.ndarray,
        test_distractor: np.ndarray,
        **kwargs,
    ):
        """Initialize the class.

        This class contains information about each cluster.

        Args:
            date_time (str): date and time of the recording session
            subject (str):  name of the subject
            area (str): area recorded
            experiment (str): experiment number
            recording (str): recording number
            ------ sp ---------
            sp_samples (np.ndarray): array of shape (trials x time) containing the number of spikes at each ms in each trial.
            cluster_id (int): kilosort cluster ID.
            cluster_ch (int): electrode channel that recorded the activity of the cluster.
            cluster_group (str): "good" if it is a neuron or "mua" if it is a multi unit activity.
            cluster_number (int): number of good or mua.
            cluster_array_pos (int): position of the cluster in SpikeDate.sp_samples.
            cluster_depth (int): depth of the cluster.
            ------ bhv ---------
            block (np.ndarray): array of shape (trials) containing:
                                - 1 when is a DMTS trial
                                - 2 when is a saccade task trial
            trial_error (np.ndarray): array of shape (trials) containing:
                                - 0 when is a correct trial
                                - n != 0 when is an incorrect trial. Each number correspond to different errors
            code_samples (np.ndarray): array of shape (trials, events) containing the timestamp of the events
                                        (timestamps correspond to sp_sample index).
            code_numbers (np.ndarray): array of shape (trials, events) containing the codes of the events.
            position (np.ndarray): array of shape (trials, 2) containing the position of the stimulus.
            pos_code (np.ndarray): array of shape (trials) containing the position code of the stimulus.
                                    - for block 1: 1 is for 'in', -1 is for 'out' the receptive field
                                    - for block 2: codes from 120 to 127 corresponding to the 8 target positions.
            sample_id (np.ndarray): array of shape (trials) containing the sample presented in each trial of block 1:
                                    - 0: neutral sample
                                    - 11: orientation 1, color 1
                                    - 51: orientation 5, color 1
                                    - 15: orientation 1, color 5
                                    - 55: orientation 5, color 5
            test_stimuli (np.ndarray): array of shape (trials,n_test_stimuli) containing the id of the test stimuli.
                                    As in sample_id, first number correspond to orientation and second to color.
            test_distractor (np.ndarray): array of shape (trials,n_test_stimuli) containing the id of the test distractor.
                                    As in sample_id, first number correspond to orientation and second to color.
        """
        self.date_time = date_time
        self.subject = subject
        self.area = area
        self.experiment = experiment
        self.recording = recording
        # --------sp-------
        self.sp_samples = sp_samples
        self.cluster_id = cluster_id
        self.cluster_ch = cluster_ch
        self.cluster_group = cluster_group
        self.cluster_number = cluster_number
        self.cluster_array_pos = cluster_array_pos
        self.cluster_depth = cluster_depth
        # -------bhv-------
        self.block = block
        self.trial_error = trial_error
        self.code_samples = code_samples
        self.code_numbers = code_numbers
        self.position = position
        self.pos_code = pos_code
        self.sample_id = sample_id
        self.test_stimuli = test_stimuli
        self.test_distractor = test_distractor
        for key in kwargs:
            setattr(self, key, kwargs[key])
        # self._check_shapes()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data and create class object
        bhv_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]
            bhv_data["date_time"] = group.attrs["date_time"]
            bhv_data["subject"] = group.attrs["subject"]
            bhv_data["area"] = group.attrs["area"]
            bhv_data["experiment"] = group.attrs["experiment"]
            bhv_data["recording"] = group.attrs["recording"]
            bhv_data["cluster_group"] = group.attrs["cluster_group"]
            for key, value in zip(group.keys(), group.values()):
                bhv_data[key] = np.array(value)
        f.close()
        return cls(**bhv_data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            group.attrs["date_time"] = self.__dict__.pop("date_time")
            group.attrs["subject"] = self.__dict__.pop("subject")
            group.attrs["area"] = self.__dict__.pop("area")
            group.attrs["experiment"] = self.__dict__.pop("experiment")
            group.attrs["recording"] = self.__dict__.pop("recording")
            group.attrs["cluster_group"] = self.__dict__.pop("cluster_group")

            for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
                group.create_dataset(key, value.shape, data=value)
        f.close()

    def get_neuron_id(self):
        nid = (
            self.cluster_group
            + str(int(self.cluster_number))
            + self.area.upper()
            + self.date_time
            + self.subject
        )
        return nid

    def align_on(
        self,
        select_block: int = 1,
        event: str = "sample_on",
        time_before: int = 500,
        error_type: int = 0,
        select_pos="in",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Check select_pos value
        if isinstance(select_pos, str):
            select_pos = (
                1 if select_pos == "in" else -1 if select_pos == "out" else np.nan
            )
            if np.isnan(select_pos):
                raise KeyError
        # Select trials with the selected error and block
        mask = np.where(
            np.logical_and(
                self.pos_code == select_pos,
                np.logical_and(
                    self.trial_error == error_type, self.block == select_block
                ),
            ),
            True,
            False,
        )
        sp_samples_m = self.sp_samples[mask]
        # Code corresponding to the event
        if select_block == 1:
            code = task_constants.EVENTS_B1[event]
        elif select_block == 2:
            code = task_constants.EVENTS_B2[event]
        else:
            return
        # Find the codes in the code_numbers matrix
        code_mask = np.where(self.code_numbers[mask] == code, True, False)
        # Wether the event ocured in each trial
        trials_mask = np.any(code_mask, axis=1)
        # Select the sample when the event ocurred
        shifts = self.code_samples[mask][code_mask]
        shifts = (shifts - time_before).astype(int)
        # align sp
        align_sp = align_trials.indep_roll(
            arr=sp_samples_m[trials_mask], shifts=-shifts, axis=1
        )
        # Create mask for selecting the trials from the original matrix size
        tr = np.arange(self.sp_samples.shape[0])
        complete_mask = np.isin(tr, tr[mask][trials_mask])

        return (align_sp, complete_mask)

    def edit_attributes(self, new_values: Dict):
        for attr_name, attr_value in zip(new_values.keys(), new_values.values()):
            setattr(self, attr_name, attr_value)

    def get_sp_per_sec(self):
        time_before = 100
        res = {}
        res["nid"] = self.get_neuron_id()
        for in_out in ["in", "out"]:
            pos_io = 1 if in_out == "in" else -1
            sp, mask = self.align_on(
                select_block=1,
                select_pos=pos_io,
                event="sample_on",
                time_before=time_before,
                error_type=0,
            )
            for nn_n in ["NN", "N"]:
                if nn_n == "NN":
                    sample_mask = self.sample_id[mask] != 0
                else:
                    sample_mask = self.sample_id[mask] == 0
                # Average fr across time
                sp_avg = firing_rate.moving_average(sp[:, :1500], win=100, step=1)
                frsignal = np.mean(
                    sp_avg[sample_mask, time_before : time_before + 450], axis=0
                )
                res_fr = smetrics.compute_fr(frsignal=frsignal)
                res[in_out + "_mean_fr_sample_" + nn_n] = res_fr["mean_fr"]
                res[in_out + "_lat_max_fr_sample_" + nn_n] = res_fr["lat_max_fr"]
                res[in_out + "_mean_max_fr_sample_" + nn_n] = res_fr["mean_max_fr"]

                frsignal = np.mean(
                    sp_avg[sample_mask, time_before + 450 : time_before + 850], axis=0
                )
                res_fr = smetrics.compute_fr(frsignal=frsignal)
                res[in_out + "_mean_fr_delay_" + nn_n] = res_fr["mean_fr"]
                res[in_out + "_lat_max_fr_delay_" + nn_n] = res_fr["lat_max_fr"]
                res[in_out + "_mean_max_fr_delay_" + nn_n] = res_fr["mean_max_fr"]

        return res

    def plot_sp_b1(self):
        # TODO
        return
