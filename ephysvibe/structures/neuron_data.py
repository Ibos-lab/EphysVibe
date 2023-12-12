import h5py
import numpy as np
from pathlib import Path
import logging


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

    def _check_shapes(self):
        n_trials, n_neurons, n_ts = self.sp_samples.shape
        n_codes = self.code_numbers.shape[1]
        # Check if the number of trials is the same than in bhv
        if self.block.shape[0] != n_trials:
            logging.warning(
                "bhv n_trials (%s) != sp n_trials (%s)"
                % (self.block.shape[0], n_trials)
            )
        if self.code_samples.shape != (n_trials, n_codes):
            raise ValueError(
                "code_samples shape: (%s, %s), expected: (%s, %s)"
                % (self.code_samples.shape, n_trials, n_codes)
            )
        if self.cluster_id.shape[0] != n_neurons:
            raise ValueError(
                "clusters_id shape: %s, expected: %s"
                % (self.clusters_id.shape[0], n_neurons)
            )
        if self.clusters_ch.shape[0] != n_neurons:
            raise ValueError(
                "clusters_ch shape: %s, expected: %s"
                % (self.clusters_ch.shape[0], n_neurons)
            )

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
