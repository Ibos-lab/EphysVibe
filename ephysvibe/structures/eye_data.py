import h5py
import numpy as np
from pathlib import Path
import logging
import dask.array as da


class EyeData:
    def __init__(
        self,
        # date_time: str,
        # subject: str,
        # experiment: str,
        # recording: str,
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
        eye: np.ndarray,
        eye_ml: np.ndarray,
    ):
        """Initialize the class.

        Args:
            sp_samples (np.ndarray): array of shape (trials x neurons x time) containing the number of spikes at each ms.
            eyes_values (np.ndarray): array of shape (trials x ch x time) containing the position of the eye (ch1=x,ch2=y)
                                    and the dilation of the eye (ch3) at each ms.
            lfp_values (np.ndarray): array of shape (trials x ch x time) containing the lfp values at each ms.
            clusters_id (np.ndarray): array of shape (neurons,1)
            clusters_ch (np.ndarray): array of shape (neurons,1)
            clustersgroup (np.ndarray): array of shape (neurons,1) containing "good" when is a neuron or "mua" when is
                                        multi unit activity.
            clusterdepth (np.ndarray): array of shape (neurons,1) containing the de depth of each neuron/mua.
            code_samples (np.ndarray): array of shape (trials x ncodes) containing the time at which the event occurred. [ms].
            idx_start (np.ndarray): array of shape (trials) containing the timestamp of the start of each trial (downsampled).
        """
        # self.date_time = date_time
        # self.subject = subject
        # self.experiment = experiment
        # self.recording = recording
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
        # -----eye--------
        self.eye = eye
        self.eye_ml = eye_ml

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            # group.attrs["date_time"] = self.__dict__.pop("date_time")
            # group.attrs["subject"] = self.__dict__.pop("subject")
            # group.attrs["area"] = self.__dict__.pop("area")
            # group.attrs["experiment"] = self.__dict__.pop("experiment")
            # group.attrs["recording"] = self.__dict__.pop("recording")
            # group.attrs["cluster_group"] = self.__dict__.pop("cluster_group")

            for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
                group.create_dataset(key, value.shape, data=value)
        f.close()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        eye_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]

            for key, value in zip(group.keys(), group.values()):
                eye_data[key] = np.array(value)
        f.close()
        return cls(**eye_data)

    @staticmethod
    def indep_roll(arr: np.ndarray, shifts: np.ndarray, axis: int = 1) -> np.ndarray:
        """Apply an independent roll for each dimensions of a single axis.
        Args:
            arr (np.ndarray): Array of any shape.
            shifts (np.ndarray): How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
            axis (int, optional): Axis along which elements are shifted. Defaults to 1.

        Returns:
            np.ndarray: shifted array.
        """
        arr = np.swapaxes(arr, axis, -1)
        all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]
        # Convert to a positive shift
        shifts[shifts < 0] += arr.shape[-1]
        all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]
        result = arr[tuple(all_idcs)]
        arr = np.swapaxes(result, -1, axis)
        return arr
