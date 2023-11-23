import h5py
import numpy as np
from pathlib import Path
import logging


class NeuronData:
    def __init__(
        self,
        id: str,
        date_time: str,
        # --------sp-------
        sp_samples: np.ndarray,
        cluster_id: int,
        cluster_ch: int,
        cluster_group: str,
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

        """
        # --------sp-------
        self.sp_samples = sp_samples
        self.cluster_id = cluster_id
        self.cluster_ch = cluster_ch
        self.cluster_group = cluster_group
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
        self._check_shapes()

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
            for key, value in zip(group.keys(), group.values()):
                bhv_data[key] = value[:]
        f.close()
        return cls(**bhv_data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
                group.create_dataset(key, value.shape, data=value)
        f.close()

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
