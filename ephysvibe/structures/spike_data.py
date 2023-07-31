import h5py
import numpy as np
from pathlib import Path
import logging


class SpikeData:
    def __init__(
        self,
        block: np.ndarray,
        code_numbers: np.ndarray,
        trial_error: np.ndarray,
        sp_samples: np.ndarray,
        clusters_id: np.ndarray,
        clusters_ch: np.ndarray,
        clustersgroup: np.ndarray,
        clusterdepth: np.ndarray,
        code_samples: np.ndarray,
        start_trials: np.ndarray,
        neuron_cond: np.ndarray = np.array([np.nan]),
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
            start_trials (np.ndarray): array of shape (trials) containing the timestamp of the start of each trial (downsampled).
        """

        # sp
        self.sp_samples = sp_samples
        self.code_samples = code_samples
        self.block = block
        self.code_numbers = code_numbers
        self.trial_error = trial_error
        self.clusters_id = clusters_id
        self.clusters_ch = clusters_ch
        self.clustersgroup = clustersgroup
        self.clusterdepth = clusterdepth
        self.neuron_cond = neuron_cond
        self.start_trials = start_trials
        self.check_shapes()

    def check_shapes(self):
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
        if self.clusters_id.shape[0] != n_neurons:
            raise ValueError(
                "clusters_id shape: %s, expected: %s"
                % (self.clusters_id.shape[0], n_neurons)
            )
        if self.clusters_ch.shape[0] != n_neurons:
            raise ValueError(
                "clusters_ch shape: %s, expected: %s"
                % (self.clusters_ch.shape[0], n_neurons)
            )
        if self.clustersgroup.shape[0] != n_neurons:
            raise ValueError(
                "clustersgroup shape: %s, expected: %s"
                % (self.clustersgroup.shape[0], n_neurons)
            )
        if self.clusterdepth.shape[0] != n_neurons:
            raise ValueError(
                "clusterdepth shape: %s, expected: %s"
                % (self.clusterdepth.shape[0], n_neurons)
            )
        if self.neuron_cond.shape[0] != n_neurons and self.neuron_cond.shape[0] != 1:
            raise ValueError(
                "neuron_cond shape: %s, expected: %s"
                % (self.neuron_cond.shape[0], n_neurons)
            )

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:

            group = f.create_group("data")
            group.create_dataset(
                "sp_samples",
                self.sp_samples.shape,
                compression="gzip",
                data=self.sp_samples,
            )
            group.create_dataset(
                "block",
                self.block.shape,
                compression="gzip",
                data=self.block,
            )
            group.create_dataset(
                "code_numbers",
                self.code_numbers.shape,
                compression="gzip",
                data=self.code_numbers,
            )
            group.create_dataset(
                "trial_error",
                self.trial_error.shape,
                compression="gzip",
                data=self.trial_error,
            )
            group.create_dataset(
                "code_samples",
                self.code_samples.shape,
                data=self.code_samples,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_id",
                self.clusters_id.shape,
                data=self.clusters_id,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_ch",
                self.clusters_ch.shape,
                data=self.clusters_ch,
                compression="gzip",
            )
            group.create_dataset(
                "clustersgroup",
                self.clustersgroup.shape,
                data=self.clustersgroup,
                compression="gzip",
            )
            group.create_dataset(
                "clusterdepth",
                self.clusterdepth.shape,
                data=self.clusterdepth,
                compression="gzip",
            )
            group.create_dataset(
                "neuron_cond",
                self.neuron_cond.shape,
                data=self.neuron_cond,
                compression="gzip",
            )
            group.create_dataset(
                "start_trials",
                self.start_trials.shape,
                data=self.start_trials,
                compression="gzip",
            )
        f.close()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["data"]
            sp_samples = group["sp_samples"][:]
            block = group["block"][:]
            code_samples = group["code_samples"][:]
            code_numbers = group["code_numbers"][:]
            trial_error = group["trial_error"][:]
            clusters_id = group["clusters_id"][:]
            clusters_ch = group["clusters_ch"][:]
            clustersgroup = np.array(group["clustersgroup"], dtype=str)
            clusterdepth = group["clusterdepth"][:]
            try:
                neuron_cond = group["neuron_cond"][:]
            except:
                neuron_cond = np.array([np.nan])
            try:
                start_trials = group["start_trials"][:]
            except:
                start_trials = np.array([np.nan])
        # create class object and return
        trials_data = {
            "sp_samples": sp_samples,
            "block": block,
            "code_samples": code_samples,
            "code_numbers": code_numbers,
            "trial_error": trial_error,
            "clusters_id": clusters_id,
            "clusters_ch": clusters_ch,
            "clustersgroup": clustersgroup,
            "clusterdepth": clusterdepth,
            "neuron_cond": neuron_cond,
            "start_trials": start_trials,
        }
        return cls(**trials_data)

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
