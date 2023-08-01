import h5py
import numpy as np
from pathlib import Path
import logging


class LfpData:
    def __init__(
        self,
        block: np.ndarray,
        eyes_values: np.ndarray,
        lfp_values: np.ndarray,
        ds_samples: np.ndarray,
        start_trials: np.ndarray = np.array([np.nan]),
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
        self.block = block
        self.eyes_values = eyes_values
        self.lfp_values = lfp_values
        self.start_trials = start_trials
        self.ds_samples = ds_samples
        self.check_shapes()

    def check_shapes(self):
        n_ch, _ = self.lfp_values.shape

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            group.create_dataset(
                "block",
                self.block.shape,
                compression="gzip",
                data=self.block,
            )
            group.create_dataset(
                "start_trials",
                self.start_trials.shape,
                data=self.start_trials,
                compression="gzip",
            )
            group.create_dataset(
                "eyes_values",
                self.eyes_values.shape,
                data=self.eyes_values,
                compression="gzip",
            )
            group.create_dataset(
                "lfp_values",
                self.lfp_values.shape,
                data=self.lfp_values,
                compression="gzip",
            )
            group.create_dataset(
                "ds_samples",
                self.ds_samples.shape,
                data=self.ds_samples,
                compression="gzip",
            )

        f.close()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["data"]

            block = group["block"][:]
            eyes_values = group["eyes_values"][:]
            start_trials = group["start_trials"][:]
            ds_samples = group["ds_samples"][:]
            lfp_values = group["lfp_values"][:]

        # create class object and return
        trials_data = {
            "block": block,
            "eyes_values": eyes_values,
            "lfp_values": lfp_values,
            "start_trials": start_trials,
            "ds_samples": ds_samples,
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
