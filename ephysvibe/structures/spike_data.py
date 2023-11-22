import h5py
import numpy as np
from pathlib import Path
import logging


class SpikeData:
    def __init__(
        self,
        sp_samples: np.ndarray,
        clusters_id: np.ndarray,
        clusters_ch: np.ndarray,
        clustersgroup: np.ndarray,
        clusterdepth: np.ndarray,
    ):
        """Initialize the class.

        Args:
            sp_samples (np.ndarray): array of shape (trials x neurons x time) containing the number of spikes at each ms.
            clusters_id (np.ndarray): array of shape (neurons,1)
            clusters_ch (np.ndarray): array of shape (neurons,1)
            clustersgroup (np.ndarray): array of shape (neurons,1) containing "good" when is a neuron or "mua" when is
                                        multi unit activity.
            clusterdepth (np.ndarray): array of shape (neurons,1) containing the de depth of each neuron/mua.
        """
        self.sp_samples = sp_samples
        self.clusters_id = clusters_id
        self.clusters_ch = clusters_ch
        self.clustersgroup = clustersgroup
        self.clusterdepth = clusterdepth

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
        f.close()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["data"]
            sp_samples = group["sp_samples"][:]
            clusters_id = group["clusters_id"][:]
            clusters_ch = group["clusters_ch"][:]
            clustersgroup = np.array(group["clustersgroup"], dtype=str)
            clusterdepth = group["clusterdepth"][:]

        # create class object and return
        trials_data = {
            "sp_samples": sp_samples,
            "clusters_id": clusters_id,
            "clusters_ch": clusters_ch,
            "clustersgroup": clustersgroup,
            "clusterdepth": clusterdepth,
        }
        return cls(**trials_data)
