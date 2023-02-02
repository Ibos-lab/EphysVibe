import h5py
import numpy as np
from pathlib import Path


class TrialsData:
    def __init__(
        self,
        sp_samples,
        blocks,
        code_numbers,
        code_samples,
        eyes_values,
        lfp_values,
        samples,
        clusters_id,
        clusters_ch,
        clustersgroup,
        clusterdepth,
    ):

        # sp
        self.sp_samples = sp_samples
        self.eyes_values = eyes_values
        self.lfp_values = lfp_values
        # bhv
        self.blocks = blocks
        self.code_numbers = code_numbers
        self.code_samples = code_samples
        self.samples = samples
        self.clusters_id = clusters_id
        self.clusters_ch = clusters_ch
        self.clustersgroup = clustersgroup
        self.clusterdepth = clusterdepth
        self._check_shapes()

    def _check_shapes(self):
        n_trials, n_neurons, n_ts = self.sp_samples.shape
        n_codes = self.code_numbers.shape[1]
        if self.blocks.shape != (n_trials,):
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )
        if self.code_numbers.shape != (n_trials, n_codes):
            raise ValueError(
                "Expected shape: (%s,%s), but blocks has shape %d"
                % (n_trials, len(self.blocks))
            )
        if self.code_samples.shape != (n_trials, n_codes):
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )
        if self.eyes_values.shape[0] != n_trials or self.eyes_values.shape[2] != n_ts:
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        f = h5py.File(save_path, "w")
        group = f.create_group("sp")
        group.create_dataset(
            "sp_samples",
            self.sp_samples.shape,
            compression="gzip",
            data=self.sp_samples,
        )
        group.create_dataset(
            "blocks", self.blocks.shape, data=self.blocks, compression="gzip"
        )
        group.create_dataset(
            "code_numbers",
            self.code_numbers.shape,
            data=self.code_numbers,
            compression="gzip",
        )
        group.create_dataset(
            "code_samples",
            self.code_samples.shape,
            data=self.code_samples,
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
            "samples", self.samples.shape, data=self.samples, compression="gzip"
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
        # load the data

        with h5py.File(load_path, "r") as data:
            #  get data
            group = data["sp"]
            sp_samples = group["sp_samples"][:]
            blocks = group["blocks"][:]
            code_numbers = group["code_numbers"][:]
            code_samples = group["code_samples"][:]
            eyes_values = group["eyes_values"][:]
            lfp_values = group["lfp_values"][:]
            samples = group["samples"][:]
            clusters_id = group["clusters_id"][:]
            clusters_ch = group["clusters_ch"][:]
            clustersgroup = group["clustersgroup"][:]
            clusterdepth = group["clusterdepth"][:]
        # create class object and return
        data = {
            "sp_samples": sp_samples,
            "blocks": blocks,
            "code_numbers": code_numbers,
            "code_samples": code_samples,
            "eyes_values": eyes_values,
            "lfp_values": lfp_values,
            "samples": samples,
            "clusters_id": clusters_id,
            "clusters_ch": clusters_ch,
            "clustersgroup": clustersgroup,
            "clusterdepth": clusterdepth,
        }
        return cls(**data)
