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
        self.sp_samples = sp_samples
        self.blocks = blocks
        self.code_numbers = code_numbers
        self.code_samples = code_samples
        self.eyes_values = eyes_values
        self.lfp_values = lfp_values
        self.samples = samples
        self.clusters_id = clusters_id
        self.clusters_ch = clusters_ch
        self.clustersgroup = clustersgroup
        self.clusterdepth = clusterdepth
        # self.check_shapes()

    def _check_shapes(self):
        n_trials, n_neurons, n_ts = self.sp_samples.shape
        if self.blocks.shape != (n_trials):
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )
        if self.blocks.shape != (n_trials):
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )
        if self.blocks.shape != (n_trials):
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )
        if self.blocks.shape != (n_trials):
            raise ValueError(
                "sp_samples and blocks must have the same number of trials (%d != %d)"
                % (n_trials, len(self.blocks))
            )

    @classmethod
    def to_python_hdf5(cls, save_path: Path):
        """Save data in hdf5 format."""
        # load the data
        with h5py.File(save_path, "w") as f:
            f.create_dataset(
                "arr", cls.sp_samples.shape, compression="gzip", data=cls.sp_samples
            )
            f.close()

    # @classmethod
    # def from_python_hdf5(cls, load_path: Path):
    #     """Load data from a file in hdf5 format from MatLab."""
    #     # load the data
    #     amdict = super().from_matlab_hdf5(load_path)
    #     with h5py.File(load_path, "r") as data:
    #         #  get sensors data
    #         dmc = data["dMc"]
    #         time = np.array(dmc["time"]).reshape(-1)
    #         n_gamma = np.array(dmc["nGamma"]).transpose()
    #         omega = np.array(dmc["omega"]).transpose()
    #         length = np.array(dmc["length"]).reshape(-1)
    #         state = np.array(dmc["state"]).reshape(-1)
    #     # create class object and return
    #     calibrated_data = {
    #         **amdict,
    #         "time": time,
    #         "n_gamma": n_gamma,
    #         "omega": omega,
    #         "length": length,
    #         "state": state,
    #     }
    #     return cls(**calibrated_data)
