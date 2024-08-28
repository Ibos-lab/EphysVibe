import h5py
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Callable, Any
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import copy


class Results:
    def __init__(
        self,
        pipeline: str,
        population: Path,
        created: str = None,
        comment: str = "",
        **kwargs,
    ):
        """Initialize the class."""
        self.pipeline = pipeline
        self.population = population
        self.comment = comment
        if created is None:
            self.__created = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        else:
            self.__created = created

        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def created(self):
        return self.__created

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format and return a class instance."""
        data = {}
        with h5py.File(load_path, "r") as f:
            for key, group in zip(f.keys(), f.values()):
                for gkey, gvalue in zip(group.keys(), group.values()):
                    data[gkey] = np.array(gvalue)
                for attr_key, attr_value in group.attrs.items():
                    data[attr_key] = attr_value
        return cls(**data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        with h5py.File(save_path, "w") as f:
            group = f.create_group("0")
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    group.create_dataset(key, value.shape, data=value)
                elif isinstance(value, Dict):
                    for dkey, dvalue in value.items():
                        group.create_dataset(dkey, np.array(dvalue).shape, data=dvalue)
                else:
                    group.attrs[key] = value
