import h5py
import numpy as np
from pathlib import Path
import logging
from ephysvibe.structures.neuron_data import NeuronData
from typing import List, Dict, Callable, Any
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import copy


class PopulationData:
    def __init__(
        self,
        population: List[NeuronData],
        created: str = None,
        comment: str = "",
        **kwargs,
    ):
        """Initialize the class.

        This class contains information about each cluster.
        Args:
            population (List[NeuronData]):
        """
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
    def cast_neurondata(
        cls,
        path: Path,
        attr_dtype: Dict = {},
        replace_nan: Dict = {},
    ) -> NeuronData:
        """Read and cast attributes.

        Args:
            path (Path): path to NeuronData.h5 file
            attr_dtype (Dict): dictionary of attribute data types
        Returns:
            NeuronData object
        """
        neu_data = NeuronData.from_python_hdf5(path)
        if not bool(attr_dtype):
            return neu_data
        for i_name, i_dtype in zip(attr_dtype.keys(), attr_dtype.values()):

            neu_attr = getattr(neu_data, i_name)
            if bool(replace_nan) and i_name in replace_nan:
                neu_attr = np.nan_to_num(neu_attr, nan=replace_nan[i_name])
                print(replace_nan[i_name])
            neu_attr = neu_attr.astype(i_dtype)
            setattr(neu_data, i_name, neu_attr)
        return neu_data

    @classmethod
    def get_population(
        cls,
        path_list: List,
        comment: str = "",
        n_jobs: int = -1,
        **args,
    ):
        """Get the population data by reading and casting attributes from multiple files.

        Args:
            path_list (List[Path]): list of paths to NeuronData.h5 files
            attr_dtype (Dict): dictionary of attribute data types
            comment
            n_jobs (int, optional): number of jobs to run in parallel. Defaults to -1
        """
        population = Parallel(n_jobs=n_jobs)(
            delayed(cls.cast_neurondata)(path, **args) for path in tqdm(path_list)
        )
        population = PopulationData(population, comment=comment)
        return population

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data and create class object
        data = {}
        neurons = []
        with h5py.File(load_path, "r") as f:
            for key, group in zip(f.keys(), f.values()):
                if key == "0":
                    data["comment"] = group.attrs["comment"]
                    data["created"] = group.attrs["created"]
                else:
                    ineu = {}
                    ineu["date_time"] = group.attrs["date_time"]
                    ineu["subject"] = group.attrs["subject"]
                    ineu["area"] = group.attrs["area"]
                    ineu["experiment"] = group.attrs["experiment"]
                    ineu["recording"] = group.attrs["recording"]
                    ineu["cluster_group"] = group.attrs["cluster_group"]
                    for nkey, nvalue in zip(group.keys(), group.values()):
                        ineu[nkey] = np.array(nvalue)
                    neurons.append(NeuronData(**ineu))

        data["population"] = neurons
        f.close()

        return cls(**data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # Cast data
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("0")
            group.attrs["comment"] = self.comment
            group.attrs["created"] = self.created
            population = self.population
            for i_d in np.arange(len(population)) + 1:
                group = f.create_group(str(i_d))
                group.attrs["date_time"] = population[i_d - 1].__dict__.pop("date_time")
                group.attrs["subject"] = population[i_d - 1].__dict__.pop("subject")
                group.attrs["area"] = population[i_d - 1].__dict__.pop("area")
                group.attrs["experiment"] = population[i_d - 1].__dict__.pop(
                    "experiment"
                )
                group.attrs["recording"] = population[i_d - 1].__dict__.pop("recording")
                group.attrs["cluster_group"] = population[i_d - 1].__dict__.pop(
                    "cluster_group"
                )
                for key, value in zip(
                    population[i_d - 1].__dict__.keys(),
                    population[i_d - 1].__dict__.values(),
                ):
                    group.create_dataset(key, np.array(value).shape, data=value)
        f.close()

    def execute_function(
        self,
        func: Callable[..., Any],
        *args,
        n_jobs: int = -1,
        ret_df: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute a provided function with given arguments and keyword arguments.

        Args:
            func (Callable[..., Any]): The function to execute.
            *args: Variable length argument list to pass to the function.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1.
            ret_df (bool, optional): Whether to return a dataframe. Defaults to True.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            pd.DataFrame: The return value of the executed function as a DataFrame.
        """
        res = Parallel(n_jobs=n_jobs)(
            delayed(func)(neu, *args, **kwargs) for neu in tqdm(self.population)
        )
        if ret_df:
            # Check res is a list of dicts
            if isinstance(res, list) and all(isinstance(item, dict) for item in res):
                df = pd.DataFrame(res)
                return df

            raise ValueError("func must return a dictionary")
        else:
            return res

    def get_subpopulation(self, nid_select):
        subpopu = []
        for ineu in range(len(self.population)):
            idneu = self.population[ineu].get_neuron_id()
            if idneu in nid_select:
                subpopu.append(copy.deepcopy(self.population[ineu]))
        return subpopu
