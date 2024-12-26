"""Execute main function of the module plot_trials."""

from typing import Dict
from . import _pipeline
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import numpy as np


def plot_trials(paths: Dict, params: Dict, **kwargs):
    print("start plot trials")
    path_list = glob.glob(paths["input"])
    if "hydra" in params and params["hydra"]:
        output_dir = os.getcwd()
    elif "output_dir" in params:
        output_dir = params["output_dir"]
    else:
        output_dir = "./"
    Parallel(n_jobs=-1)(
        delayed(_pipeline.prepare_and_plot)(
            neupath=path,
            format=params["format"],
            percentile=params["percentile"],
            cerotr=params["cerotr"],
            b=params["b"],
            output_dir=output_dir,
        )
        for path in tqdm(path_list)
    )
    print(f"Current working directory : {os.getcwd()}")
