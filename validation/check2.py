from ephysvibe.structures.trials_data import TrialsData
import numpy as np
from pathlib import Path
import cv2
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# selected by the user
datetime = "2022-11-28_10-23-27"
area = "lip"
subject = "Riesling"
experiment = "e1"
recording = "r1"
data_path = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/"
path_img = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/"

# define paths and read data/img
path_id = datetime + "_" + subject + "_" + area + "_" + experiment + "_" + recording
c_data_path = "/".join(
    [os.path.normpath(data_path)] + [subject] + [area] + [path_id + ".h5"]
)
