from ephysvibe.structures.trials_data import TrialsData
import numpy as np
from pathlib import Path
import cv2
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import PySimpleGUI as sg
import tkinter as tk

layout = [[sg.Text("Hello from PySimpleGUI")], [sg.Button("OK")]]

# Create the window
window = sg.Window("Demo", layout)

# selected by the user in the front
datetime = "2022-11-28_10-23-27"
area = "lip"
subject = "Riesling"
experiment = "e1"
recording = "r1"
data_path = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/session_struct/"
path_img = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/"


# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "OK" or event == sg.WIN_CLOSED:
        break

window.close()


# in the back
# define paths and read data/img
path_id = [datetime + "_" + subject + "_" + area + "_" + experiment + "_" + recording]
c_data_path = "/".join(
    [os.path.normpath(data_path)] + [subject] + [area] + [path_id + ".h5"]
)

data = TrialsData.from_python_hdf5(c_data_path)
data.clustersgroup
data.clusters_id
# I can do a for loop or see if I can list all the imgs
c_path_img = "/".join(
    [os.path.normpath(path_img)]
    + [area]
    + [path_id + "_" + c_group + "_" + n_group + ".jpg"]
)
img = cv2.imread(c_path_img)
cv2.imshow("ff", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

val = input("enter value")
print(val)
