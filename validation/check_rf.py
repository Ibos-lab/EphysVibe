from ephysvibe.structures.trials_data import TrialsData
import numpy as np
from pathlib import Path
import cv2
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

import tkinter as tk
from tkinter import PhotoImage, filedialog, Label
from PIL import ImageTk, Image


def openfilename():

    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')

    return filename


def save():
    print("save")


def load_img():
    path_img = openfilename()
    # img = Image.open(path_img)
    print(path_img)
    image = PhotoImage(file=path_img)
    # create a label
    panel = Label(window, image=image)

    # set the image as img
    panel.image = image
    panel.grid(row=2)


# selected by the user in the front
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

# Create the window
window = tk.Tk()
greeting = tk.Label(text="Hello, Tkinter")
exit_button = tk.Button(
    text="Exit",
    width=25,
    height=5,
    bg="blue",
    fg="white",
    command=window.quit,
)
exit_button.place(x=75, y=75)
exit_button.pack()
save_button = tk.Button(
    text="Save",
    width=25,
    height=5,
    bg="blue",
    fg="white",
    command=save,
)
save_button.place(x=75, y=100)
save_button.pack()

text_box = tk.Text()
text_box.pack()
load_img_button = tk.Button(
    text="load_img",
    width=25,
    height=5,
    bg="blue",
    fg="white",
    command=load_img(),
)
load_img_button.place(x=50, y=100)
load_img_button.pack()

window.mainloop()


# # Create an event loop
# while True:
#     event, values = window.read()
#     # End program if user closes window or
#     # presses the OK button
#     if event == "OK" or event == sg.WIN_CLOSED:
#         break

# window.close()


# # in the back


# data = TrialsData.from_python_hdf5(c_data_path)
# data.clustersgroup

# # I can do a for loop or see if I can list all the imgs
# c_path_img = "/".join(
#     [os.path.normpath(path_img)]
#     + [area]
#     + [path_id + "_" + c_group + "_" + n_group + ".jpg"]
# )
# img = cv2.imread(c_path_img)
# cv2.imshow("ff", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# val = input("enter value")
# print(val)
