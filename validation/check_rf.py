from ephysvibe.structures.trials_data import TrialsData
import numpy as np
from pathlib import Path
import cv2
import sys
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


path_img = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/plots2/lip/2022-11-28_10-23-27_Riesling_lip_e1_r1_mua_1.jpg"
img = cv2.imread(path_img)
cv2.imshow("ff", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

val = input("enter value")
print(val)
