"""Configuration file."""
DEPTH = 10
N_EYES_CH = 3


HP_FC = 3  # cutoff freq [Hz]
LP_FC = 250  # cutoff freq [Hz]
FS = 30000  # sample freq [Hz]
HP_ORDER = 6  # filter order [1]
LP_ORDER = 6  # filter order [1]

DOWNSAMPLE = 30  # [1]
T_EVENT = 10  # time before the start of the first event [s]

START_CODE = 9
END_CODE = 18
KILOSORT_FOLDER_NAME = "kilosor*"
