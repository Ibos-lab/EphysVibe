import numpy as np

SAMPLES_COND = {
    # samples IN
    "o1_c1_in": np.arange(1, 8),
    "o1_c5_in": np.arange(8, 15),
    "o5_c1_in": np.arange(15, 22),
    "o5_c5_in": np.arange(22, 29),
}
EVENTS_B1 = {
    "start_trial": 9,
    "fixation": 8,
    "sample_on": 23,
    "sample_off": 24,
    "test_on_1": 25,
    "test_off_1": 26,
    "end_trial": 18,
}
PALETTE_B1 = {
    "o1_c1": "firebrick",
    "o1_c5": "teal",
    "o5_c1": "tomato",
    "o5_c5": "turquoise",
    "o0_c0": "grey",
}
