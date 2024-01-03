# SAMPLES_COND = {
#     # samples IN
#     "o1_c1_in": np.arange(1, 8),
#     "o1_c5_in": np.arange(8, 15),
#     "o5_c1_in": np.arange(15, 22),
#     "o5_c5_in": np.arange(22, 29),
#     "o1_c1_out": np.arange(1, 8) + 28,
#     "o1_c5_out": np.arange(8, 15) + 28,
#     "o5_c1_out": np.arange(15, 22) + 28,
#     "o5_c5_out": np.arange(22, 29) + 28,
# }
EVENTS_B1 = {
    "start_trial": 9,
    "fix_on": 35,
    "fixation": 8,
    "sample_on": 23,
    "sample_off": 24,
    "test_on_1": 25,
    "test_off_1": 26,
    "test_on_2": 27,
    "test_off_2": 28,
    "test_on_3": 29,
    "test_off_3": 30,
    "test_on_4": 31,
    "test_off_4": 32,
    "test_on_5": 33,
    "test_off_5": 34,
    "fix_off": 36,
    "end_trial": 18,
}
PALETTE_B1 = {
    "o1_c1": "firebrick",
    "o1_c5": "teal",
    "o5_c1": "tomato",
    "o5_c5": "lightseagreen",
    "o0_c0": "grey",
}

EVENTS_B2 = {
    "start_trial": 9,
    "target_on": 37,
    "target_off": 38,
    "fix_spot_off": 36,
    "eye_in_target": 10,
    "correct_response": 40,
    "end_trial": 18,
}
