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
    "bar_hold": 7,
    "fix_spot_on": 35,
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
    "end_trial": 18,
    "bar_release": 4,
    "fix_spot_off": 36,
    "reward": 96,
    "fix_break": 97,
}
EVENTS_B1_SHORT = {
    "start_trial": "trst",
    "bar_hold": "barh",
    "fix_spot_on": "fspoton",
    "fixation": "fixst",
    "sample_on": "son",
    "sample_off": "soff",
    "test_on_1": "t1on",
    "test_off_1": "t1off",
    "test_on_2": "t2on",
    "test_off_2": "t2off",
    "test_on_3": "t3on",
    "test_off_3": "t3off",
    "test_on_4": "t4on",
    "test_off_4": "t4off",
    "test_on_5": "t5on",
    "test_off_5": "t5off",
    "end_trial": "trend",
    "bar_release": "barr",
    "fix_spot_off": "fspotoff",
    "reward": "rwd",
    "fix_break": "fixbrk",
}
PALETTE_B1 = {
    "11": "firebrick",  # o1_c1
    "15": "teal",  # o1_c5
    "51": "tomato",  # o5_c1
    "55": "lightseagreen",  # o5_c5
    "0": "grey",  # o0_c0
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
