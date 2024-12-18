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

CODES_AND_POS = {
    # code: [[MonkeyLogic axis (variable)], [plot axis], [angle]]
    "127": [[10, 0], [1, 2], [0]],
    "126": [[7, 7], [0, 2], [45]],
    "125": [[0, 10], [0, 1], [90]],
    "124": [[-7, 7], [0, 0], [135]],
    "123": [[-10, 0], [1, 0], [180]],
    "122": [[-7, -7], [2, 0], [225]],
    "121": [[0, -10], [2, 1], [270]],
    "120": [[7, -7], [2, 2], [315]],
}
