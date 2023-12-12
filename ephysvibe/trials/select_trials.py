import numpy as np


def select_trials_block(sp_py, n_block):
    # Select trials in a block
    trials_idx = np.where(sp_py["blocks"] == n_block)[0]
    print("Number of trials in block %d: %d" % (n_block, len(trials_idx)))
    return trials_idx


def select_correct_trials(bhv_py, trials_idx):
    # Select correct trials
    correct_mask = []
    for n_trial in trials_idx:
        correct_mask.append(bhv_py[n_trial]["TrialError"][0][0] == 0.0)
    print("Number of correct trials in block 2: %d" % sum(correct_mask))
    trials_idx = trials_idx[correct_mask]
    return trials_idx


def get_trials_by_sample(sample_id: np.ndarray) -> np.ndarray:
    o1_c1_idx = np.where(sample_id == 11)[0]
    o1_c5_idx = np.where(sample_id == 15)[0]
    o5_c1_idx = np.where(sample_id == 51)[0]
    o5_c5_idx = np.where(sample_id == 55)[0]
    o0_c0_idx = np.where(sample_id == 0)[0]

    return o1_c1_idx, o1_c5_idx, o5_c1_idx, o5_c5_idx, o0_c0_idx


def get_sp_by_sample(sp, sample_id: np.ndarray) -> np.ndarray:
    o1_c1_sp = sp[np.where(sample_id == 11)[0]]
    o1_c5_sp = sp[np.where(sample_id == 15)[0]]
    o5_c1_sp = sp[np.where(sample_id == 51)[0]]
    o5_c5_sp = sp[np.where(sample_id == 55)[0]]
    o0_c0_sp = sp[np.where(sample_id == 0)[0]]

    return {
        "o1_c1": o1_c1_sp,
        "o1_c5": o1_c5_sp,
        "o5_c1": o5_c1_sp,
        "o5_c5": o5_c5_sp,
        "o0_c0": o0_c0_sp,
    }
