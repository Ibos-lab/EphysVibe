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
