import numpy as np
from typing import Dict, List


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


def get_sp_by_sample(
    sp: np.ndarray, sample_id: np.ndarray, samples: List = [11, 15, 51, 55, 0]
) -> Dict:
    sp_samples = {}
    for s_id in samples:
        s_sp = sp[np.where(sample_id == s_id, True, False)]
        # Check number of trials
        if s_sp.shape[0] > 0:
            sp_samples[str(s_id)] = s_sp
        else:
            sp_samples[str(s_id)] = np.array([np.nan])
    return sp_samples


def select_trials_by_percentile(x: np.ndarray, mask: np.ndarray = None):
    ntr = x.shape[0]
    if mask is None:
        mask = np.full(ntr, True)

    mntr = x[mask].shape[0]

    if mntr < 2:
        return np.full(ntr, True)
    mean_trs = np.mean(x, axis=1)

    q25, q75 = np.percentile(mean_trs[mask], [25, 75])
    iqr = q75 - q25
    upper_limit = q75 + 1.5 * iqr
    lower_limit = q25 - 1.5 * iqr

    q1mask = mean_trs > lower_limit
    q2mask = mean_trs < upper_limit

    qmask = np.logical_and(q1mask, q2mask)
    return qmask
