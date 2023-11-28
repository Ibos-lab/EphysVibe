import numpy as np
from ephysvibe.task import task_constants


def indep_roll(arr: np.ndarray, shifts: np.ndarray, axis: int = 1) -> np.ndarray:
    """Apply an independent roll for each dimensions of a single axis.
    Args:
        arr (np.ndarray): Array of any shape.
        shifts (np.ndarray): How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
        axis (int, optional): Axis along which elements are shifted. Defaults to 1.

    Returns:
        np.ndarray: shifted array.
    """
    arr = np.swapaxes(arr, axis, -1)
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]
    # Convert to a positive shift
    shifts[shifts < 0] += arr.shape[-1]
    all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]
    result = arr[tuple(all_idcs)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def align_on(
    sp_samples: np.ndarray,
    code_samples: np.ndarray,
    code_numbers: np.ndarray,
    trial_error: np.ndarray,
    block: np.ndarray,
    pos_code: np.ndarray,
    select_block: int = 1,
    event: str = "sample_on",
    time_before: int = 500,
    correct_trials: bool = True,
    select_pos: int = 1,
) -> [np.ndarray, np.ndarray]:
    if correct_trials:  # select correct trials in select_block and select position
        mask = np.where(
            np.logical_and(
                pos_code == select_pos,
                np.logical_and(trial_error == 0, block == select_block),
            ),
            True,
            False,
        )
    else:
        mask = np.where(block == select_block, True, False)
    sp_samples = sp_samples[mask]
    if select_block == 1:
        code = task_constants.EVENTS_B1[event]
    elif select_block == 2:
        code = task_constants.EVENTS_B2[event]
    else:
        return
    shifts = code_samples[mask][np.where(code_numbers[mask] == code, True, False)]
    shifts = (shifts - time_before).astype(int)
    # align sp
    align_sp = indep_roll(arr=sp_samples, shifts=-shifts, axis=1)
    return align_sp, mask
