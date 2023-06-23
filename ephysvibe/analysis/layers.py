import numpy as np


def compute_csd(
    lfp: np.ndarray, inter_channel_distance: float, step: int = 2
) -> np.ndarray:
    """Compute current source density on the average LFP.
    (Mitzdorf U., 1985).
    CSD is computed as the second spatial derivative of the extracellular field potential what
    is proportional to the current sinks and sources in the extracellular space.

    Args:
        lfp (np.ndarray): array of shape (n channels, timestamps) containing the local field potential.
        inter_channel_distance (float): distance between electrode contacts.
        step (int, optional): number of channels between the middel channel and the extreme ones. Defaults to 2.

    Returns:
        np.ndarray: current source density.
    """
    if lfp.ndim != 2:
        raise ValueError("lfp must be a 2d array")

    csd = []
    n_channels = lfp.shape[0]
    for channel in range(0 + step, n_channels - step):
        csd.append(
            (-2 * lfp[channel] + lfp[channel - step] + lfp[channel + step])
            / ((step * inter_channel_distance) ** 2)
        )

    return np.array(csd)
