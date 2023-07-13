import numpy as np
from numpy.core.umath import pi, mod
import numpy.core.numeric as _nx
from scipy.fft import fft, ifft
import scipy as sp
import logging


def rewrap(p):
    p = p - 2 * np.pi * np.floor((p - np.pi) / (2 * np.pi)) - 2 * np.pi
    return p


def unwrap(p, discont=pi, axis=-1):
    """
    Unwrap by changing deltas between values to 2*pi complement.
    Unwrap radian phase `p` by changing absolute jumps greater than
    `discont` to their 2*pi complement along the given axis.
    Parameters
    ----------
    p : array_like
        Input array.
    discont : float, optional
        Maximum discontinuity between values, default is ``pi``.
    axis : int, optional
        Axis along which unwrap will operate, default is the last axis.
    Returns
    -------
    out : ndarray
        Output array.
    See Also
    --------
    rad2deg, deg2rad
    Notes
    -----
    If the discontinuity in `p` is smaller than ``pi``, but larger than
    `discont`, no unwrapping is done because taking the 2*pi complement
    would only make the discontinuity larger.
    Examples
    --------
    """
    p = np.asarray(p)
    nd = p.ndim
    dd = np.diff(p, axis=axis)
    slice1 = [slice(None, None)] * nd  # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    ddmod = mod(dd + pi, 2 * pi) - pi
    _nx.copyto(ddmod, pi, where=(ddmod == -pi) & (dd > 0))
    ph_correct = ddmod - dd
    _nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    up = np.array(p, copy=True, dtype="d")
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up


def compute_generalized_phase(x, dt):

    # Handle input
    [nch, npts] = x.shape
    # Analytic signal representation (single sided Fourier representation)
    x0 = fft(x, npts, axis=1)
    h = np.zeros((nch, npts))
    # ! improve:
    #
    if npts > 0 and np.mod(npts, 2) == 0:
        h[:, 0 : npts // 2] = 1
        h[:, 1 : npts // 2 - 1] = 2
    else:
        h[:, 0] = 1
        h[:, 1 : npts // 2] = 2
    x0 = ifft(x0 * h, axis=1)
    ph = np.angle(x0)
    md = np.abs(x0)

    # Calculate IF
    wt = np.zeros(x0.shape)
    wt[:, 0:-1] = np.angle(x0[:, 1:] * np.conj(x0[:, :-1])) / (2 * np.pi * dt)

    # Rectify rotation
    # ! This can be improved
    sign_if = np.sign(np.nanmean(wt, axis=1))
    for i, i_sign_if in enumerate(sign_if):
        if i_sign_if == -1:
            modulus = abs(x0[i])
            ang = i_sign_if * np.angle(x0[i])
            x0[i] = modulus * np.exp(1j * ang)
            ph[i] = np.angle(x0[i])
            md = abs(x0[i])
            wt[i, 0:-1] = np.angle(x0[i, 1:] * np.conj(x0[i, :-1])) / (2 * np.pi * dt)

    # # check if nan channels
    # if np.all(np.isnan(ph)):
    #     continue

    # Find negative freq epochs
    idx = (wt < 0).astype(int)
    idx[:, 0] = 0
    diff = np.diff(idx, axis=1)
    r_start, c_start = np.where(diff == 1)
    r_end, c_end = np.where(diff == -1)
    duration = c_end - (c_start + 1)
    [start, end] = c_start + 1, (c_start + 1 + duration * 3) + 1
    for i, i_r_start in enumerate(r_start):
        idx[i_r_start, start[i] : end[i]] = 1

    ph[idx.astype(bool)] = np.nan
    if ~np.all(np.isnan(ph)):
        for i, i_ph in enumerate(ph):
            not_nan = ~np.isnan(i_ph)
            ph[i, not_nan] = unwrap(i_ph[not_nan])

        dx = np.linspace(0, 1, npts)
        isnan = np.isnan(ph)
        for i, i_ch in enumerate(ph):
            if np.sum(~np.isnan(i_ch)) < 2:
                continue
            f = sp.interpolate.PchipInterpolator(dx[~isnan[i]], ph[i, ~isnan[i]])
            ph[i, isnan[i]] = f(dx[isnan[i]])
        ph = rewrap(ph)

    xgp = md * np.exp(1j * ph)

    return xgp, wt


def complex_mean(alpha, w=None, axis=None, axial_correction=1):
    if w is None:
        w = np.ones_like(alpha)
        alpha = np.asarray(alpha)
    assert w.shape == alpha.shape, (
        "Dimensions of data "
        + str(alpha.shape)
        + " and w "
        + str(w.shape)
        + " do not match!"
    )

    return (w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(
        w, axis=axis
    )


def mean_direction(alpha, w=None, ci=None, d=None, axis=None, axial_correction=1):
    """
    Compute mean direction of circular data.
    :param alpha: circular data
    :param w: 	 weightings in case of binned angle data
    :param ci: if not None, the upper and lower 100*ci% confidence
               interval is returned as well
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :return: circular mean if ci=None, or circular mean as well as lower and
             upper confidence interval limits

    """

    cmean = complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)

    mu = np.angle(cmean) / axial_correction

    if ci is None:
        return mu
    else:
        if axial_correction > 1:
            print("Axial correction ignored for confidence intervals.")
        # t = mean_ci_limits(alpha, ci=ci, w=w, d=d, axis=axis)
        return mu  # , CI(mu - t, mu + t)


def center(*args, **kwargs):
    """
    Centers the data on its circular mean.
    Each non-keyword argument is another data array that is centered.
    :param axis: the mean is computed along this dimension (default axis=None).
                **Must be used as a keyword argument!**
    :return: tuple of centered data arrays
    """

    axis = kwargs.pop("axis", None)
    if axis is None:
        axis = 0
        args = [a.ravel() for a in args]

    reshaper = tuple(
        slice(None, None) if i != axis else np.newaxis
        for i in range(len(args[0].shape))
    )
    if len(args) == 1:
        return args[0] - mean_direction(args[0], axis=axis)
    else:
        return tuple(
            [
                a - mean_direction(a, axis=axis)[reshaper]
                for a in args
                if isinstance(a, np.ndarray)
            ]
        )


def resultant_vector_length(
    alpha, w=None, d=None, axis=None, axial_correction=1, ci=None, bootstrap_iter=None
):
    """Computes mean resultant vector length for circular data.
    This statistic is sometimes also called vector strength.
    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
    default None.
    :param d: spacing of bin centers for binned data, if supplied
    correction factor is used to correct for bias in
    estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
    (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
    (number of samples if None)
    :return: mean resultant length
    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_"""
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()
    cmean = complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)
    # obtain length
    r = np.abs(cmean)
    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            print("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


def corrcc(alpha1, alpha2, ci=None, axis=None, bootstrap_iter=None):
    """
    Circular correlation coefficient for two circular random variables.
    If a confidence level is specified, confidence limits are bootstrapped.
    The number of bootstrapping iterations is min(number of data points
    along axis, bootstrap_max_iter).
    :param alpha1: sample of angles in radians
    :param alpha2: sample of angles in radians
    :param axis: correlation coefficient is computed along this dimension
                 (default axis=None, across all dimensions)
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations
                           (number of samples if None)
    :return: correlation coefficient if ci=None, otherwise correlation
             coefficient with lower and upper confidence limits
    References: [Jammalamadaka2001]_
    """
    assert alpha1.shape == alpha2.shape, "Input dimensions do not match."

    # center data on circular mean
    alpha1, alpha2 = center(alpha1, alpha2, axis=axis)

    # compute correlation coeffcient from p. 176
    num = np.sum(np.sin(alpha1) * np.sin(alpha2), axis=axis)
    den = np.sqrt(
        np.sum(np.sin(alpha1) ** 2, axis=axis) * np.sum(np.sin(alpha2) ** 2, axis=axis)
    )
    return num / den
