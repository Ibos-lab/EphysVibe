import numpy as np
from scipy import stats
from ..stats import processing
from ..trials import align_trials
from sklearn import metrics
from ..trials import firing_rate
from typing import Tuple


def compute_roc_auc(group1, group2):
    roc_score = []
    p = []
    for n_win in np.arange(group1.shape[1]):
        g1 = group1[:, n_win]
        g2 = group2[:, n_win]
        p.append(stats.ttest_ind(g1, g2)[1])
        thresholds = np.unique(np.concatenate([g1, g2]))
        y_g1, y_g2 = np.ones(len(g1)), np.zeros(len(g2))
        score = 0.5
        fpr, tpr = [], []
        for threshold in thresholds:
            g1_y_pred, g2_y_pred = np.zeros(len(g1)), np.zeros(len(g2))
            g1_mask, g2_mask = g1 >= threshold, g2 >= threshold
            g1_y_pred[g1_mask], g2_y_pred[g2_mask] = 1, 1
            tp = sum(np.logical_and(y_g1 == 1, g1_y_pred == 1))
            fn = sum(np.logical_and(y_g1 == 1, g1_y_pred == 0))
            tpr.append(tp / (tp + fn))
            fp = sum(np.logical_and(y_g2 == 0, g2_y_pred == 1))
            tn = sum(np.logical_and(y_g2 == 0, g2_y_pred == 0))
            fpr.append(fp / (fp + tn))
        if len(fpr) > 1:
            fpr, tpr = np.array(fpr), np.array(tpr)
            score = metrics.auc(fpr[fpr.argsort()], tpr[fpr.argsort()])
        roc_score.append(score)
    roc_score = np.array(roc_score)
    roc_score = processing.scale_signal(np.round(roc_score, 2), out_range=[-1, 1])
    return roc_score, np.array(p)


def find_latency(
    p_value: np.ndarray, win: int, step: int = 1, p_treshold: float = 0.01
) -> np.ndarray:
    sig = np.full(p_value.shape[0], False)
    # sig[p_value < 0.01] = True
    for i_step in np.arange(0, sig.shape[0], step):
        sig[i_step] = np.where(
            np.all(p_value[i_step : i_step + win] < p_treshold), True, False
        )
    latency = np.where(sig)[0]

    if len(latency) != 0:
        endl = np.where(sig[latency[0] :] == False)[0]
        endl = endl[0] if len(endl) != 0 else -1
        return latency[0], endl + latency[0] + win
    else:
        return np.nan, np.nan


def get_selectivity(
    sp_1, sp_2, win, scores=False
) -> Tuple[float, np.ndarray, np.ndarray]:
    nanarray = np.array([np.nan])
    if np.logical_or(sp_1.ndim < 2, sp_2.ndim < 2):
        return np.nan, nanarray, nanarray
    if np.logical_or(sp_1.shape[0] < 2, sp_2.shape[0] < 2):
        return np.nan, nanarray, nanarray
    roc_score, p_value = compute_roc_auc(sp_1, sp_2)
    lat, _ = find_latency(p_value, win=win, step=1)
    if np.isnan(lat):
        roc_score = roc_score if scores else nanarray
        return lat, roc_score, p_value
    roc_score = roc_score if scores else np.array(roc_score[lat])
    return lat, roc_score, p_value


# def get_vd_index(bl, group1, group2, step=1, avg_win=100, pwin=75):
# Computes the index using the significant period
#     p_son, p_d = [], []
#     bl = np.mean(bl, axis=1)
#     for i in range(0, group1.shape[1] - avg_win, step):
#         g1 = np.mean(group1[:, i : i + avg_win], axis=1)
#         p_son.append(stats.ranksums(bl, g1)[1])
#     for i in range(0, group2.shape[1] - avg_win, step):
#         g2 = np.mean(group2[:, i : i + avg_win], axis=1)
#         p_d.append(stats.ranksums(bl, g2)[1])
#     p_son = np.array(p_son)
#     p_d = np.array(p_d)
#     lat_son, end_son = find_latency(p_value=p_son, win=pwin, step=1)
#     lat_d, end_d = find_latency(p_value=p_d, win=pwin, step=1)
#     if np.logical_and(np.isnan(lat_son), ~np.isnan(lat_d)):
#         g1 = group1
#         g2 = group2[:, lat_d:end_d]
#     elif np.logical_and(~np.isnan(lat_son), np.isnan(lat_d)):
#         g1 = group1[:, lat_son:end_son]
#         g2 = group2
#     elif np.logical_and(np.isnan(lat_son), np.isnan(lat_d)):
#         return np.nan, np.nan, np.nan, np.nan
#     else:
#         g1 = group1[:, lat_son:end_son]
#         g2 = group2[:, lat_d:end_d]
#     bl_mean = np.mean(bl)
#     g1_mean = np.mean(g1)
#     g2_mean = np.mean(g2)
#     g2_mean_bl = np.abs(g2_mean - bl_mean)
#     g1_mean_bl = np.abs(g1_mean - bl_mean)
#     vd_idx = (g2_mean_bl - g1_mean_bl) / (g1_mean_bl + g2_mean_bl)
#     return vd_idx, bl_mean, g1_mean, g2_mean


def get_vd_index(bl, group1, group2, st_v, end_v, st_d, end_d, pwin=75):
    p_son, p_d = [], []
    bl = np.mean(bl, axis=1)
    for i in range(st_v, end_v):
        g1 = group1[:, i]
        p_son.append(stats.ranksums(bl, g1)[1])  # Wilcoxon rank-sum
    for i in range(st_d, end_d):
        g2 = group2[:, i]
        p_d.append(stats.ranksums(bl, g2)[1])
    p_son = np.array(p_son)
    p_d = np.array(p_d)
    lat_son, end_son = find_latency(p_value=p_son, win=pwin, step=1)
    lat_don, end_don = find_latency(p_value=p_d, win=pwin, step=1)
    if np.logical_and(np.isnan(lat_son), np.isnan(lat_don)):
        return np.nan, np.nan, np.nan, np.nan
    else:
        g1 = group1[:, st_v:end_v]
        g2 = group2[:, st_d:end_d]
    bl_mean = np.mean(bl)
    g1_mean = np.mean(g1)
    g2_mean = np.mean(g2)
    g2_mean_bl = np.abs(g2_mean - bl_mean)
    g1_mean_bl = np.abs(g1_mean - bl_mean)
    # if np.logical_and(g1_mean_bl*1000<3,g2_mean_bl*1000<3):
    #     return np.nan, np.nan, np.nan, np.nan
    vd_idx = (g2_mean_bl - g1_mean_bl) / (g1_mean_bl + g2_mean_bl)

    if np.logical_and(np.isnan(lat_son), vd_idx < 0):
        return np.nan, np.nan, np.nan, np.nan
    elif np.logical_and(np.isnan(lat_don), vd_idx > 0):
        return np.nan, np.nan, np.nan, np.nan
    else:
        return vd_idx, bl_mean * 1000, g1_mean * 1000, g2_mean * 1000


def compute_vd_idx(
    neu_data=None,
    time_before=None,
    st_v=50,
    end_v=200,
    st_d=100,
    end_d=300,
    vd_pwin=75,
    vd_avg_win=200,
    sp_s=None,
    sp_d=None,
    in_out=1,
):
    if neu_data is not None:
        # get spike matrices in and out conditions
        sp_s, mask_s = align_trials.get_align_tr(
            neu_data,
            select_block=1,
            select_pos=in_out,
            time_before=time_before + vd_avg_win,
        )
        sp_s = sp_s[neu_data.sample_id[mask_s] != 0]
        sp_d, mask_d = align_trials.get_align_tr(
            neu_data,
            select_block=1,
            select_pos=in_out,
            time_before=0 + vd_avg_win,
            event="sample_off",
        )
        sp_d = sp_d[neu_data.sample_id[mask_d] != 0]

        sp_s = firing_rate.moving_average(data=sp_s, win=vd_avg_win, step=1)[
            :, vd_avg_win:
        ]
        sp_d = firing_rate.moving_average(data=sp_d, win=vd_avg_win, step=1)[
            :, vd_avg_win:
        ]

    #### Compute VD index
    # get avg fr over trials and time
    vd_idx, bl_mean, g1_mean, g2_mean = np.nan, np.nan, np.nan, np.nan

    if np.logical_and(sp_d.shape[0] > 2, sp_d.ndim > 1):
        vd_idx, bl_mean, g1_mean, g2_mean = get_vd_index(
            bl=sp_s[:, :time_before],
            group1=sp_s[:, time_before : time_before + st_v + 460],
            group2=sp_d[:, : st_d + 460],
            st_v=st_v,
            end_v=end_v,
            st_d=st_d,
            end_d=end_d,
            pwin=vd_pwin,
        )

    return vd_idx, bl_mean, g1_mean, g2_mean


def compute_fr(frsignal, st_max=0, win=100):
    nan_init = [np.nan] * 3
    mean_fr, lat_max_fr, mean_max_fr = nan_init
    win = int(win / 2)
    if ~np.all(np.isnan(frsignal)):
        # Mean fr during epochs
        mean_fr = np.nanmean(frsignal) * 1000
        # Max fr
        imax = np.nanargmax(frsignal[st_max:])
        if ~np.isnan(imax):
            imax = int(imax)
            lat_max_fr = imax
            imax = win if imax < win else imax
            mean_max_fr = np.mean(frsignal[imax - win : imax + win]) * 1000

    return {
        "mean_fr": mean_fr,
        "lat_max_fr": lat_max_fr,
        "mean_max_fr": mean_max_fr,
    }
