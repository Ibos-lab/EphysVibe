import numpy as np
from matplotlib import pyplot as plt


def get_neuron_in_trials(code, trials_idx, sp_py, neuron, target_on):
    # create list of neurons containing trials belonging to one target location
    target_trials = trials_idx[code]["trials_idx"]  # select trials idx
    code_trials = sp_py["sp_samples"][target_trials]  # select trials
    neuron_trials = []
    for i_t, n_trial in zip(target_trials, code_trials):
        idx_target_on = np.where(sp_py["code_numbers"][i_t] == target_on)[0]
        sample_target_on = sp_py["code_samples"][i_t][idx_target_on]
        neuron_trials.append(n_trial[neuron] - sample_target_on)
    return np.array(neuron_trials, dtype="object")


def trial_average_fr(neuron_trials):
    # Compute the Average firing rate
    sorted_sp_neuron = np.sort(np.concatenate(neuron_trials))
    sum_sp = np.zeros(sorted_sp_neuron[-1] - sorted_sp_neuron[0] + 1)
    sorted_sp_shift = sorted_sp_neuron - sorted_sp_neuron[0]
    for i in sorted_sp_shift:
        sum_sp[i] += 1
    trial_average_sp = sum_sp / len(neuron_trials)
    return trial_average_sp, sorted_sp_neuron


def plot_raster_fr(
    trial_average_sp,
    sorted_sp_neuron,
    conv,
    fs,
    neuron_trials,
    code,
    ax,
    fig,
    i,
    x_lim_max,
    x_lim_min,
):
    num_trials = len(neuron_trials)
    ax2 = ax.twinx()
    # fr
    ax.plot((np.arange(len(trial_average_sp)) + sorted_sp_neuron[0]) / fs, conv)
    # raster
    max_conv = int(max(conv[: int(np.round(x_lim_max - x_lim_min, 0)) * fs]) + 2)
    lineoffsets = np.arange(max_conv, num_trials + max_conv)
    ax2.eventplot(neuron_trials / fs, color=".2", lineoffsets=1, linewidths=0.8)
    # events
    ax.vlines(0, 0, lineoffsets[-1], color="b", linestyles="dashed")
    # figure setings
    ax.set(xlabel="Time (s)", ylabel="Average firing rate")
    ax2.set(xlabel="Time (s)", ylabel="trials")
    ax2.set_yticks(range(-max_conv, num_trials))
    ax.set_title("Code %s" % (code), fontsize=8)
    ax.set_xlim(-0.7, x_lim_max)
    plt.setp(ax2.get_yticklabels(), visible=False)
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    fig.suptitle("Neuron %d" % (i + 1), x=0)
    return fig
