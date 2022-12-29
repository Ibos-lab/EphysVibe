import numpy as np
from matplotlib import pyplot as plt


def select_events_timestamps(sp_py, trials_idx, events):
    events_timestamps = []
    for i_t in trials_idx:
        e_timestamps = []
        for _, event in events.items():
            idx_event = np.where(sp_py["code_numbers"][i_t] == event)[0]
            if len(idx_event) == 0:
                sample_event = [np.nan]
            else:
                sample_event = sp_py["code_samples"][i_t][idx_event]
            e_timestamps.append(sample_event)
        events_timestamps.append(np.concatenate(e_timestamps))

    return np.array(events_timestamps, dtype="object")


def align_neuron_spikes(trials_idx, sp_py, neuron, event_timestamps):
    # create list of neurons containing the spikes timestamps aligned with the event
    neuron_trials = []
    for i, i_t in enumerate(trials_idx):
        neuron_trials.append(sp_py["sp_samples"][i_t][neuron] - event_timestamps[i])
    return np.array(neuron_trials, dtype="object")


def trial_average_fr(neuron_trials):
    # Compute the Average firing rate
    sorted_sp_neuron = np.sort(np.concatenate(neuron_trials))
    sum_sp = np.zeros(int(sorted_sp_neuron[-1] - sorted_sp_neuron[0] + 1))
    sorted_sp_shift = np.array(sorted_sp_neuron - sorted_sp_neuron[0], dtype=int)
    for i in sorted_sp_shift:
        sum_sp[i] += 1
    trial_average_sp = sum_sp / len(neuron_trials)
    return trial_average_sp, sorted_sp_neuron


def plot_raster_fr(
    sample_first_sp,
    conv,
    fs,
    neuron_trials,
    code,
    ax,
    fig,
    i,
    x_lim_max,
    x_lim_min,
    conv_max,
    events,
):
    num_trials = len(neuron_trials)
    ax2 = ax.twinx()
    conv_max = int(round(conv_max, 0)) + 2
    # fr
    ax.plot((np.arange(len(conv)) + sample_first_sp) / fs, conv)
    # raster
    # max_conv = int(max(conv[: int(np.round(x_lim_max - x_lim_min, 0)) * fs]) + 2)
    lineoffsets = np.arange(conv_max, num_trials + conv_max)
    ax2.eventplot(neuron_trials / fs, color=".2", lineoffsets=1, linewidths=0.8)
    # events
    ax.vlines(
        events[0] / fs, 0, lineoffsets[-1], color="b", linestyles="dashed"
    )  # target_on
    ax.vlines(
        events[1] / fs, 0, lineoffsets[-1], color="k", linestyles="dashed"
    )  # target_off
    ax.vlines(
        events[2] / fs, 0, lineoffsets[-1], color="k", linestyles="dashed"
    )  # fix_spot_off
    ax.vlines(
        events[3] / fs, 0, lineoffsets[-1], color="k", linestyles="dashed"
    )  # response

    # figure setings
    ax.set(xlabel="Time (s)", ylabel="Average firing rate")
    ax2.set(xlabel="Time (s)", ylabel="trials")
    ax2.set_yticks(range(-conv_max, num_trials))
    ax.set_title("Code %s" % (code), fontsize=8)
    ax.set_xlim(x_lim_min, x_lim_max)
    plt.setp(ax2.get_yticklabels(), visible=False)
    fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    fig.suptitle("Neuron %d" % (i + 1), x=0)
    return fig
