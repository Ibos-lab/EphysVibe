import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from ..trials.spikes import firing_rate

py_filepath = "C:/Users/camil/Documents/int/data/openephys/test_results/LIP/2022-11-22_10-59-03_Riesling_lip_e1_r1.npy"

py_f = np.load(py_filepath, allow_pickle=True).item(0)
sp_py = py_f["sp_data"]
bhv_py = py_f["bhv"]


# Select relevant trials
# Selec trials in a block
block = 2
block_idx = np.where(sp_py["blocks"] == 2)[0]
print("Number of trials in block 2: %d" % len(block_idx))
# Selec correct trials
correct_mask = []
for n_trial in block_idx:
    correct_mask.append(bhv_py[n_trial]["TrialError"][0][0] == 0.0)
print("Number of correct trials in block 2: %d" % sum(correct_mask))
block_idx = block_idx[correct_mask]
b_sp_samples = [sp_py["sp_samples"][i] for i in block_idx]
print("Number of clusters: %d" % len(b_sp_samples[0]))


# Define target codes
target_codes = {
    # code: [ML axis], [plot axis]
    "127": [[10, 0], [1, 2]],
    "126": [[7, 7], [0, 2]],
    "125": [[0, 10], [0, 1]],
    "124": [[-7, 7], [0, 0]],
    "123": [[-10, 0], [1, 0]],
    "122": [[-7, -7], [2, 0]],
    "121": [[0, -10], [2, 1]],
    "120": [[7, -7], [2, 2]],
}

# create dict with the trials that have each code
trials_idx = {}
for i_key, key in enumerate(target_codes.keys()):
    trials = []
    code_idx = []
    for i_trial, code in zip(block_idx, sp_py["code_numbers"][block_idx]):
        idx = np.where(int(key) == code)[0]
        if len(idx) != 0:
            code_idx.append(idx[0])
            trials.append(i_trial)
    trials_idx[key] = {"code_idx": code_idx, "trials_idx": trials}

# Plot rasters for each neuron for each code
target_on = 37
# kernel parameters
fs = 30000
w = 0.015  # seconds = 15 ms
w_size = 0.1  # seconds = 100ms

# define kernel for the convolution
kernel = signal.gaussian(M=w_size * fs, std=w * fs)
kernel = kernel / sum(kernel)  # area of the kernel must be one

fs = 30000
x_lim_max = 2
x_lim_min = -0.7
neurons = np.where(sp_py["clustersgroup"] == "good")[0]
for i, neuron in enumerate(neurons):
    fig, ax = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)
    for code in target_codes.keys():
        neuron_trials = firing_rate.get_neuron_in_trials(
            code, trials_idx, sp_py, neuron, target_on=target_on
        )
        axis = target_codes[code][1]
        ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
        if len(neuron_trials[0]) != 0:
            # Compute trial average fr
            trial_average_sp, sorted_sp_neuron = firing_rate.trial_average_fr(
                neuron_trials
            )
        else:
            trial_average_sp = [0] * len(kernel)
            sorted_sp_neuron = [0] * len(kernel)
        conv = np.convolve(trial_average_sp, kernel, mode="same") * fs
        # plot
        fig = firing_rate.plot_raster_fr(
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
        )
    fig.legend(["Target ON"], fontsize=9)
    plt.show()
