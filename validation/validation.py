import h5py
import re
import numpy as np


def read_matlab_file(mat_filepath):
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            split_name = re.split(r"[/]", node.name)
            if split_name[1] != "#refs#" and split_name[2] != "BHV":
                if node.dtype != "O":
                    mat_res[node.name] = np.array(node)
                else:
                    mat_res[node.name] = []
                    for i in range(node.size):
                        mat_res[node.name].append({})
                        for ref in np.array(f[node[i][0]]):
                            loc = f[node[i][0]].name + "/" + ref
                            mat_res[node.name][i][ref] = np.array(f[loc])

    mat_res = {}
    f = h5py.File(mat_filepath, "r")
    f.visititems(visitor_func)
    return mat_res


def check_spikes(mat_res, py_f, start_timestamps, end_timestamps):
    flag_n_sp = 0
    flag_t_sp = 0

    if len(mat_res["/data/NEURO/Neuron"]) == np.sum(py_f["clustersgroup"] == "good"):
        print("Number of neurons do match")

    if len(mat_res["/data/NEURO/MUA"]) == np.sum(py_f["clustersgroup"] == "mua"):
        print("Number of mua do match")

    idx_py_neurons = np.where(py_f["clustersgroup"] == "good")[0]
    idx_py_mua = np.where(py_f["clustersgroup"] == "mua")[0]

    for n, neuron in enumerate(mat_res["/data/NEURO/Neuron"]):
        for i_trial, (i_start, i_end) in enumerate(
            zip(start_timestamps, end_timestamps)
        ):
            times_n = neuron["times"][0]
            if (
                py_f["times"][i_trial, idx_py_neurons[n]].shape
                != times_n[np.logical_and(times_n >= i_start, times_n < i_end)].shape
            ):

                flag_n_sp = 1

            if (
                np.sum(
                    py_f["times"][i_trial, idx_py_neurons[n]]
                    - times_n[np.logical_and(times_n >= i_start, times_n < i_end)]
                )
                != 0
            ):

                flag_t_sp = 1

    if flag_n_sp == 1:
        print("Number of spikes do not match")
    if flag_t_sp == 1:
        print("Spike times do not match")
    if flag_n_sp == 0 and flag_t_sp == 0:
        print("Number of spikes do match")
        print("Spike times do match")
