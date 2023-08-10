## Steps for preprocessing
### 1 bhv to mat
- Converts MonkeyLogic (ML) file from .bhv to .mat using [bhv_to_mat.m](/EphysVibe/matlab/)
- Input: .bhv file
- Output: .mat file
### 2 Check bhv
- Compares ML and OpenEphys data
- Input: 
- Output: 
- Example: 

`python -m ephysvibe.pipelines.preprocessing.check_bhv `
### 3 Compute spikes
- Gets spikes from kilosort files using [compute_spikes.py](/EphysVibe/ephysvibe/pipelines/preprocessing/compute_spikes.py)
- Input: 
- Output: 
- Example: 

`python -m ephysvibe.pipelines.preprocessing.compute_spk []`
### 4 Compute lfp
- Computes local field potentials from raw data using [compute_lfp.py](/EphysVibe/ephysvibe/pipelines/preprocessing/compute_lfp.py)
- Input: continuous.dat file, path to the bhv.h5
- Output: lfp.h5
- Example: 

`python -m ephysvibe.pipelines.preprocessing.compute_lfp [path_continuous] [path_bhv] [-o output_path] [-a areas] [-s start_ch] [-n n_ch]`


![flow](/EphysVibe/img/flow.svg)


## Information within each structure

