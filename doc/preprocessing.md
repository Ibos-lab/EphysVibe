## Steps for preprocessing
### 1 bhv to mat
- Converts MonkeyLogic (ML) file from .bhv to .mat using [bhv_to_mat.m](https://github.com/camilosada/EphysVibe/blob/master/matlab/bhv_to_mat.m)
- Input: path to .bhv file
- Output: .mat file
### 2 Check bhv
- Compares ML and OpenEphys data
- Input: path to .mat file
- Output: bhv.h5
- Example: 

`python -m ephysvibe.pipelines.preprocessing.check_bhv [path_mat] [-o output_path]`
### 3 Compute spikes
- Gets spikes from kilosort files using [compute_spikes.py](https://github.com/camilosada/EphysVibe/blob/master/ephysvibe/pipelines/preprocessing/compute_spikes.py)
- Input: continuous.dat file, path to the bhv.h5
- Output: sp.h5
- Example: 

### Compute neurons
- 


`python -m ephysvibe.pipelines.preprocessing.compute_spk [path_continuous] [path_bhv] [-o output_path] `
### 4 Compute lfp
- Computes local field potentials from raw data using [compute_lfp.py](https://github.com/camilosada/EphysVibe/blob/master/ephysvibe/pipelines/preprocessing/compute_lfp.py)
- Input: continuous.dat file, path to the bhv.h5
- Output: lfp.h5
- Example: 

`python -m ephysvibe.pipelines.preprocessing.compute_lfp [path_continuous] [path_bhv] [-o output_path] [-a areas] [-s start_ch] [-n n_ch]`


![flow](https://github.com/camilosada/EphysVibe/blob/master/img/flow.svg)


## Information within each structure

### bhv_data
* trial_error: 
  * 0: correct trial 
  * 1: no bartouch
  * 2: no fixation
  * 3: break fixation
  * 4: no fixation
  * 5: bar release (before test period)
  * 6: false alarm (relase during test period)
  * 8: missed target
  * 10: break fixation during delay presentaton
* 