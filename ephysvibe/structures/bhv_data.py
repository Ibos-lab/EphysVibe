import h5py
import numpy as np
from pathlib import Path


class BhvData:
    def __init__(
        self,
        # both
        block: np.ndarray,
        iti: np.ndarray,
        position: np.ndarray,
        reward_plus: np.ndarray,
        trial_error: np.ndarray,
        delay_time: np.ndarray,
        fix_time: np.ndarray,
        fix_window_radius: np.ndarray,
        idletime3: np.ndarray,
        rand_delay_time: np.ndarray,
        reward_dur: np.ndarray,
        wait_for_fix: np.ndarray,
        # sacc
        fix_post_sacc_blank: np.ndarray,
        max_reaction_time: np.ndarray,
        stay_time: np.ndarray,
        fix_fp_t_time: np.ndarray,
        fix_fp_post_t_time: np.ndarray,
        fix_fp_pre_t_time: np.ndarray,
        fix_close: np.ndarray,
        fix_far: np.ndarray,
        closeexc: np.ndarray,
        excentricity: np.ndarray,
        farexc: np.ndarray,
        # dmts
        eye_ml: np.ndarray,
        condition: np.ndarray,
        code_numbers: np.ndarray,
        code_times: np.ndarray,
        stim_match: np.ndarray,
        pos_code: np.ndarray,
        stim_total: np.ndarray,
        test_distractor: np.ndarray,
        test_stimuli: np.ndarray,
        sample_time: np.ndarray,
        test_time: np.ndarray,
        sample_id: np.ndarray,
        **kwargs,
    ):
        """_summary_

        Args:
            block (np.ndarray): shape: (trials).
            iti (np.ndarray): duration of the intertrial interval. shape: (trials).
            position (np.ndarray): position of the stimulus.  shape: (trials, 2).
            reward_plus (np.ndarray): the amount of reward if more was given. shape: (trials).
            trial_error (np.ndarray): if 0: correct trial else: code of the error. shape: (trials).
            delay_time (np.ndarray): duration of the delay. shape: (trials).
            fix_time (np.ndarray): duration of the fixation. shape: (trials).
            fix_window_radius (np.ndarray): shape: (trials).
            idletime3 (np.ndarray):
            rand_delay_time (np.ndarray): range of the delay variation. shape: (trials).
            reward_dur (np.ndarray): duration of the reward. shape: (trials).
            wait_for_fix (np.ndarray): max time to fixate before the trial starts. shape: (trials).
            sacc_code (np.ndarray): shape: (trials).
            fix_post_sacc_blank (np.ndarray):
            max_reaction_time (np.ndarray): max time the monkey has to do the sacc
            stay_time (np.ndarray): post sacc fix. shape: (trials).
            fix_fp_t_time (np.ndarray): fixation fix point target time
            fix_fp_post_t_time (np.ndarray):
            fix_fp_pre_t_time (np.ndarray):
            fix_close (np.ndarray):
            fix_far (np.ndarray): scaling
            closeexc (np.ndarray):
            excentricity (np.ndarray):
            farexc (np.ndarray):
            eye_ml (np.ndarray):
            condition (np.ndarray): condition in the txt file. shape: (trials).
            code_numbers (np.ndarray):array of shape (trials, events) containing the codes of the events.
            code_samples (np.ndarray):array of shape (trials, events) containing the timestamp of the events
            code_times (np.ndarray): exact time when each event ocurred during the trial. shape: (trials, events).
            stim_match (np.ndarray):
            pos_code (np.ndarray): position of the sample stimulus. shape: (trials, 2).
            stim_total (np.ndarray):
            test_distractor (np.ndarray): orientation, color
            test_stimuli (np.ndarray): orientation, color
            sample_time (np.ndarray):
            test_time (np.ndarray):
            sample_id (np.ndarray):
        """
        self.block = block
        self.code_numbers = code_numbers
        self.code_times = code_times
        self.condition = condition
        self.eye_ml = eye_ml
        self.fix_fp_t_time = fix_fp_t_time
        self.fix_fp_post_t_time = fix_fp_post_t_time
        self.fix_fp_pre_t_time = fix_fp_pre_t_time
        self.fix_close = fix_close
        self.fix_far = fix_far
        self.iti = iti
        self.stim_match = stim_match
        self.pos_code = pos_code
        self.position = position
        self.reward_plus = reward_plus
        self.test_distractor = test_distractor
        self.test_stimuli = test_stimuli
        self.sample_id = sample_id
        self.stim_total = stim_total
        self.trial_error = trial_error
        self.closeexc = closeexc
        self.delay_time = delay_time
        self.excentricity = excentricity
        self.farexc = farexc
        self.fix_post_sacc_blank = fix_post_sacc_blank
        self.fix_time = fix_time
        self.fix_window_radius = fix_window_radius
        self.idletime3 = idletime3
        self.max_reaction_time = max_reaction_time
        self.rand_delay_time = rand_delay_time
        self.reward_dur = reward_dur
        self.sample_time = sample_time
        self.stay_time = stay_time
        self.test_time = test_time
        self.wait_for_fix = wait_for_fix
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self._check_shapes()

    def _check_shapes(self):
        n_trials = self.code_numbers.shape[1]
        # n_codes = self.code_numbers.shape[0]

    @classmethod
    def from_matlab_mat(cls, load_path: Path):
        """Load data from a file in mat format from Matlab."""
        # load the data
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["New"]
            block = group["Block"][:].reshape(-1)
            code_numbers = group["CodeNumbers"][:].transpose()
            code_times = group["CodeTimes"][:].transpose()
            condition = group["Condition"][:].reshape(-1)
            eye_ml = np.array(group["Eye"]).transpose()  # n_trials, n_times, n_ch
            fix_fp_t_time = group["Fix_FP_T_time"][:].reshape(-1)
            fix_fp_post_t_time = group["Fix_FP_post_T_time"][:].reshape(-1)
            fix_fp_pre_t_time = group["Fix_FP_pre_T_time"][:].reshape(-1)
            fix_close = group["Fixclose"][:].reshape(-1)
            fix_far = group["Fixfar"][:].reshape(-1)
            iti = group["ITI_value"][:].reshape(-1)
            stim_match = group["StimMatch"][:].reshape(-1)
            pos_code = group["SampPosCode"][:].reshape(-1)
            position = group["Position"][:].transpose()
            reward_plus = group["Reward_plus"][:].reshape(-1)
            test_distractor = group["TestDistractor"][:].transpose()  # orient,color
            test_stimuli = group["TestStimuli"][:].transpose()  # orientation, color
            stim_total = group["StimTotal"][:].reshape(-1)
            sample_id = group["SampleId"][:].reshape(-1)  # orientation, color
            trial_error = group["TrialError"][:].reshape(-1)
            closeexc = group["closeexc"][:].reshape(-1)
            delay_time = group["delay_time"][:].reshape(-1)
            excentricity = group["excentricity"][:].reshape(-1)
            farexc = group["farexc"][:].reshape(-1)
            fix_post_sacc_blank = group["fix_post_sacc_blank"][:].reshape(-1)
            fix_time = group["fix_time"][:].reshape(-1)
            fix_window_radius = group["fix_window_radius"][:].reshape(-1)
            idletime3 = group["idletime3"][:].reshape(-1)
            max_reaction_time = group["max_reaction_time"][:].reshape(-1)
            rand_delay_time = group["rand_delay_time"][:].reshape(-1)
            reward_dur = group["reward_dur"][:].reshape(-1)
            sample_time = group["sample_time"][:].reshape(-1)
            stay_time = group["stay_time"][:].reshape(-1)
            test_time = group["test_time"][:].reshape(-1)
            wait_for_fix = group["wait_for_fix"][:].reshape(-1)
        f.close()
        # create class object and return
        data = {
            "block": block,
            "code_numbers": code_numbers,
            "code_times": code_times,
            "condition": condition,
            "eye_ml": eye_ml,
            "fix_fp_t_time": fix_fp_t_time,
            "fix_fp_post_t_time": fix_fp_post_t_time,
            "fix_fp_pre_t_time": fix_fp_pre_t_time,
            "fix_close": fix_close,
            "fix_far": fix_far,
            "iti": iti,
            "stim_match": stim_match,
            "pos_code": pos_code,
            "position": position,
            "reward_plus": reward_plus,
            "test_distractor": test_distractor,
            "test_stimuli": test_stimuli,
            "stim_total": stim_total,
            "sample_id": sample_id,
            "trial_error": trial_error,
            "closeexc": closeexc,
            "delay_time": delay_time,
            "excentricity": excentricity,
            "farexc": farexc,
            "fix_post_sacc_blank": fix_post_sacc_blank,
            "fix_time": fix_time,
            "fix_window_radius": fix_window_radius,
            "idletime3": idletime3,  #
            "max_reaction_time": max_reaction_time,
            "rand_delay_time": rand_delay_time,
            "reward_dur": reward_dur,
            "sample_time": sample_time,
            "stay_time": stay_time,
            "test_time": test_time,
            "wait_for_fix": wait_for_fix,
        }
        return cls(**data)

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data and create class object
        bhv_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]
            for key, value in zip(group.keys(), group.values()):
                bhv_data[key] = value[:]
        f.close()
        return cls(**bhv_data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
                group.create_dataset(key, value.shape, data=value)
        f.close()
