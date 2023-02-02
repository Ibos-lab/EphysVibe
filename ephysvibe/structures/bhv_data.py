import h5py
import numpy as np
from pathlib import Path


class BhvData:
    def __init__(
        self,
        block: np.ndarray,
        sacc_code: np.ndarray,
        code_numbers: np.ndarray,
        code_times: np.ndarray,
        condition: np.ndarray,
        eye_ml: np.ndarray,
        fix_fp_t_time: np.ndarray,
        fix_fp_post_t_time: np.ndarray,
        fix_fp_pre_t_time: np.ndarray,
        fix_close: np.ndarray,
        fix_far: np.ndarray,
        iti: np.ndarray,
        stim_match: np.ndarray,
        samp_pos: np.ndarray,
        position: np.ndarray,
        reward_plus: np.ndarray,
        test_distractor: np.ndarray,
        test_stimuli: np.ndarray,
        stim_total: np.ndarray,
        trial_error: np.ndarray,
        closeexc: np.ndarray,
        delay_time: np.ndarray,
        excentricity: np.ndarray,
        farexc: np.ndarray,
        fix_post_sacc_blank: np.ndarray,
        fix_time: np.ndarray,
        fix_window_radius: np.ndarray,
        idletime3: np.ndarray,
        max_reaction_time: np.ndarray,
        rand_delay_time: np.ndarray,
        reward_dur: np.ndarray,
        sample_time: np.ndarray,
        stay_time: np.ndarray,
        test_time: np.ndarray,
        wait_for_fix: np.ndarray,
    ):
        """_summary_

        Args:
            block (np.ndarray): dim(trials)
            sacc_code (np.ndarray):
            code_numbers (np.ndarray):
            code_times (np.ndarray):
            condition (np.ndarray): condition in the txt file. dim(trials).
            eye_ml (np.ndarray):
            fix_fp_t_time (np.ndarray): fixation fix point target time
            fix_fp_post_t_time (np.ndarray):
            fix_fp_pre_t_time (np.ndarray):
            fix_close (np.ndarray):
            fix_far (np.ndarray): scaling
            iti (np.ndarray):
            stim_match (np.ndarray):
            samp_pos (np.ndarray):
            position (np.ndarray):
            reward_plus (np.ndarray):
            test_distractor (np.ndarray):
            test_stimuli (np.ndarray):
            stim_total (np.ndarray):
            trial_error (np.ndarray):
            closeexc (np.ndarray):
            delay_time (np.ndarray):
            excentricity (np.ndarray):
            farexc (np.ndarray):
            fix_post_sacc_blank (np.ndarray):
            fix_time (np.ndarray):
            fix_window_radius (np.ndarray):
            idletime3 (np.ndarray):
            max_reaction_time (np.ndarray): max time the monkey has to do the sacc
            rand_delay_time (np.ndarray): range of the delay variation
            reward_dur (np.ndarray):
            sample_time (np.ndarray):
            stay_time (np.ndarray): post sacc fix. dim(trials).
            test_time (np.ndarray):
            wait_for_fix (np.ndarray): max time to fixate before the trial starts. dim(trials).
        """
        self.block = block
        self.sacc_code = sacc_code
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
        self.samp_pos = samp_pos
        self.position = position
        self.reward_plus = reward_plus
        self.test_distractor = test_distractor
        self.test_stimuli = test_stimuli
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
        self._check_shapes()

    def _check_shapes(self):
        n_trials = self.code_numbers.shape[1]
        n_codes = self.code_numbers.shape[0]

    @classmethod
    def from_matlab_mat(cls, load_path: Path):
        """Load data from a file in mat format from Matlab."""
        # load the data
        with h5py.File(load_path, "r") as data:
            #  get data
            group = data["New"]
            block = group["Block"][:].reshape(-1)
            sacc_code = group["SaccCode"][:].reshape(-1)
            code_numbers = group["CodeNumbers"][:].transpose()
            code_times = group["CodeTimes"][:].transpose()
            condition = group["Condition"][:].reshape(-1)
            eye_ml = group["Eye"][:].transpose()  # n_trials, n_times, n_ch
            fix_fp_t_time = group["Fix_FP_T_time"][:].reshape(-1)
            fix_fp_post_t_time = group["Fix_FP_post_T_time"][:].reshape(-1)
            fix_fp_pre_t_time = group["Fix_FP_pre_T_time"][:].reshape(-1)
            fix_close = group["Fixclose"][:].reshape(-1)
            fix_far = group["Fixfar"][:].reshape(-1)
            iti = group["ITI_value"][:].reshape(-1)
            stim_match = group["StimMatch"][:].reshape(-1)
            samp_pos = group["SampPos"][:].reshape(-1)
            position = group["Position"][:].transpose()
            reward_plus = group["Reward_plus"][:].reshape(-1)
            test_distractor = group["TestDistractor"][:].transpose()  # orient,color
            test_stimuli = group["TestStimuli"][:].transpose()  # orientation, color
            stim_total = group["StimTotal"][:].reshape(-1)
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
        data.close()
        # create class object and return
        data = {
            "block": block,
            "sacc_code": sacc_code,
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
            "samp_pos": samp_pos,
            "position": position,
            "reward_plus": reward_plus,
            "test_distractor": test_distractor,
            "test_stimuli": test_stimuli,
            "stim_total": stim_total,
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