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
        sacc_code: np.ndarray,
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
        samp_pos: np.ndarray,
        stim_total: np.ndarray,
        test_distractor: np.ndarray,
        test_stimuli: np.ndarray,
        sample_time: np.ndarray,
        test_time: np.ndarray,
        code_samples: np.ndarray = np.array([np.nan]),
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
            code_numbers (np.ndarray): code of the events.  shape: (trials, events).
            code_samples
            code_times (np.ndarray): exact time when each event ocurred during the trial. shape: (trials, events).
            stim_match (np.ndarray):
            samp_pos (np.ndarray): position of the sample stimulus. shape: (trials, 2).
            stim_total (np.ndarray):
            test_distractor (np.ndarray):
            test_stimuli (np.ndarray):
            sample_time (np.ndarray):
            test_time (np.ndarray):
        """
        self.block = block
        self.sacc_code = sacc_code
        self.code_numbers = code_numbers
        self.code_samples = code_samples
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
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["New"]
            block = group["Block"][:].reshape(-1)
            sacc_code = group["SaccCode"][:].reshape(-1)
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
        f.close()
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

    def from_python_hdf5(load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["data"]
            block = group["block"][:]
            sacc_code = group["sacc_code"][:]
            code_numbers = group["code_numbers"][:]
            code_samples = group["code_samples"][:]
            code_times = group["code_times"][:]
            condition = group["condition"][:]
            eye_ml = group["eye_ml"][:]
            fix_fp_t_time = group["fix_fp_t_time"][:]
            fix_fp_post_t_time = group["fix_fp_post_t_time"][:]
            fix_fp_pre_t_time = group["fix_fp_pre_t_time"][:]
            fix_close = group["fix_close"][:]
            fix_far = group["fix_far"][:]
            iti = group["iti"][:]
            stim_match = group["stim_match"][:]
            samp_pos = group["samp_pos"][:]
            position = group["position"][:]
            reward_plus = group["reward_plus"][:]
            test_distractor = group["test_distractor"][:]
            test_stimuli = group["test_stimuli"][:]
            stim_total = group["stim_total"][:]
            trial_error = group["trial_error"][:]
            closeexc = group["closeexc"][:]
            delay_time = group["delay_time"][:]
            excentricity = group["excentricity"][:]
            farexc = group["farexc"][:]
            fix_post_sacc_blank = group["fix_post_sacc_blank"][:]
            fix_time = group["fix_time"][:]
            fix_window_radius = group["fix_window_radius"][:]
            idletime3 = group["idletime3"][:]
            max_reaction_time = group["max_reaction_time"][:]
            rand_delay_time = group["rand_delay_time"][:]
            reward_dur = group["reward_dur"][:]
            sample_time = group["sample_time"][:]
            stay_time = group["stay_time"][:]
            test_time = group["test_time"][:]
            wait_for_fix = group["wait_for_fix"][:]
        f.close()
        # create class object and return
        bhv_data = {
            "block": block,
            "sacc_code": sacc_code,
            "code_numbers": code_numbers,
            "code_samples": code_samples,
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
            "idletime3": idletime3,
            "max_reaction_time": max_reaction_time,
            "rand_delay_time": rand_delay_time,
            "reward_dur": reward_dur,
            "sample_time": sample_time,
            "stay_time": stay_time,
            "test_time": test_time,
            "wait_for_fix": wait_for_fix,
        }
        return bhv_data

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:

            group = f.create_group("data")
            group.create_dataset("block", self.block.shape, data=self.block)
            group.create_dataset("iti", self.iti.shape, data=self.iti)
            group.create_dataset("position", self.position.shape, data=self.position)
            group.create_dataset(
                "reward_plus", self.reward_plus.shape, data=self.reward_plus
            )
            group.create_dataset(
                "trial_error", self.trial_error.shape, data=self.trial_error
            )
            group.create_dataset(
                "delay_time", self.delay_time.shape, data=self.delay_time
            )
            group.create_dataset("fix_time", self.fix_time.shape, data=self.fix_time)
            group.create_dataset(
                "fix_window_radius",
                self.fix_window_radius.shape,
                data=self.fix_window_radius,
            )
            group.create_dataset("idletime3", self.idletime3.shape, data=self.idletime3)
            group.create_dataset(
                "rand_delay_time", self.rand_delay_time.shape, data=self.rand_delay_time
            )
            group.create_dataset(
                "reward_dur", self.reward_dur.shape, data=self.reward_dur
            )
            group.create_dataset(
                "wait_for_fix", self.wait_for_fix.shape, data=self.wait_for_fix
            )
            # sacc
            group.create_dataset("sacc_code", self.sacc_code.shape, data=self.sacc_code)
            group.create_dataset(
                "fix_post_sacc_blank",
                self.fix_post_sacc_blank.shape,
                data=self.fix_post_sacc_blank,
            )
            group.create_dataset(
                "max_reaction_time",
                self.max_reaction_time.shape,
                data=self.max_reaction_time,
            )
            group.create_dataset("stay_time", self.stay_time.shape, data=self.stay_time)
            group.create_dataset(
                "fix_fp_t_time",
                self.fix_fp_t_time.shape,
                data=self.fix_fp_t_time,
            )
            group.create_dataset(
                "fix_fp_post_t_time",
                self.fix_fp_post_t_time.shape,
                data=self.fix_fp_post_t_time,
            )
            group.create_dataset(
                "fix_fp_pre_t_time",
                self.fix_fp_pre_t_time.shape,
                data=self.fix_fp_pre_t_time,
            )
            group.create_dataset("fix_close", self.fix_close.shape, data=self.fix_close)
            group.create_dataset("fix_far", self.fix_far.shape, data=self.fix_far)
            group.create_dataset("closeexc", self.closeexc.shape, data=self.closeexc)
            group.create_dataset(
                "excentricity", self.excentricity.shape, data=self.excentricity
            )
            group.create_dataset("farexc", self.farexc.shape, data=self.farexc)
            # dmts
            group.create_dataset("eye_ml", self.eye_ml.shape, data=self.eye_ml)
            group.create_dataset("condition", self.condition.shape, data=self.condition)
            group.create_dataset(
                "code_numbers", self.code_numbers.shape, data=self.code_numbers
            )
            group.create_dataset(
                "code_samples", self.code_samples.shape, data=self.code_samples
            )
            group.create_dataset(
                "code_times", self.code_times.shape, data=self.code_times
            )
            group.create_dataset(
                "stim_match", self.stim_match.shape, data=self.stim_match
            )
            group.create_dataset("samp_pos", self.samp_pos.shape, data=self.samp_pos)
            group.create_dataset(
                "stim_total", self.stim_total.shape, data=self.stim_total
            )
            group.create_dataset(
                "test_distractor", self.test_distractor.shape, data=self.test_distractor
            )
            group.create_dataset(
                "test_stimuli", self.test_stimuli.shape, data=self.test_stimuli
            )
            group.create_dataset(
                "sample_time", self.sample_time.shape, data=self.sample_time
            )
            group.create_dataset("test_time", self.test_time.shape, data=self.test_time)
        f.close()
