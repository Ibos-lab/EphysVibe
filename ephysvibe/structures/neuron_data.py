import h5py
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from ephysvibe.task import task_constants
from ephysvibe.trials import align_trials
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


class NeuronData:
    def __init__(
        self,
        date_time: str,
        subject: str,
        area: str,
        experiment: str,
        recording: str,
        # --------sp-------
        sp_samples: np.ndarray,
        cluster_id: int,
        cluster_ch: int,
        cluster_group: str,
        cluster_number: int,
        cluster_array_pos: int,
        cluster_depth: int,
        # -------bhv-------
        block: np.ndarray,
        trial_error: np.ndarray,
        code_samples: np.ndarray,
        code_numbers: np.ndarray,
        position: np.ndarray,
        pos_code: np.ndarray,
        sample_id: np.ndarray,
        test_stimuli: np.ndarray,
        test_distractor: np.ndarray,
        **kwargs,
    ):
        """Initialize the class.

        This class contains information about each cluster.

        Args:
            date_time (str): date and time of the recording session
            subject (str):  name of the subject
            area (str): area recorded
            experiment (str): experiment number
            recording (str): recording number
            ------ sp ---------
            sp_samples (np.ndarray): array of shape (trials x time) containing the number of spikes at each ms in each trial.
            cluster_id (int): kilosort cluster ID.
            cluster_ch (int): electrode channel that recorded the activity of the cluster.
            cluster_group (str): "good" if it is a neuron or "mua" if it is a multi unit activity.
            cluster_number (int): number of good or mua.
            cluster_array_pos (int): position of the cluster in SpikeDate.sp_samples.
            cluster_depth (int): depth of the cluster.
            ------ bhv ---------
            block (np.ndarray): array of shape (trials) containing:
                                - 1 when is a DMTS trial
                                - 2 when is a saccade task trial
            trial_error (np.ndarray): array of shape (trials) containing:
                                - 0 when is a correct trial
                                - n != 0 when is an incorrect trial. Each number correspond to different errors
            code_samples (np.ndarray): array of shape (trials, events) containing the timestamp of the events
                                        (timestamps correspond to sp_sample index).
            code_numbers (np.ndarray): array of shape (trials, events) containing the codes of the events.
            position (np.ndarray): array of shape (trials, 2) containing the position of the stimulus.
            pos_code (np.ndarray): array of shape (trials) containing the position code of the stimulus.
                                    - for block 1: 1 is for 'in', -1 is for 'out' the receptive field
                                    - for block 2: codes from 120 to 127 corresponding to the 8 target positions.
            sample_id (np.ndarray): array of shape (trials) containing the sample presented in each trial of block 1:
                                    - 0: neutral sample
                                    - 11: orientation 1, color 1
                                    - 51: orientation 5, color 1
                                    - 15: orientation 1, color 5
                                    - 55: orientation 5, color 5
            test_stimuli (np.ndarray): array of shape (trials,n_test_stimuli) containing the id of the test stimuli.
                                    As in sample_id, first number correspond to orientation and second to color.
            test_distractor (np.ndarray): array of shape (trials,n_test_stimuli) containing the id of the test distractor.
                                    As in sample_id, first number correspond to orientation and second to color.
        """
        self.date_time = date_time
        self.subject = subject
        self.area = area
        self.experiment = experiment
        self.recording = recording
        # --------sp-------
        self.sp_samples = sp_samples
        self.cluster_id = cluster_id
        self.cluster_ch = cluster_ch
        self.cluster_group = cluster_group
        self.cluster_number = cluster_number
        self.cluster_array_pos = cluster_array_pos
        self.cluster_depth = cluster_depth
        # -------bhv-------
        self.block = block
        self.trial_error = trial_error
        self.code_samples = code_samples
        self.code_numbers = code_numbers
        self.position = position
        self.pos_code = pos_code
        self.sample_id = sample_id
        self.test_stimuli = test_stimuli
        self.test_distractor = test_distractor
        for key in kwargs:
            setattr(self, key, kwargs[key])
        # self._check_shapes()

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data and create class object
        bhv_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]
            bhv_data["date_time"] = group.attrs["date_time"]
            bhv_data["subject"] = group.attrs["subject"]
            bhv_data["area"] = group.attrs["area"]
            bhv_data["experiment"] = group.attrs["experiment"]
            bhv_data["recording"] = group.attrs["recording"]
            bhv_data["cluster_group"] = group.attrs["cluster_group"]
            for key, value in zip(group.keys(), group.values()):
                bhv_data[key] = np.array(value)
        f.close()
        return cls(**bhv_data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            group.attrs["date_time"] = self.__dict__.pop("date_time")
            group.attrs["subject"] = self.__dict__.pop("subject")
            group.attrs["area"] = self.__dict__.pop("area")
            group.attrs["experiment"] = self.__dict__.pop("experiment")
            group.attrs["recording"] = self.__dict__.pop("recording")
            group.attrs["cluster_group"] = self.__dict__.pop("cluster_group")

            for key, value in zip(self.__dict__.keys(), self.__dict__.values()):
                group.create_dataset(key, value.shape, data=value)
        f.close()

    def get_neuron_id(self):
        nid = f"{self.date_time}_{self.subject}_{self.area}_e{self.experiment}_r{self.recording}_{self.cluster_group}{self.cluster_number}"

        return nid

    def align_on(
        self,
        select_block: int,
        event: str,
        time_before: int,
        error_type: int,
        rf_stim_loc: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align spike data on a specific event and return the aligned data along with a trial mask.

        Args:
            select_block (int): Block number to select trials from.
            event (str): Event to align on.
            time_before (int): Time before the event to include in the alignment.
            error_type (int): Type of error trials to include (e.g., 0 for correct trials).
            rf_stim_loc (str): Position code for selecting trials.
                                - block 1: Must be one of "in", "out", "ipsi", or "contra".

        Raises:
            KeyError: If rf_stim_loc is not one of "in", "out", "ipsi", or "contra".
            ValueError: If select_block is not valid.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Aligned spike data.
                - Mask for selecting trials from the original data.
        """
        # Determine the event code based on the block number
        if select_block == 1:
            code = task_constants.EVENTS_B1[event]
            # Check rf_stim_loc value
            if rf_stim_loc == "in":
                rf_stim_loc, rfstim_loc = 1, self.rf_loc
            elif rf_stim_loc == "out":
                rf_stim_loc, rfstim_loc = -1, self.rf_loc
            elif rf_stim_loc == "contra":
                rf_stim_loc, rfstim_loc = 1, self.pos_code
            elif rf_stim_loc == "ipsi":
                rf_stim_loc, rfstim_loc = -1, self.pos_code
            else:
                raise KeyError(
                    "Invalid rf_stim_loc value: %s. Must be one of 'in', 'out', 'ipsi', or 'contra'."
                    % rf_stim_loc
                )
            # Create mask to select trials based on position, error type, and block
            mask = (
                (rfstim_loc == rf_stim_loc)
                & (self.trial_error == error_type)
                & (self.block == select_block)
            )

        elif select_block == 2:
            code = task_constants.EVENTS_B2[event]

            mask = (self.trial_error == error_type) & (self.block == select_block)
        else:
            raise ValueError(
                "Invalid select_block value: %d. Must be 1 or 2." % select_block
            )

        sp_samples_m = self.sp_samples[mask]

        # Find event occurrences in the code_numbers matrix
        code_mask = np.where(self.code_numbers[mask] == code, True, False)
        # Check if the event occurred in each trial
        trials_mask = np.any(code_mask, axis=1)
        # Get the sample indices where the event occurred and shift by time_before
        shifts = self.code_samples[mask][code_mask]
        shifts = (shifts - time_before).astype(int)
        # Align spike data based on the calculated shifts
        align_sp = align_trials.indep_roll(
            arr=sp_samples_m[trials_mask], shifts=-shifts, axis=1
        )
        # Create mask for selecting the trials from the original matrix size
        tr = np.arange(self.sp_samples.shape[0])
        complete_mask = np.isin(tr, tr[mask][trials_mask])

        return (align_sp, complete_mask)

    def edit_attributes(self, new_values: Dict):
        for attr_name, attr_value in zip(new_values.keys(), new_values.values()):
            setattr(self, attr_name, attr_value)

    def check_fr_loc(self, rf_loc_df: pd.DataFrame):
        """Add receptive field position to NeuronData object.

        Args:
            rf_loc (pd.DataFrame): DataFrame containing neuron IDs ('nid') and their corresponding 'rf_loc'.

        Returns:
            NeuronData: The modified NeuronData object.

        Raises:
            ValueError: If rf_loc is not 'ipsi' or 'contra'.
            IndexError: If the neuron ID (nid) is not found in rf_loc_df.
        """
        nid = self.get_neuron_id()
        # Filter the DataFrame and check if any results are found
        filtered_rf_loc = rf_loc_df[rf_loc_df["nid"] == nid]
        if filtered_rf_loc.empty:
            raise IndexError(f"No rf_loc found for neuron ID {nid}")
        rfloc = filtered_rf_loc["rf_loc"].values[0]
        pos_code = self.pos_code
        rf_loc = np.zeros(pos_code.shape, dtype=np.int8)
        if rfloc == "ipsi":
            rf_loc[pos_code == 1] = -1  # out
            rf_loc[pos_code == -1] = 1  # in
        elif rfloc == "contra":
            rf_loc[pos_code == 1] = 1
            rf_loc[pos_code == -1] = -1
        else:
            raise ValueError('rf_loc must be "ipsi" or "contra"')
        setattr(self, "rf_loc", rf_loc)
        return self

    def get_neu_align(
        self, params: List, delete_att: List = None, rf_loc: pd.DataFrame = None
    ):
        """Read, align, and add spiking activity to the NeuronData object.

        Args:
            params (List[dict]): List of dictionaries containing the following keys:
                - 'loc': str, location code ('in', 'out', 'ipsi', 'contra')
                - 'event': str, event name (e.g., 'sample_on')
                - 'time_before': int, time before event
                - 'time_after': int, time after event
                - 'select_block': int, block number
            delete_att (List[str], optional): List of attribute names to delete. Defaults to None.
            rf_loc (pd.DataFrame, optional): DataFrame containing neuron IDs ('nid') and their corresponding 'rf_loc'. Defaults to None.

        Returns:
            NeuronData: The modified NeuronData object with added spiking activity.
        """
        if rf_loc is not None:
            self = self.check_fr_loc(rf_loc)
        for it in params:
            # Alignment and extraction of spike and mask data
            sp, mask = self.align_on(
                select_block=it["select_block"],
                rf_stim_loc=it["loc"],
                event=it["event"],
                time_before=it["time_before"],
                error_type=0,
            )
            endt = it["time_before"] + it["time_after"]
            # Set name based on the event and rf/stimulus location
            att_name = f"{task_constants.EVENTS_B1_SHORT[it['event']]}_{it['loc']}"
            # Set attributes with appropriate data types
            setattr(self, f"sp_{att_name}", np.array(sp[:, :endt], dtype=np.int8))
            setattr(self, f"mask_{att_name}", np.array(mask, dtype=bool))
            setattr(
                self,
                f"time_before_{att_name}",
                np.array(it["time_before"], dtype=np.int32),
            )
        # Delete specified attributes if delete_att is provided
        if delete_att:
            for iatt in delete_att:
                if hasattr(self, iatt):
                    setattr(self, iatt, np.array([]))
                else:
                    print(
                        f"Warning: Attribute '{iatt}' does not exist and cannot be deleted."
                    )
        return self

    def plot_sp_b1(self, sp: Dict, conv: Dict):
        key1, key2 = list(sp.keys())
        # define kernel for convolution

        sampleco = {
            "0": "neutral",
            "11": "o1 c1",
            "15": "o1 c5",
            "51": "o5 c1",
            "55": "o5 c5",
        }
        t_before = 200
        # Iterate by sample and condition
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10), sharey=True)
        ax2 = [ax[0].twinx(), ax[1].twinx()]
        all_max_conv = 0
        all_max_trial = 0
        conv = {key1: conv[key1], key2: conv[key2]}
        sp = {key1: sp[key1], key2: sp[key2]}
        for i_ax, cond in enumerate([key1, key2]):
            count_trials = 0
            max_conv = 0
            for i_sample in conv[cond].keys():
                max_conv = (
                    np.max(conv[cond][i_sample])
                    if np.max(conv[cond][i_sample]) > max_conv
                    else max_conv
                )
                time = np.arange(0, len(conv[cond][i_sample])) - t_before
                ax[i_ax].plot(
                    time,
                    conv[cond][i_sample],
                    color=task_constants.PALETTE_B1[i_sample],
                )
                # Plot spikes
                count_t = len(sp[cond][i_sample])
                rows, cols = np.where(sp[cond][i_sample] >= 1)
                ax2[i_ax].scatter(
                    cols - t_before,
                    rows + count_trials,
                    marker="|",
                    alpha=1,
                    color=task_constants.PALETTE_B1[i_sample],
                    label=sampleco[i_sample],
                )
                count_trials += count_t
            all_max_conv = max_conv if max_conv > all_max_conv else all_max_conv
            all_max_trial = (
                count_trials if count_trials > all_max_trial else all_max_trial
            )
            ax[i_ax].set_title(cond, fontsize=15)
        for i_ax in range(2):
            ax[i_ax].set_ylim(0, all_max_conv + all_max_trial + 5)
            ax[i_ax].set_yticks(np.arange(0, all_max_conv + 5, 10))
            ax2[i_ax].set_yticks(np.arange(-all_max_conv - 5, all_max_trial))
            plt.setp(ax2[i_ax].get_yticklabels(), visible=False)
            plt.setp(ax2[i_ax].get_yaxis(), visible=False)
            ax[i_ax].set_xticks([0, 450, 1250, 1700])
            ax[i_ax].set_xticklabels(["0", "450", "0", "450"])

            ax[i_ax].vlines(
                [0, 450, 450 + 400 + 400],
                0,
                all_max_conv + all_max_trial + 5,
                color="k",
                linestyles="-",
            )
            ax[i_ax].vlines(
                [850],
                0,
                all_max_conv + all_max_trial + 5,
                color="k",
                linestyles="--",
            )
            ax2[i_ax].spines["right"].set_visible(False)
            ax2[i_ax].spines["top"].set_visible(False)
            ax[i_ax].spines["right"].set_visible(False)
            ax[i_ax].spines["top"].set_visible(False)
        ax[0].set(xlabel="Time (ms)", ylabel="Average firing rate")
        ax2[1].set(xlabel="Time (ms)", ylabel="trials")
        ax[1].set_xlabel(xlabel="Time (ms)", fontsize=18)
        # ax[1].set_xticks(fontsize=15)
        ax[0].set_xlabel(xlabel="Time (ms)", fontsize=18)
        ax[0].set_ylabel(ylabel="Average firing rate", fontsize=15)
        for xtick in ax[0].xaxis.get_major_ticks():
            xtick.label1.set_fontsize(15)
        for ytick in ax[0].yaxis.get_major_ticks():
            ytick.label1.set_fontsize(15)
        for xtick in ax[1].xaxis.get_major_ticks():
            xtick.label1.set_fontsize(15)
        ax2[1].legend(
            fontsize=15,
            scatterpoints=5,
            columnspacing=0.5,
            framealpha=0,
            loc="upper right",
        )
        fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.8)
        fig.suptitle(
            "%s: %s %d " % (self.area.upper(), self.cluster_group, self.cluster_number),
            x=0.05,
            y=0.99,
            fontsize=15,
        )
        return fig

    def plot_sp_b2(self, sp_pos, conv_pos, max_n_tr, conv_max, visual_rf=True, inout=1):
        start = -200
        PAC = task_constants.CODES_AND_POS
        # Get samples position in b1 and b2
        b1_mask = self.block == 1
        b2_mask = self.block == 2
        in_mask = self.pos_code == 1
        u_pos, u_count = np.unique(
            self.position[np.logical_and(b1_mask, in_mask)], axis=0, return_counts=True
        )
        imax = np.argmax(u_count)
        x_pos_b1, y_pos_b1 = u_pos[imax][0][0], u_pos[imax][0][1]
        pos_b2 = np.unique(self.position[b2_mask][:, 0], axis=0)
        x_pos_b2, y_pos_b2 = pos_b2[:, 0], pos_b2[:, 1]
        # Start plot
        fig, axs = plt.subplots(figsize=(8, 8), sharex=True, sharey=True)
        axs.set_axis_off()
        for code in sp_pos.keys():
            axis = PAC[code][1]
            sp = sp_pos[code]
            conv = conv_pos[code]
            ax = plt.subplot2grid((3, 3), (axis[0], axis[1]))
            time = np.arange(0, len(conv)) + start
            ax2 = ax.twinx()
            # ----- plot conv----------
            ax.plot(time, conv, color="navy")
            # ----- plot spikes----------
            rows, cols = np.where(sp >= 1)
            cols = cols + start
            rows = rows + rows * 2
            ax2.scatter(cols, rows, marker="|", alpha=1, color="grey")
            ax.set_title("Code %s" % (code), fontsize=8)
            ax.set_ylim(0, conv_max + max_n_tr * 3)
            ax.set_yticks(np.arange(0, conv_max, 10))
            ax2.set_ylim(-conv_max, max_n_tr)
            ax2.set_yticks(np.arange(-conv_max, max_n_tr * 3, 10))
            if visual_rf:
                bl_sp = np.mean(sp[:, :200], axis=1)
                tg_sp = np.mean(sp[:, 200:400], axis=1)
                n_tr = len(bl_sp)
                fr = np.sum(tg_sp != 0)
                if n_tr < 5 or fr < 5:
                    p = False
                else:
                    p = stats.ttest_rel(bl_sp, tg_sp)
                    p = p[1] < 0.05
                if p:
                    ax.set_facecolor("bisque")
            plt.setp(ax2.get_yticklabels(), visible=False)
            ax.vlines(
                [0, 100, 1100],
                0,
                conv_max + max_n_tr * 3,
                color="k",
                linestyles="dashed",
            )
            if code == "122":
                ax.set_ylabel(ylabel="Average firing rate", fontsize=10, loc="bottom")
                ax.set_xlabel(xlabel="Time (s)", fontsize=10)
        ax = plt.subplot2grid((3, 3), (1, 1))
        ax.scatter(x_pos_b2, y_pos_b2)
        ax.scatter(inout * x_pos_b1, inout * y_pos_b1)
        fig.tight_layout(pad=0.4, h_pad=0.2, w_pad=0.2)
        fig.suptitle(
            "%s: %s %s %d "
            % (self.date_time, self.area, self.cluster_group, self.cluster_number),
            x=0.5,
            y=1.05,
            fontsize=12,
        )
        return fig

    def get_performance(self):
        eventsb1 = task_constants.EVENTS_B1
        b1mask = self.block == 1
        ntest = self.test_stimuli.shape[1]
        bhv_matrix = np.full((np.sum(b1mask), ntest), np.nan)
        match_matrix = np.full((np.sum(b1mask), ntest), np.nan)
        # Check if there was a bar release in any moment of the trial
        maskbar = align_trials.indep_roll(
            arr=self.code_numbers[b1mask] == eventsb1["bar_release"],
            shifts=np.array([-1] * sum(b1mask)),
            axis=1,
        )
        rowbar, colbar = np.where(maskbar)
        testlist = ["test_on_" + str(i + 1) for i in range(ntest)]
        for itest, event in enumerate(testlist):
            evmask = self.code_numbers[b1mask][maskbar] == eventsb1[event]
            # 1 if bar released in test presentation n
            bhv_matrix[rowbar[evmask], itest] = 1

        # Check if break fixation
        idx_brfix = np.where(
            np.sum(
                np.sum(
                    (
                        self.code_numbers[b1mask] == eventsb1["fix_break"],
                        self.code_numbers[b1mask] == eventsb1["fix_spot_off"],
                    ),
                    axis=0,
                ),
                axis=1,
            )
            == 2
        )[0]
        _, c_break = np.where(
            self.code_numbers[b1mask][idx_brfix] == eventsb1["fix_break"]
        )
        _, c_fixoff = np.where(
            self.code_numbers[b1mask][idx_brfix] == eventsb1["fix_spot_off"]
        )
        idx_breakfix = np.where(c_break < c_fixoff)[0]
        if len(idx_breakfix) != 0:
            bhv_matrix[idx_brfix[idx_breakfix]] = -2
        # Check if release bar error
        idx_rbe = np.where(
            np.sum(
                np.sum(
                    (
                        self.code_numbers[b1mask] == eventsb1["bar_release"],
                        self.code_numbers[b1mask] == eventsb1["test_on_1"],
                    ),
                    axis=0,
                ),
                axis=1,
            )
            == 2
        )[0]
        _, c_rbe = np.where(
            self.code_numbers[b1mask][idx_rbe] == eventsb1["bar_release"]
        )
        _, c_t1on = np.where(
            self.code_numbers[b1mask][idx_rbe] == eventsb1["test_on_1"]
        )

        idx_rbarerror = np.where(c_rbe < c_t1on)[0]
        if len(idx_rbarerror) != 0:
            bhv_matrix[idx_rbe[idx_rbarerror]] = -2

        bhv_matrix = np.where(bhv_matrix == 1, 2, bhv_matrix)

        # Trials where all tests were presented but there was no response (catch (all CR:0))
        maskcatch1 = (
            np.sum(
                self.test_stimuli[b1mask] != self.sample_id[b1mask].reshape(-1, 1),
                axis=1,
            )
            == self.test_stimuli.shape[1]
        )
        maskcatch2 = np.all(~np.isnan(self.test_stimuli[b1mask]), axis=1)
        maskcatch3 = np.logical_and(maskcatch1, maskcatch2)
        catch_trial = np.logical_and(maskcatch3, self.sample_id[b1mask] != 0)
        match_matrix[catch_trial] = 0
        test_match = self.test_stimuli[b1mask] == self.sample_id[b1mask].reshape(-1, 1)
        match_matrix[test_match] = 1
        nanmask = np.logical_and(np.isnan(bhv_matrix), np.isnan(match_matrix))
        # bhv_matrix[test_match] = np.nansum([bhv_matrix[test_match],np.array([1]*np.sum(test_match))],axis=0)
        bhv_matrix = np.nansum([bhv_matrix, match_matrix], axis=0)
        bhv_matrix[nanmask] = np.nan

        r, c = np.where(bhv_matrix == 3)
        mask = np.where(bhv_matrix == 3, True, False)
        cols = np.arange(bhv_matrix.shape[1])  # Column indices
        maskcols = cols <= (c - 1).reshape(-1, 1)  # Create a mask based on idx
        mask[r] = maskcols
        bhv_matrix[mask] = 0

        return bhv_matrix

    @classmethod
    def get_tests_tr_bhv_clasification(
        cls,
        sp: np.ndarray,
        performance: np.ndarray,
        codes: np.ndarray,
        code_samples: np.ndarray,
        code_numbers: np.ndarray,
        time_before: int = 500,
        time_after: int = 1000,
        name_code: Dict = {},
    ) -> np.ndarray:
        tests = {}
        for iperf in codes:
            key = iperf if not bool(name_code) else name_code[iperf]
            rM, cM = np.where(performance == iperf)
            # Select the sample when the event ocurred
            shifts = code_samples[rM][code_numbers[rM] == (cM * 2 + 25).reshape(-1, 1)]
            shifts = (shifts - time_before).astype(int)
            # align sp
            align_sp = align_trials.indep_roll(arr=sp[rM], shifts=-shifts, axis=1)
            tests[key] = align_sp[:, : time_before + time_after].astype(np.int8)
            tests[str(key) + "_" + "pos"] = cM
        return tests
