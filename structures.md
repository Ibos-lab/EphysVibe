## Information within each structure

### neuron_data
* date_time: date and time of the recording session
* subject:  name of the subject
* area: area recorded
* experiment: experiment number
* recording: recording number
##### sp
* sp_samples: array of shape (trials x time) containing the number of spikes at each ms in each trial.
* cluster_id: kilosort cluster ID.
* cluster_ch: electrode channel that recorded the activity of the cluster.
* cluster_group: "good" if it is a neuron or "mua" if it is a multi unit activity.
* cluster_number: number of good or mua.
* cluster_array_pos: position of the cluster in SpikeDate.sp_samples.
* cluster_depth: depth of the cluster.
##### bhv
* block: array of shape (trials) containing:
                    - 1 when is a DMTS trial
                    - 2 when is a saccade task trial
* trial_error: array of shape (trials) containing:
                    - 0 when is a correct trial
                    - n != 0 when is an incorrect trial. Each number correspond to different errors
* code_samples: array of shape (trials, events) containing the timestamp of the events
                            (timestamps correspond to sp_sample index).
* code_numbers: array of shape (trials, events) containing the codes of the events.
    - block 1:
        start_trial -> 9
        bar_hold -> 7
        fix_spot_on -> 35
        fixation -> 8
        sample_on -> 23
        sample_off -> 24
        test_on_1 -> 25
        test_off_1 -> 26
        test_on_2 -> 27
        test_off_2 -> 28
        test_on_3 -> 29
        test_off_3 -> 30
        test_on_4 -> 31
        test_off_4 -> 32
        test_on_5 -> 33
        test_off_5 -> 34
        end_trial -> 18
        bar_release -> 4
        fix_spot_off -> 36
        reward -> 96
    - block 2
        start_trial -> 9
        target_on -> 37
        target_off -> 38
        fix_spot_off -> 36
        eye_in_target -> 10
        correct_response -> 40
        end_trial -> 18
* position: array of shape (trials, 2) containing the position of the stimulus.
* pos_code: array of shape (trials) containing the position code of the stimulus.
  - block 1:               
    - 1: contralateral 
    - -1: ipsilateral to the recording side 
  - block 2, codes from 120 to 127 corresponding to the 8 target positions:
    - 127: middle right (10,0)
    - 126: upper right (7,7)
    - 125: upper middle(0,10)
    - 124: upper left (-7,7)
    - 123: middle left (-10,0)
    - 122: bottom left (-7,-7)
    - 121: bottom middle (0,-10)
    - 120: bottom right (7,-7)
* sample_id: array of shape (trials) containing the sample presented in each trial of block 1:
                        - 0: neutral sample
                        - 11: orientation 1, color 1
                        - 51: orientation 5, color 1
                        - 15: orientation 1, color 5
                        - 55: orientation 5, color 5
* test_stimuli: array of shape (trials,n_test_stimuli) containing the id of the test stimuli.
                        As in sample_id, first number correspond to orientation and second to color.
* test_distractor: array of shape (trials,n_test_stimuli) containing the id of the test distractor.
                        As in sample_id, first number correspond to orientation and second to color.



### bhv_data
* block: 
  * 1: DMTS task
  * 2: saccadic task 
* iti: duration of the intertrial interval
* position: array of shape (trials, 2) containing the position of the stimulus.
* pos_code: array of shape (trials) containing the position code of the stimulus.
  - block 1:               
    - 1: contralateral 
    - -1: ipsilateral to the recording side 
  - block 2, codes from 120 to 127 corresponding to the 8 target positions:
    - 127: middle right (10,0)
    - 126: upper right (7,7)
    - 125: upper middle(0,10)
    - 124: upper left (-7,7)
    - 123: middle left (-10,0)
    - 122: bottom left (-7,-7)
    - 121: bottom middle (0,-10)
    - 120: bottom right (7,-7)
* reward_plus: amount of extra reward if given
* trial_error: 
  - 0: correct trial 
  - 1: no bartouch
  - 2: no fixation
  - 3: break fixation
  - 4: no fixation
  - 5: bar release (before test period)
  - 6: false alarm (relase during test period)
  - 8: missed target
  - 10: break fixation during delay presentaton
* delay_time: duration of the delay
* fix_time: duration of the fixation
* fix_window_radius: radius of the fixation window
* idletime3:
* rand_delay_time: range of delay variation
* reward_dur: duration of the reward
* wait_for_fix: max time to fixate before the trial starts
##### saccadic task
* fix_post_sacc_blank
* max_reaction_time
* stay_time
* fix_fp_t_time
* fix_fp_post_t_time
* fix_fp_pre_t_time
* fix_close
* fix_far
* closeexc
* excentricity
* farexc
##### DMTS task
* eye_ml: position of the eye recorded with MonkeyLogic
* condition:
* code_numbers: code of the events
* code_times: timestamp of the events
* stim_match
* stim_total
* test_distractor
* test_stimuli
* sample_time
* test_time
* sample_id
