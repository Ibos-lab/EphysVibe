## Information within each structure

### bhv_data
* block: 
  * 1: DMTS task
  * 2: saccadic task 
* iti: duration of the intertrial interval
* position: position of the stimulus in the screen 
* pos_code: 
  * 1: inside rf (DMTS)
  * -1: outside rf (DMTS)
  * 127: middle right (10,0)
  * 126: upper right (7,7)
  * 125: upper middle(0,10)
  * 124: upper left (-7,7)
  * 123: middle left (-10,0)
  * 122: bottom left (-7,-7)
  * 121: bottom middle (0,-10)
  * 120: bottom right (7,-7)
* reward_plus: amount of extra reward if given
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


* 