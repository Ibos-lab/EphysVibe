% Restructures the .bhv file from ML and saves it as .mat
% You must enter datapath all the way to experiment1 for example :
% datapath='/envau/work/invibe/USERS/IBOS/data/openephys/2022-05-25_10-29-38/Record Node 102/experiment1/';
% and the savepath
%% 
addpath(genpath('/home/INT/losada.c/Documents/codes/matlab'))  
savepath='/envau/work/invibe/USERS/IBOS/openephys/Riesling/2023-03-03_10-59-32/Record Node 102/experiment1/recording1/';
%%datapath=uigetdir;
%cd(datapath)

cd '/envau/work/invibe/USERS/IBOS/openephys/Riesling/2023-03-03_10-59-32/Record Node 102/experiment1/recording1/'
directory=pwd;
bhvfiles=dir('*.bhv2');
%path = "\\envau_cifs.intlocal.univ-amu.fr\work\invibe\USERS\IBOS\openephys\221201_TSCM_5cj_5cl_Riesling.bhv2";
bhv=mlread(bhvfiles.name);
%bhv=mlread(path);

%%
% Define constants
n_test_stim = 5;
flag_n_test = 0;
n_trials = size(bhv,2);
n_block2 = size(find(extractfield(bhv,'Block')==2),2);
% Iterate by trials to get max dim
max_codes = 0;
max_eyes = 0;
for i_trial=1:n_trials
    len_codes = length(bhv(i_trial).BehavioralCodes.CodeTimes);
    len_eyes = length(bhv(i_trial).AnalogData.Eye);
    if len_codes > max_codes
        max_codes = len_codes;
    end
    if len_eyes > max_eyes
        max_eyes = len_eyes;
    end
end
% Define variables
CodeTimes = nan(n_trials,max_codes);
CodeNumbers = nan(n_trials,max_codes);
Eye = nan(n_trials,3,max_eyes);%nan(n_trials,max_eyes,3);
PupilSize =[];
Position = nan(n_trials,2,2);
% bhv.UserVars
TestStimuli = nan(n_trials,n_test_stim); % 5 is de max number of stim in each trial
TestDistractor = nan(n_trials,n_test_stim);
% bhv.VariableChanges
reward_dur =nan(n_trials,1);
fix_time =[];
fix_window_radius =[];
ITI_value =[];
wait_for_fix =[];
Reward_plus =[];
sample_time =[];
delay_time =[];
rand_delay_time =[];
test_time =[];
idletime3 =[];
farexc =[];
Fixfar =[];
closeexc =[];
Fixclose =[];
excentricity =[];
Fix_FP_pre_T_time =[];
Fix_FP_T_time =[];
Fix_FP_post_T_time =[];
fix_post_sacc_blank =[];
max_reaction_time =[];
stay_time =[];
% bhv.TaskObject.Attribute
SampleId = [];%nan(n_trials,1);
% bhv.TaskObject.CurrentConditionInfo
%SaccCode =[];
PosCode =[];
Match =[];
Total =[];

% Iterate by trials and concatenate
for i_trial=1:n_trials 
    n_codes = size(bhv(i_trial).BehavioralCodes.CodeTimes,1);
    CodeTimes(i_trial,1:n_codes) = bhv(i_trial).BehavioralCodes.CodeTimes;
    CodeNumbers(i_trial,1:n_codes) = bhv(i_trial).BehavioralCodes.CodeNumbers;
    PupilSize= bhv(i_trial).AnalogData.General.Gen1;
    if isempty(PupilSize)
        PupilSize = nan(length(bhv(i_trial).AnalogData.Eye),1);
    end
    eye_pupil = cat(2,bhv(i_trial).AnalogData.Eye,PupilSize);
    n_eyes = size(bhv(i_trial).AnalogData.Eye,1);
    Eye(i_trial,:,1:n_eyes) = transpose(eye_pupil);
    % bhv.VariableChanges
    
    fix_time = cat(1,fix_time,bhv(i_trial).VariableChanges.fix_time);
    fix_window_radius = cat(1,fix_window_radius,bhv(i_trial).VariableChanges.fix_window_radius);
    ITI_value = cat(1,ITI_value,bhv(i_trial).VariableChanges.ITI_value);
    wait_for_fix = cat(1,wait_for_fix,bhv(i_trial).VariableChanges.wait_for_fix);
    Reward_plus = cat(1,Reward_plus,bhv(i_trial).VariableChanges.Reward_plus);
    sample_time = cat(1,sample_time,bhv(i_trial).VariableChanges.sample_time);
    delay_time = cat(1,delay_time,bhv(i_trial).VariableChanges.delay_time);
    rand_delay_time = cat(1,rand_delay_time,bhv(i_trial).VariableChanges.rand_delay_time);
    test_time = cat(1,test_time,bhv(i_trial).VariableChanges.test_time);
    idletime3 = cat(1,idletime3,bhv(i_trial).VariableChanges.idletime3);
    
    farexc = cat(1,farexc,bhv(i_trial).VariableChanges.farexc);
    Fixfar = cat(1,Fixfar,bhv(i_trial).VariableChanges.Fixfar);
    closeexc = cat(1,closeexc,bhv(i_trial).VariableChanges.closeexc);
    Fixclose = cat(1,Fixclose,bhv(i_trial).VariableChanges.Fixclose);
    excentricity = cat(1,excentricity,bhv(i_trial).VariableChanges.excentricity);
    Fix_FP_pre_T_time = cat(1,Fix_FP_pre_T_time,bhv(i_trial).VariableChanges.Fix_FP_pre_T_time);
    Fix_FP_T_time = cat(1,Fix_FP_T_time,bhv(i_trial).VariableChanges.Fix_FP_T_time);
    Fix_FP_post_T_time = cat(1,Fix_FP_post_T_time,bhv(i_trial).VariableChanges.Fix_FP_post_T_time);
    fix_post_sacc_blank = cat(1,fix_post_sacc_blank,bhv(i_trial).VariableChanges.fix_post_sacc_blank);
    max_reaction_time = cat(1,max_reaction_time,bhv(i_trial).VariableChanges.max_reaction_time);
    stay_time = cat(1,stay_time,bhv(i_trial).VariableChanges.stay_time);
    % bhv.TaskObject.CurrentConditionInfo
    if bhv(i_trial).Block == 2
        % SaccCode = cat(1,SaccCode,bhv(i_trial).TaskObject.CurrentConditionInfo.Code);
        PosCode = cat(1,PosCode,bhv(i_trial).TaskObject.CurrentConditionInfo.Code);
        Match = cat(1,Match,[nan]);
        Total = cat(1,Total,[nan]);
        SampleId=cat(1,SampleId,[nan]);
        %n_pos = size(bhv(i_trial).ObjectStatusRecord.Position{1},1);
        Position(i_trial,1,:) = bhv(i_trial).ObjectStatusRecord.Position{1}(2,:);
        reward_dur(i_trial) = bhv(i_trial).VariableChanges.reward_sacc_dur;
    else % Block == 1
        % SaccCode = cat(1,SaccCode,[nan]);
        PosCode = cat(1,PosCode,bhv(i_trial).TaskObject.CurrentConditionInfo.pos);
        Match = cat(1,Match,bhv(i_trial).TaskObject.CurrentConditionInfo.match);
        Total = cat(1,Total,bhv(i_trial).TaskObject.CurrentConditionInfo.total);
        sample =  bhv(i_trial).TaskObject.Attribute{2}{2}(end-5:end-2);
        SampleId = cat(1,SampleId,str2num(strcat(sample(2),sample(end))));
        if size(bhv(i_trial).ObjectStatusRecord.Position,2) ~= 0
            Position(i_trial,:,:) = cat(1,bhv(i_trial).ObjectStatusRecord.Position{1}(2,:),bhv(i_trial).ObjectStatusRecord.Position{1}(end,:));
        end
        reward_dur(i_trial) = bhv(i_trial).VariableChanges.reward_dur;
        n_test = (length(fieldnames(bhv(i_trial).UserVars))-1)/2; 
        if n_test == 7 || n_test == 6
            n_test = 5;
        end
        if n_test == 5 
            flag_n_test = 1;
            s1 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1(end-8), bhv(i_trial).UserVars.Stim_Filename_1(end-4)));
            s2 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2(end-8), bhv(i_trial).UserVars.Stim_Filename_2(end-4)));
            s3 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_3(end-8), bhv(i_trial).UserVars.Stim_Filename_3(end-4)));
            s4 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_4(end-8), bhv(i_trial).UserVars.Stim_Filename_4(end-4)));
            s5 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_5(end-8), bhv(i_trial).UserVars.Stim_Filename_5(end-4)));
            s1_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1d(end-8), bhv(i_trial).UserVars.Stim_Filename_1d(end-4)));
            s2_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2d(end-8), bhv(i_trial).UserVars.Stim_Filename_2d(end-4)));
            s3_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_3d(end-8), bhv(i_trial).UserVars.Stim_Filename_3d(end-4)));
            s4_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_4d(end-8), bhv(i_trial).UserVars.Stim_Filename_4d(end-4)));
            s5_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_5d(end-8), bhv(i_trial).UserVars.Stim_Filename_5d(end-4)));
            TestStimuli(i_trial,:) = [s1 s2 s3 s4 s5];
            TestDistractor(i_trial,:) = [s1_d s2_d s3_d s4_d s5_d];
        elseif n_test == 4
            s1 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1(end-8), bhv(i_trial).UserVars.Stim_Filename_1(end-4)));
            s2 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2(end-8), bhv(i_trial).UserVars.Stim_Filename_2(end-4)));
            s3 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_3(end-8), bhv(i_trial).UserVars.Stim_Filename_3(end-4)));
            s4 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_4(end-8), bhv(i_trial).UserVars.Stim_Filename_4(end-4)));
            s1_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1d(end-8), bhv(i_trial).UserVars.Stim_Filename_1d(end-4)));
            s2_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2d(end-8), bhv(i_trial).UserVars.Stim_Filename_2d(end-4)));
            s3_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_3d(end-8), bhv(i_trial).UserVars.Stim_Filename_3d(end-4)));
            s4_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_4d(end-8), bhv(i_trial).UserVars.Stim_Filename_4d(end-4)));
            TestStimuli(i_trial,1:n_test) = [s1 s2 s3 s4];
            TestDistractor(i_trial,1:n_test) = [s1_d s2_d s3_d s4_d];
        elseif n_test == 3
            s1 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1(end-8), bhv(i_trial).UserVars.Stim_Filename_1(end-4)));
            s2 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2(end-8), bhv(i_trial).UserVars.Stim_Filename_2(end-4)));
            s3 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_3(end-8), bhv(i_trial).UserVars.Stim_Filename_3(end-4)));
            s1_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1d(end-8), bhv(i_trial).UserVars.Stim_Filename_1d(end-4)));
            s2_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2d(end-8), bhv(i_trial).UserVars.Stim_Filename_2d(end-4)));
            s3_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_3d(end-8), bhv(i_trial).UserVars.Stim_Filename_3d(end-4)));
            TestStimuli(i_trial,1:n_test) = [s1 s2 s3];
            TestDistractor(i_trial,1:n_test) = [s1_d s2_d s3_d];
        elseif n_test == 2
            s1 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1(end-8), bhv(i_trial).UserVars.Stim_Filename_1(end-4)));
            s2 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2(end-8), bhv(i_trial).UserVars.Stim_Filename_2(end-4)));
            s1_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1d(end-8), bhv(i_trial).UserVars.Stim_Filename_1d(end-4)));
            s2_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_2d(end-8), bhv(i_trial).UserVars.Stim_Filename_2d(end-4)));
            TestStimuli(i_trial,1:n_test) = [s1 s2];
            TestDistractor(i_trial,1:n_test) = [s1_d s2_d]; 
        elseif n_test == 1
            s1 = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1(end-8), bhv(i_trial).UserVars.Stim_Filename_1(end-4)));
            s1_d = str2num(strcat(bhv(i_trial).UserVars.Stim_Filename_1d(end-8), bhv(i_trial).UserVars.Stim_Filename_1d(end-4)));           
            TestStimuli(i_trial,1:n_test) = [s1];
            TestDistractor(i_trial,1:n_test) = [s1_d]; 
        end
    end       
end
%% Asing values to the new struct
New.Block = reshape(extractfield(bhv,'Block'),[],1);
New.Condition = reshape(extractfield(bhv,'Condition'),[],1); 
New.TrialError = reshape(extractfield(bhv,'TrialError'),[],1); 
% Values from the loop
% bhv.UserVars
if flag_n_test == 0
    TestStimuli = TestStimuli(:,1:4);
    TestDistractor = TestDistractor(:,1:4);
end
New.TestStimuli = TestStimuli;
New.TestDistractor = TestDistractor;
% bhv.VariableChanges
New.CodeTimes = CodeTimes;
New.CodeNumbers = CodeNumbers;
New.Eye = Eye;
New.Position = Position;
New.reward_dur = reward_dur;
New.fix_time = fix_time;
New.fix_window_radius = fix_window_radius;
New.ITI_value = ITI_value;
New.wait_for_fix = wait_for_fix;
New.Reward_plus = Reward_plus;
New.sample_time = sample_time;
New.delay_time = delay_time;
New.rand_delay_time = rand_delay_time;
New.test_time = test_time;
New.idletime3 = idletime3;

New.farexc = farexc;
New.Fixfar = Fixfar;
New.closeexc = closeexc;
New.Fixclose = Fixclose;
New.excentricity = excentricity;
New.Fix_FP_pre_T_time = Fix_FP_pre_T_time;
New.Fix_FP_T_time = Fix_FP_T_time;
New.Fix_FP_post_T_time = Fix_FP_post_T_time;
New.fix_post_sacc_blank = fix_post_sacc_blank;
New.max_reaction_time = max_reaction_time;
New.stay_time = stay_time;
% bhv.TaskObject.CurrentConditionInfo
% New.SaccCode = SaccCode;
New.SampPosCode = PosCode;
New.StimMatch = Match;
New.StimTotal = Total;
New.SampleId = SampleId;
%%
% cd(savepath)
% disp(['saving data files...' [bhvfiles.name(1:end-4) 'mat']]);
% save([bhvfiles.name(1:end-4) 'mat'],'New','-v7.3' )
% %clear all
% disp('done')
