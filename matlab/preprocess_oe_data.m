% function preprocess_oe_data(datapath)
% Loads and preprocesses openephys recording
% Merge Monkeylogic BHV files with electrophy data from Binary format of open ephys
% in structure data.BHV and data.NEURO respectively
% You must enter datapath all the way to experiment1 for example :
% datapath='/envau/work/invibe/USERS/IBOS/data/openephys/2022-05-25_10-29-38/Record Node 102/experiment1/';
% This will probably have to be changed once we have other recording nodes
% corresponding to other cortical areas
% Warning, name of the recording area is not specified yet.
% It requires sevearl toolboxes: matlab analyses for open ephys:
% https://github.com/open-ephys/analysis-tools
% npy-matlab for reading numpy files:
% https://github.com/kwikteam/npy-matlab
% mlread for reading bhv2 files from MonkeyLogic:
% https://monkeylogic.nimh.nih.gov/
%
% Function created by Guilhem Ibos, June 28th 2022

addpath(genpath('C:\Users\camil\Documents\int\open-ephys-matlab-tools-master'))
% addpath(genpath('/envau/work/invibe/USERS/IBOS/open-ephys-matlab-tools-master'))

% "\\envau_cifs.intlocal.univ-amu.fr\work\invibe\USERS\IBOS\data\openephys\2022-05-25_10-29-38\Record Node 102\experiment1"
%cd 'C:\Users\camil\Documents\int\data\openephys\2022-05-25_10-29-38\Record Node 102\experiment1'
cd 'C:\Users\camil\Documents\int\data\openephys\2022-06-09_10-40-34\Record Node 102\experiment1'
% cd /home/spikesorting/Documents/data/IBOS

addpath(genpath('C:\Users\camil\Documents\int\npy-matlab-master\npy-matlab-master'))  

datapath=uigetdir;
cd(datapath)
directory=pwd;



%% low pass filter continuous data
cut_fq=250;
fs=30000;
P=6;
[b,a] = butter(P,cut_fq/(fs/2));


%%



for Count_record=1:Num_recordings
    clear data
   
    %% enter depth of the electrode manually
    prompt='Electrode depth (lowest contact)';
    depth=inputdlg(prompt);
    
    %%
    disp(['Recording # ' num2str(Count_record) '/' num2str(Num_recordings)]);
    recdir=[directory '/recording' num2str(Count_record)];
    cd(recdir);
    bhvfiles=dir('*.bhv2');
    BHV=mlread(bhvfiles.name);

    disp('loading continuous data...');
    tic
    D=load_open_ephys_binary('structure.oebin','continuous',1, 'mmap');
    toc
    disp('loading events...');
    E=load_open_ephys_binary('structure.oebin','events',1);
    %% 1st day selected channels are different
    Selected_channels=[1:34];% 65:66];
    if (E.Timestamps(1)-10*fs)>0
        if ~isempty(find(D.Timestamps==E.Timestamps(1)-10*fs))
            Starttime=find(D.Timestamps==E.Timestamps(1)-10*fs);
        else
            Starttime=1;
        end
    else
        Starttime=1;
    end
    num_samples=size(D.Data.Data.mapped,2);
    Filtered_Timestamps=int64(D.Timestamps(Starttime:30:end));
    raw_timestamps  =   D.Timestamps;
    
    %%
    %filtered_cont=nan(size(Starttime:30:num_samples));
    disp('filtering LFPs')
    
    
    % BEWARE THE SELECTED CHANNELS
    %%%If not enough memory space (niolon or personnal computer, decompose
    %%%by recording channel
    filtered_cont=nan(size(Selected_channels,2), size(Filtered_Timestamps,1));
    Decomp=1;
    tic
    if Decomp==1
        for i=1:size(Selected_channels,2)
            tmp=filter(b,a,D.Data.Data(1).mapped(Selected_channels(i),:),[], 2);
            filtered_cont(i,:)=int64(tmp(Starttime:30:end));
            clear tmp
        end
    else
        tmp=filter(b,a,D.Data.Data(1).mapped(Selected_channels(1:32),:),[], 2);
        filtered_cont(1:32,:)=int64(tmp(:,Starttime:30:end));
    end
    toc
    %%

    %%
    disp('Reconstructing 8 bit words')
    Real_strobes=E.Timestamps(E.ChannelIndex==8 & E.Data>0 & E.Timestamps>0);


    channel=E.ChannelIndex(E.ChannelIndex~=8 & E.Timestamps~=0);
    state=E.Data(E.ChannelIndex~=8 & E.Timestamps~=0);
    TS=E.Timestamps(E.ChannelIndex~=8 & E.Timestamps~=0);

    Current_8code=('00000000');
    FullWord=[];

    for i=1:size(Real_strobes,1)  %     find codes between two strobes
        if i==1
            Status_change=find(TS<Real_strobes(1)); %     find indices before the first strobe
            for j=1:size(Status_change,1)
                if state(Status_change(j))>0
                    Current_8code(9-channel(Status_change(j)))='1';
                else
                    Current_8code(9-channel(Status_change(j)))='0';
                end
            end
            %     Change the 8 bit status according to each modified channel
            FullWord(i)=bin2dec(Current_8code);
        else
            Status_change=find(TS>Real_strobes(i-1) & TS<Real_strobes(i));

            for j=1:size(Status_change,1)
                if state(Status_change(j))>0
                    Current_8code(9-channel(Status_change(j)))='1';
                else
                    Current_8code(9-channel(Status_change(j)))='0';
                end
            end
            FullWord(i)=bin2dec(Current_8code);
        end
    end

    NumTrial  = size(BHV,2);
    BHVcodes  = [];

    for ll=1:NumTrial
        BHVcodes=[BHVcodes  BHV(ll).BehavioralCodes.CodeNumbers'];
    end
    
    if size(FullWord,2)~=size(Real_strobes,1)
        disp('Warning, Strobe and codes number do not match')
        disp(['Strobes =', num2str(size(Real_strobes,1))])
        disp(['codes number =', num2str(size(FullWord,2))])
    else
        disp('Strobe and codes number do match')
        disp(['Strobes =', num2str(size(Real_strobes,1))])
        disp(['codes number =', num2str(size(FullWord,2))])
    end

    if size(BHVcodes,2)-size(FullWord,2)~=0
        disp('Warning, ML and OE code numbers do not match')
    else
        disp('ML and OE code numbers do match')
        if sum(BHVcodes-FullWord)~=0
            disp('Warning, ML and OE codes are different')
        else
            disp('ML and OE codes are the same')
        end
    end

    %% get spikes info (timing, id, position)
    disp('getting spikes...')
    spikedir=[recdir '/continuous/Rhythm_FPGA-100.0/kilosort3'];
    cd(spikedir);
    Sp_filename = [spikedir '/spike_times.npy'];

    spiketimes=readNPY(Sp_filename);
    spiketimes_clusters_id = readNPY([spikedir '/spike_clusters.npy']);
    
    %%%% This part could be improved and might be specific of each
    %%%% installation. Check cluster_info.tsv header for good formating
    fid=fopen('cluster_info.tsv');
    C = textscan(fid, '%f %f %f %s %f %f %f %f %s %f %f', 'HeaderLines', 1);

    %cd ../..

    spiketimestamps =   raw_timestamps(spiketimes);

    disp('building data files...');
    clear data
    data.BHV=BHV;
    data.NEURO.LFP.samples      =   filtered_cont(1:32,:);
    data.NEURO.LFP.timestamps   =   Filtered_Timestamps;

    data.NEURO.Eyes.samples     =   filtered_cont(33:34,:);

    data.NEURO.CodeNumbers      =   FullWord;
    data.NEURO.CodeTimes        =   Real_strobes;

    G=0;
    M=0;
    for j=1:size(C{1},1)
        if strcmp(C{9}{j},'good')
            G=G+1;
            data.NEURO.Neuron{G}.times        = spiketimestamps(spiketimes_clusters_id==C{1}(j));
            data.NEURO.Neuron{G}.clustersid   = C{1}(j);
            data.NEURO.Neuron{G}.clustersch   = C{6}(j);
            data.NEURO.Neuron{G}.clustersgroup= C{9}(j);
            data.NEURO.Neuron{G}.clusterdepth = C{7}(j);

        elseif   strcmp(C{9}{j},'mua')
            M=M+1;
            data.NEURO.MUA{M}.times       =   spiketimestamps(spiketimes_clusters_id==C{1}(j));
            data.NEURO.MUA{M}.clustersid  =   C{1}(j);
            data.NEURO.MUA{M}.clustersch  =   C{6}(j);
            data.NEURO.MUA{M}.clustersgroup=  C{9}(j);
            data.NEURO.MUA{M}.clusterdepth=   C{7}(j);
        end
    end
%    data.NEURO.depth=str2num(depth{1});
%    savepath='/home/spikesorting/Documents/data/IBOS/cells_struct/KS';
%    cd(savepath)
    disp(['saving data files...' [bhvfiles.name(1:end-4) 'mat']]);
    %save([bhvfiles.name(1:end-4) 'mat'], 'data','-v7.3' )
    %clear data BHV D E
end
disp('done')
