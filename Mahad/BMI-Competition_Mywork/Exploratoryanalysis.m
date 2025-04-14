%% EXPLORATORY_DATA_ANALYSIS.M
% This script assumes that 'monkeydata_training.mat' is in the current folder
% or on the MATLAB path. It loads the data into a variable named 'trial',
% which has size (100 x 8) in the default dataset.
%
% Each element of trial(n,k) is a struct with fields:
%   - trialId (scalar)
%   - spikes (98 x T matrix: 98 neurons, T time bins in ms)
%   - handPos (3 x T matrix: x, y, z positions in mm, T time bins in ms)
%
% The script demonstrates:
%   1) Population raster plot (single trial)
%   2) Single-neuron raster plot across multiple trials
%   3) Peri-stimulus time histogram (PSTH) for one neuron
%   4) Hand trajectory plots across multiple trials

clear; clc; close all;

%% 0) LOAD THE DATA
% Ensure the data file is in the current folder or on your MATLAB path.
dataFile = 'monkeydata_training.mat';
if ~exist(dataFile, 'file')
    error('File %s not found in the current directory or MATLAB path.', dataFile);
end
load(dataFile, 'trial');  % Loads the variable 'trial' into the workspace

% Quick checks on the structure dimensions:
[nTrials, nAngles] = size(trial);
fprintf('Data loaded. trial is %dx%d.\n', nTrials, nAngles);

% Each trial(n,k).spikes: 98 x T
% Each trial(n,k).handPos: 3 x T

%% 1) POPULATION RASTER PLOT FOR A SINGLE TRIAL
% Pick trial(1,1) (i.e., first trial, first angle) as an example.
trialIdx = 1;
angleIdx = 1;

spikesThisTrial = trial(trialIdx, angleIdx).spikes;  % [98 x T]
[numNeurons, T] = size(spikesThisTrial);

figure('Name','(1) Population Raster - Single Trial','NumberTitle','off');
hold on;
for neuronIdx = 1:numNeurons
    % Find time bins where this neuron fired
    spikeTimes = find(spikesThisTrial(neuronIdx,:) == 1);
    % Plot those times on the x-axis, with neuronIdx on the y-axis
    plot(spikeTimes, neuronIdx*ones(size(spikeTimes)), 'k.');
end
xlabel('Time (ms)');
ylabel('Neuron Index (1 to 98)');
title(sprintf('Population Raster: trial(%d,%d)', trialIdx, angleIdx));
axis tight;

%% 2) RASTER FOR ONE NEURON ACROSS MULTIPLE TRIALS
% We'll pick neuron #10, angle #1, and plot it over all trials (1..nTrials).
neuronID = 10;
angleID  = 1;

figure('Name','(2) Single Neuron Across Multiple Trials','NumberTitle','off');
hold on;
cmap = lines(nTrials);  % distinct colors for each trial
for tIdx = 1:nTrials
    spikesThisNeuron = trial(tIdx, angleID).spikes(neuronID,:);  % 1 x T
    spikeTimes = find(spikesThisNeuron == 1);
    plot(spikeTimes, tIdx*ones(size(spikeTimes)), '.', 'Color', cmap(tIdx,:));
end
xlabel('Time (ms)');
ylabel('Trial Number');
title(sprintf('Raster: Neuron #%d across %d Trials (Angle %d)', neuronID, nTrials, angleID));
axis tight;

%% 3) PERI-STIMULUS TIME HISTOGRAM (PSTH) FOR ONE NEURON
% We'll compute the average firing rate over trials at a given angle.
% Because trial lengths can differ, we find a minimum T so that we can
% average up to that point across all trials.
neuronID = 10;
angleID  = 1;

% Collect the minimum trial length for angleID
trialLengths = zeros(nTrials,1);
for tIdx = 1:nTrials
    trialLengths(tIdx) = size(trial(tIdx, angleID).spikes,2);
end
Tmin = min(trialLengths);

% Accumulate spike counts across trials (only up to Tmin for uniformity)
spikeSum = zeros(1, Tmin);
for tIdx = 1:nTrials
    spikeSum = spikeSum + trial(tIdx, angleID).spikes(neuronID, 1:Tmin);
end

% PSTH = average spikes per bin, i.e. firing rate in spikes/ms
PSTH = spikeSum / nTrials;

% Optional smoothing (simple moving average)
windowSize = 50;  % 50 ms window for smoothing
smoothKernel = ones(1, windowSize) / windowSize;
smoothedPSTH = conv(PSTH, smoothKernel, 'same');

figure('Name','(3) PSTH','NumberTitle','off');
plot(1:Tmin, smoothedPSTH, 'LineWidth', 1.5, 'Color', 'b');
xlabel('Time (ms)');
ylabel('Firing Rate (spikes/ms)');
title(sprintf('Smoothed PSTH for Neuron #%d (Angle %d)', neuronID, angleID));
axis tight;

%% 4) PLOT HAND POSITIONS ACROSS MULTIPLE TRIALS (X vs. Y)
% Let's look at all trials for angle #1, from start to end of each trial.
angleID = 3;

figure('Name','(4) Hand Trajectories','NumberTitle','off');
hold on;
for tIdx = 1:nTrials
    handPos = trial(tIdx, angleID).handPos;  % 3 x T
    plot(handPos(1,:), handPos(2,:));
end
xlabel('X-position (mm)');
ylabel('Y-position (mm)');
title(sprintf('Hand Trajectories - Angle %d', angleID));
axis equal;  % So 1 mm in X is same scale as 1 mm in Y
axis tight;

