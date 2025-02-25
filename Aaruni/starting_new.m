clc; clear; close all;

%% Choose between 'rastor', 'psth pad', 'psth trunc', 'tuning'
load('monkeydata_training.mat');
method = 'tuning'; 
[numTrials,numAngles] = size(trial); % J is rows, I is columns
angle = 1;
neuronIndex = 10;
numNeurons = size(trial(1,1).spikes, 1);

%% Rastor plot
if strcmp(method, 'rastor')
    fields = fieldnames(trial);    
    
    for j = 1:numAngles
        figure(j);
        uniqueTrials = randperm(numTrials, 16); % Generate 16 unique random numbers from 1 to 100
        for i = 1:16
            subplot(4,4,i)
            rn = uniqueTrials(i); % Get unique random trial index
    
            % Select a specific trial (e.g., trial 1, reaching angle 1)
            trialData = trial(rn,j);
            spikes = trialData.spikes;  % Dimensions: [numNeurons x numTimeBins]
            [numNeurons, numTimeBins] = size(spikes);
            hold on;
            % Loop through each neuron to plot its spike times
            for neuron = 1:numNeurons
                spikeTimes = find(spikes(neuron, :));  % Find time bins with a spike
                plot(spikeTimes, neuron * ones(size(spikeTimes)), 'k.', 'MarkerSize', 5);
            end
            xlabel('Time (ms)'); ylabel('Neuron Index');
            title(sprintf('Trial %d', rn));
            hold off;
        end
        set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
        sgtitle(sprintf('Population Raster Plot for Reaching Angle %d', j));
    end
end

%% Peri stimulus time histograms (truncate data)
if strcmp(method, 'psth trunc')    
    % Find the minimum length of any trial for this angle
    minTimeBins = inf;
    for trIdx = 1:numTrials
        currTimeBins = size(trial(trIdx, angle).spikes, 2);
        if currTimeBins < minTimeBins
            minTimeBins = currTimeBins;
        end
    end
    
    psth = zeros(1, minTimeBins);
    
    for trIdx = 1:numTrials
        spikes = trial(trIdx, angle).spikes(neuronIndex, 1:minTimeBins);
        psth = psth + spikes;
    end
    
    psth = psth / numTrials;
    firingRate = psth * 1000;
    
    % Smooth the PSTH (optional)
    windowSize = 30;
    gaussFilter = fspecial('gaussian', [1 windowSize], 5);
    smoothedFiringRate = conv(firingRate, gaussFilter, 'same');
    
    figure (2);
    plot(firingRate, 'LineWidth', 1.2, DisplayName='Original'); hold on;
    plot(smoothedFiringRate, 'LineWidth', 2, DisplayName='Gaussian Smoothened'); 
    xlabel('Time (ms)');
    ylabel('Firing Rate (Hz)');
    title(sprintf('PSTH (truncated) for Neuron %d, Angle %d', neuronIndex, angle));
    legend show;
end

%% Peri stimulus time histograms (padded data)
if strcmp(method, 'psth pad') 
    % Find the maximum length of any trial for this angle
    maxTimeBins = 0;
    for trIdx = 1:numTrials
        currTimeBins = size(trial(trIdx, angle).spikes, 2);
        if currTimeBins > maxTimeBins
            maxTimeBins = currTimeBins;
        end
    end
    
    % Initialize PSTH
    psth = zeros(1, maxTimeBins);
    
    % Sum spike counts for the selected neuron across all trials
    for trIdx = 1:numTrials
        spikes = trial(trIdx, angle).spikes(neuronIndex, :);
        % If this trial is shorter, pad with zeros
        currTimeBins = length(spikes);
        if currTimeBins < maxTimeBins
            spikes = [spikes, zeros(1, maxTimeBins - currTimeBins)];
        end
        psth = psth + spikes;
    end
    
    % Average over trials => spikes per ms
    psth = psth / numTrials;
    
    % Convert to firing rate in Hz (optional)
    firingRate = psth * 1000;
    
    % Smooth the PSTH (optional)
    windowSize = 30; 
    gaussFilter = fspecial('gaussian', [1 windowSize], 5);
    smoothedFiringRate = conv(firingRate, gaussFilter, 'same');
    
    % Plot
    figure (3);
    plot(firingRate, 'LineWidth', 1.2, DisplayName='Original'); hold on;
    plot(smoothedFiringRate, 'LineWidth', 2, DisplayName='Gaussian Smoothened'); 
    xlabel('Time (ms)');
    ylabel('Firing Rate (Hz)');
    title(sprintf('PSTH (padded) for Neuron %d, Angle %d', neuronIndex, angle));
    legend show;
end

%% Tuning Curve
if strcmp(method, 'tuning')
    % Preallocate a matrix to store average firing rates:
    % rows = neurons, columns = angles
    avgFiringRate = zeros(numNeurons, numAngles);
    
    % For each angle
    for ang = 1:numAngles
        % For each neuron
        for neuron = 1:numNeurons
            firingRates = [];  % store firing rates across trials
            
            % Collect firing rates across all trials for this angle
            for tr = 1:numTrials
                spikes = trial(tr, ang).spikes(neuron, :);
                duration = size(spikes, 2);  % number of time bins
                
                % Compute average firing rate for this trial in spikes/ms
                % Multiply by 1000 to convert to spikes/s (Hz)
                trialFiringRate = (sum(spikes) / duration) * 1000;
                
                firingRates = [firingRates, trialFiringRate];
            end
            
            % Average firing rate across all trials for this angle
            avgFiringRate(neuron, ang) = mean(firingRates);
        end
    end
    
    %% Now, pick a neuron to plot
    for no = 1:4
        figure(no);
        % uniqueTrials = randperm(numTrials, 25); % Generate 16 unique random numbers from 1 to 100
        uniqueTrials = (no-1)*25 + 1 : no*25;
        for ni = 1:length(uniqueTrials)
            subplot(5,5,ni)
            plot(1:numAngles, avgFiringRate(uniqueTrials(ni), :), '-o', 'LineWidth', 2);
            xlabel('Reaching Angle (1 to 8)');
            ylabel('Firing Rate (Hz)');
            title(sprintf('Tuning Curve for Neuron %d', uniqueTrials(ni)));
        end
    end
end