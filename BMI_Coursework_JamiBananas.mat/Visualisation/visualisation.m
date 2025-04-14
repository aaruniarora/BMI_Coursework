clc; close all; clear;

%% Load dataset
addpath('..\')
load('monkeydata_training.mat');
numTrials = length(trial);

reaching_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23,18, 31/18, 35/18] .* pi;

try11 = trial(1,1);

spikes = try11.spikes;
handPos = try11.handPos;

t_ax = 0:1:length(spikes)-1; % in ms

% figure(1); 
% for i = 1:size(spikes)
%     plot(t_ax, spikes(i,:)); hold on;
% end
% hold off;

%% Histogram for time
spike_counts = sum(spikes, 1);  % 1×672 vector

% figure(2);
% bar(spike_counts);
% xlabel('Time Bins');
% ylabel('Spike Count');
% title('Spike Frequency Over Time');
% grid on;

%% Heat map
figure(3);
imagesc(spikes);  % Display matrix as a color-coded heatmap
colormap gray;colorbar;  % Show color scale
xlabel('Time (ms)');
ylabel('Neuron #');
title('Heatmap of Neuronal Activity');

% Adjust axis
set(gca, 'YDir', 'normal');  % Make sure neurons are plotted in the correct order

%% Rastor Plot
numTrials   = length(trial);
numNeurons  = size(trial(1).spikes, 1);
numTimeBins = size(trial(1).spikes, 2);

% figure(4); hold on;
% for trialIdx = 1:numTrials
%     spikes = trial(trialIdx).spikes; % Get spikes for current trial
%     offset = (trialIdx - 1) * numNeurons; % Compute offset so that each trial's neurons appear in its own vertical block
%     for neuronIdx = 1:numNeurons
%         spikeTimes = find(spikes(neuronIdx, :) == 1);
%         % Plot a short vertical line for each spike
%         for s = 1:length(spikeTimes)
%             line([spikeTimes(s) spikeTimes(s)], [neuronIdx - 0.4 + offset, neuronIdx + 0.4 + offset], 'Color', 'k');
%         end
%     end
% end
% 
% xlabel('Time (ms)');
% ylabel('Neuron (stacked by trial)');
% title('Raster Plot (Vertical Lines) for 100 Trials');
% xlim([0, numTimeBins]);
% ylim([0.5, numTrials*numNeurons + 0.5]);
% set(gca, 'YDir', 'normal');  % Ensure lower indices appear at the bottom
% hold off;

%% 1. Plot Hand Positions Across Trials
figure (6);
hold on;
% Optionally, use a different color for each trial
colors = lines(numTrials);
for i = 1:numTrials
    pos = trial(i).handPos;  % Assume handPos is 2 x T [X; Y]
    % Plot hand trajectory for trial i
    plot(pos(1,:), pos(2,:), 'Color', colors(i,:), 'LineWidth', 1.5);
end
xlabel('X Position');
ylabel('Y Position');
title('Hand Positions Across Trials');
grid on;
hold off;

%% 2. Compute and Plot Tuning Curves for Movement Direction
% Determine the movement direction for each trial.
% If a 'target' field exists, use it. Otherwise, compute from hand positions.
figure (7); %tiledlayout(numNeurons/2,2)
for hello=1
    allTargets = zeros(numTrials,1);
    % Plot tuning curves for a subset of neurons (e.g., neurons 1 to 5)
    % subplot(4,2,hello);
    for i = 1:numTrials
        % Compute direction as the angle from start to end of the hand trajectory.
        pos = trial(i,hello).handPos;
        allTargets(i) = atan2(pos(2,end)-pos(2,1), pos(1,end)-pos(1,1));
    end
    
    uniqueTargets = unique(allTargets);
    numTargets = length(uniqueTargets);
    
    % For each neuron, compute the average (time-averaged) firing rate
    % and its standard deviation across trials for each target direction.
    numNeurons = size(trial(hello).spikes, 1);
    meanFR = zeros(numNeurons, numTargets);
    stdFR  = zeros(numNeurons, numTargets);
    tol = 1e-3;  % Tolerance for comparing angles
    
    for n = 1:numNeurons
        for t = 1:numTargets
            % Find indices of trials with this target direction (within tolerance)
            trialIdx = find(abs(allTargets - uniqueTargets(t)) < tol);
            rates = zeros(length(trialIdx),1);
            for j = 1:length(trialIdx)
                spk = trial(trialIdx(j)).spikes;
                % Compute firing rate as total spikes divided by the number of time bins.
                rates(j) = sum(spk(n,:)) / size(spk,2);
            end
            meanFR(n,t) = mean(rates);
            stdFR(n,t)  = std(rates);
        end
    end

    % count = 1;
    for n = 1:98
        errorbar(uniqueTargets, meanFR(n,:), stdFR(n,:), 'o-', LineWidth=1, MarkerSize=3, DisplayName=['Neuron ' num2str(n)]); 
        hold on;
        %count = 1 + count;
    end
    xlabel('Movement Direction (rad)');
    ylabel('Firing Rate (spikes/ms)');
    title(['Reaching Angle ', num2str(hello)])%title(['Neuron ' num2str(n)]);
    grid on; legend show; hold off;
end
sgtitle('Tuning Curves for Selected Neurons');

% %% 3. Population Vector Algorithm to Predict Arm Movements
% % First, estimate each neuron’s preferred direction (PD) from its tuning curve.
% % Here we use the vector-sum method.
% preferredDirs = zeros(numNeurons, 1);
% for n = 1:numNeurons
%     x_comp = sum(meanFR(n,:) .* cos(uniqueTargets)');
%     y_comp = sum(meanFR(n,:) .* sin(uniqueTargets)');
%     preferredDirs(n) = atan2(y_comp, x_comp);
% end
% 
% % Now predict the movement direction for each trial using the population vector.
% predictedDirs = zeros(numTrials,1);
% for i = 1:numTrials
%     spk = trial(i).spikes;
%     % Compute the time-averaged firing rate for each neuron in this trial.
%     FR_trial = sum(spk, 2) / size(spk,2);
% 
%     % Compute the population vector (weighted sum of preferred directions)
%     x_sum = sum(FR_trial .* cos(preferredDirs));
%     y_sum = sum(FR_trial .* sin(preferredDirs));
%     predictedDirs(i) = atan2(y_sum, x_sum);
% end
% 
% % Plot predicted versus actual movement directions.
% figure (8);
% plot(allTargets, predictedDirs, 'ko', 'MarkerFaceColor','b');
% xlabel('Actual Movement Direction (rad)');
% ylabel('Predicted Movement Direction (rad)');
% title('Population Vector Prediction');
% grid on;
% hold on;
% % Plot a reference line (perfect prediction)
% plot([-pi, pi], [-pi, pi], 'r--', 'LineWidth', 2);
% hold off;
