clc 
close all

%%
trialElement = trial(2,2);  % Extract one element (1st trial, 1st angle)
disp(trialElement);         % Display structure fields

%%
figure;
% Extract spike data for the specific trial and angle, and slice time steps from 300 to 572
timeSteps = 301:length(spikeCount)-100
spikeData = trial(2, 2).spikes(:, timeSteps); 

% Plot spike activity for the selected time range
imagesc(spikeData);              % Plot spike activity for time steps 300 to 572
colormap(gray);                  % Use grayscale colormap
xlabel('Time (ms)');
ylabel('Neural Channels');
title(['Raster Plot of Spikes - Trial ', num2str(trial(2,2).trialId)]);
colorbar; % Show intensity scale
%%
figure;
meanFiringRate = mean(trial(2,2).spikes, 1);  % Mean across 98 channels
plot(meanFiringRate, 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Mean Firing Rate');
title(['Mean Firing Rate Over Time - Trial ', num2str(trial(2,2).trialId)]);
grid on;

%%
figure;
plot(trial(2,2).handPos(1,:), trial(2,2).handPos(2,:), 'LineWidth', 2);
xlabel('X Position');
ylabel('Y Position');
title(['Hand Trajectory - Trial ', num2str(trial(2,2).trialId)]);
grid on;

%%
figure;
plot3(trial(2,2).handPos(1,:), trial(2,2).handPos(2,:), trial(2,2).handPos(3,:), 'LineWidth', 2);
xlabel('X Position');
ylabel('Y Position');
zlabel('Z Position');
title(['3D Hand Trajectory - Trial ', num2str(trial(2,2).trialId)]);
grid on;

%%
% Select a specific trial and angle
trialIdx = 1; % Change this to choose a different trial (1 to 100)
angleIdx = 1; % Change this to choose a different angle (1 to 8)

% Select a neuron randomly
neuronIdx = randi(size(trial(trialIdx, angleIdx).spikes, 1)); 

% Extract the spike activity for the chosen neuron
spikeTrain = trial(trialIdx, angleIdx).spikes(neuronIdx, :); 

% Plot the spike activity over 672 time points
plot(spikeTrain, 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Spike Activity');
title(['Spike Activity of Neuron ', num2str(neuronIdx), ...
       ' - Trial ', num2str(trial(trialIdx, angleIdx).trialId), ...
       ' - Angle ', num2str(angleIdx)]);
grid on;

%%
% Select a specific trial and angle
trialIdx = 1; % Change this to choose a different trial (1 to 100)
angleIdx = 1; % Change this to choose a different angle (1 to 8)

% Extract spike data for all 98 neurons in the chosen trial and angle
spikeData = trial(trialIdx, angleIdx).spikes;

% Count the number of spikes (1s) at each time point across all neurons
spikeCount = sum(spikeData == 1, 1); % Sum across neurons (rows) for each time step

% Plot the histogram of spike counts per time step
figure;
bar(spikeCount, 'FaceColor', 'b');
xlabel('Time Step');
ylabel('Number of Neurons Spiking');
title(['Spike Count at Each Time Step - Trial ', num2str(trial(trialIdx, angleIdx).trialId), ...
       ' - Angle ', num2str(angleIdx)]);
grid on;

%%
% Select a specific trial and angle
trialIdx = 1; % Change this to choose a different trial (1 to 100)
angleIdx = 1; % Change this to choose a different angle (1 to 8)

% Extract spike data for all 98 neurons in the chosen trial and angle
spikeData = trial(trialIdx, angleIdx).spikes;

% Count the number of spikes (1s) at each time point across all neurons
spikeCount = sum(spikeData == 1, 1); % Sum across neurons (rows) for each time step

% Remove the first 300 and last 100 time steps from the graph (not data)
timeSteps = 301:length(spikeCount)-100; % Time steps to display

% Plot the histogram of spike counts per time step
figure;
bar(timeSteps, spikeCount(timeSteps), 'FaceColor', 'b');
xlabel('Time Step');
ylabel('Number of Neurons Spiking');
title(['Spike Count at Each Time Step - Trial ', num2str(trial(trialIdx, angleIdx).trialId), ...
       ' - Angle ', num2str(angleIdx)]);
grid on;


