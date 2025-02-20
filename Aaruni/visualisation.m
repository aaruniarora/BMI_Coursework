clc; clear; close all;

%% Load dataset

load('monkeydata_training.mat');

reaching_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23,18, 31/18, 35/18] .* pi;

try11 = trial(1,1);

spikes = try11.spikes;
handPos = try11.handPos;

t_ax = 0:1:length(spikes)-1; % in ms

figure(1); 
for i = 1:size(spikes)
    plot(t_ax, spikes(i,:)); hold on;
end
hold off;

%% Histogram for time
spike_counts = sum(spikes, 1);  % 1×672 vector
figure(2);
bar(spike_counts);
xlabel('Time Bins');
ylabel('Spike Count');
title('Spike Frequency Over Time');
grid on;

%% Heat map
figure(3);
imagesc(spikes);  % Display matrix as a color-coded heatmap
colorbar;  % Show color scale
xlabel('Time (ms)');
ylabel('Neuron #');
title('Heatmap of Neuronal Activity');

% Adjust axis
set(gca, 'YDir', 'normal');  % Make sure neurons are plotted in the correct order

%% Histogram for 

