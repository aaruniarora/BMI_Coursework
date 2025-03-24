clc
clear all

load('monkeydata_training.mat');

%%

reach_angles = [1/6,7/18,11/18,15/18,19/18,23/18,31/18,35/18].*pi;

angle_no = length(reach_angles);

time_step = 1*10^-3;

trial1 = trial(1,1);

trials = 100;

% spikes = trial1.spikes;
% handpos = trial1.handPos;

before = 300;
after =100;

% spikes = spikes(:, 301:end-100);
% 
% shape = size(spikes);
% 
% neurons = shape(1);
% time_points = shape(2);
% 
% time = linspace (0,time_points,time_points);
%%
% figure;
% hold on
% for i = 1:neurons
% plot(time,spikes(i,:))
% end
% hold off

%%
% Sum across units (rows) to get total activity per time point

spike_counts = sum(spikes, 1);  % 1×672 vector

figure;
bar(spike_counts);
xlabel('Time Bins');
ylabel('Spike Count');
title('Spike Frequency Over Time');
grid on;

%%
clc
spikes_all = {};
for i = 1:trials
    for x = 1:angle_no
        array = trial(i,x).spikes;
        spikes_all{i,x} = array;
    end
end

handpos_all = {};
for i = 1:trials
    for x = 1:angle_no
        array = trial(i,x).handPos;
        handpos_all{i,x} = array;
    end
end

%% Define Time Window to Remove
sampling_rate = 1; % Adjust if necessary (assuming 1 ms bins)
remove_start = 300 / sampling_rate; % Convert ms to index
remove_end = 100 / sampling_rate; % Convert ms to index

%% Extract Number of Trials & Neurons
num_trials = size(trial, 1);
num_angles = size(trial, 2);
num_neurons = size(trial(1,1).spikes, 1); % Assuming all trials have same neuron count

% Determine the minimum number of time points after trimming (using the shortest trial)
min_time_points = min(cellfun(@(x) size(x, 2), {trial.spikes})) - 400;

%% 1) Population Raster Plot for All Trials & All Neurons (Following the example)
figure; hold on;
colors = lines(num_trials); % Different colors for different trials

for trial_idx = 1:num_trials
    for angle_idx = 1:num_angles
        neural_data = trial(trial_idx, angle_idx).spikes(:, remove_start+1:end-remove_end); % Remove time window
        [neurons, times] = find(neural_data); % Get spike times
        
        % Plot the spikes using small scatter points
        scatter(times, neurons + (trial_idx - 1) * num_neurons, 5, colors(trial_idx, :), 'filled');
    end
end

xlabel('Time (ms)'); ylabel('Neural Units');
title('Full Population Raster Plot (Trimmed)');
hold off;

%% 2) PSTH Over All Neurons and Trials (Adjusting for Variable-Length Data)
bin_size = 20; % Time bin size
psth_all = zeros(num_neurons, min_time_points-1); % Initialize PSTH array with the shortest length

for trial_idx = 1:num_trials
    for angle_idx = 1:num_angles
        % Get the length of the current trial after trimming
        spike_matrix = trial(trial_idx, angle_idx).spikes;
        num_time_points_current = size(spike_matrix, 2) - (remove_start + remove_end); % Adjust length after trimming
        
        % Ensure the current spike matrix is not shorter than the min_time_points
        if num_time_points_current > 0
            spikes_trimmed = spike_matrix(:, remove_start+1:end-remove_end); % Remove time window
            % Ensure each trial is trimmed to the minimum time points
            spikes_trimmed = spikes_trimmed(:, 1:min(num_time_points_current, min_time_points-1)); 

            % Sum the spikes
            psth_all = psth_all + spikes_trimmed; % Sum spikes
        end
    end
end

% Compute average firing rate
psth_avg = mean(psth_all, 1) / (num_trials * num_angles);
smooth_psth = movmean(psth_avg, bin_size); % Smoothed

% Plot PSTH
figure; hold on;
bar(psth_avg, 'FaceColor', [0.7 0.7 0.7]); % Raw PSTH
plot(smooth_psth, 'r', 'LineWidth', 2); % Smoothed PSTH
xlabel('Time (bins)'); ylabel('Firing Rate (spikes/bin)');
title('PSTH Over All Neurons and Trials (Trimmed)');
hold off;

%% 4) Plot Hand Positions for Different Trials (X vs Y) with Same Color
figure; hold on;

color = 'b'; % Choose a color (e.g., blue)

for trial_idx = 1:num_trials
    for angle_idx = 1:num_angles
        handPos = trial(trial_idx, angle_idx).handPos; % Hand positions for this trial and angle
        
        % Plot the x vs y hand positions (2D trajectory) with the same color
        plot(handPos(1, :), handPos(2, :), 'Color', color); % X vs Y positions
    end
end

xlabel('X Position');
ylabel('Y Position');
title('Hand Position for Different Trials (X vs Y) - Same Color');
hold off;

%% 5) Tuning Curves for Movement Direction
bin_size = 20; % Time bin size for averaging
num_neurons = size(trial(1, 1).spikes, 1); % Number of neurons
num_angles = length(reach_angles); % Number of angles

% Define min_time_points based on the shortest spike matrix length after trimming
min_time_points = min(cellfun(@(x) size(x, 2), {trial(1, :).spikes})) - remove_start - remove_end;

firing_rates = zeros(num_neurons, num_angles); % Store firing rates for all neurons and angles
firing_rate_std = zeros(num_neurons, num_angles); % Store standard deviation for error bars

for neuron_idx = 1:num_neurons
    for angle_idx = 1:num_angles
        spike_counts = zeros(num_trials, 1); % For storing spike counts for each trial
        
        for trial_idx = 1:num_trials
            % Extract the spike data for this trial, neuron, and angle
            spike_matrix = trial(trial_idx, angle_idx).spikes(neuron_idx, remove_start+1:end-remove_end);
            
            % Calculate the firing rate for this trial and this angle
            spike_counts(trial_idx) = sum(spike_matrix) / min_time_points; % Firing rate for this trial
            
        end
        
        % Compute the average firing rate and the standard deviation across trials
        firing_rates(neuron_idx, angle_idx) = mean(spike_counts);
        firing_rate_std(neuron_idx, angle_idx) = std(spike_counts);
    end
end

% Plot tuning curves for each neuron with error bars
figure; hold on;
for neuron_idx = 1:num_neurons
    errorbar(reach_angles, firing_rates(neuron_idx, :), firing_rate_std(neuron_idx, :), 'o-', ...
        'DisplayName', ['Neuron ' num2str(neuron_idx)]);
end

xlabel('Movement Direction (radians)');
ylabel('Firing Rate (spikes/s)');
title('Tuning Curves for Movement Direction');
legend;
hold off;

%% 7) Population Vector Algorithm to Predict Arm Movements
num_neurons = size(trial(1, 1).spikes, 1); % Number of neurons
num_angles = length(reach_angles); % Number of angles

% Initialize an array to store the predicted angles for each trial
predicted_angles = zeros(num_trials, num_angles);

% Loop through each trial
for trial_idx = 1:num_trials
    for angle_idx = 1:num_angles
        % Initialize the population vector (x and y components)
        population_vector_x = 0;
        population_vector_y = 0;
        
        % Loop over neurons to calculate the population vector
        for neuron_idx = 1:num_neurons
            spike_matrix = trial(trial_idx, angle_idx).spikes(neuron_idx, remove_start+1:end-remove_end);
            
            % Calculate the firing rate for this neuron
            firing_rate = sum(spike_matrix) / min_time_points; % Firing rate for this trial
            
            % Calculate the unit vector for this neuron based on the angle
            unit_vector_x = cos(reach_angles(angle_idx));
            unit_vector_y = sin(reach_angles(angle_idx));
            
            % Weight the unit vector by the neuron's firing rate
            population_vector_x = population_vector_x + firing_rate * unit_vector_x;
            population_vector_y = population_vector_y + firing_rate * unit_vector_y;
        end
        
        % Calculate the predicted movement direction (angle of the population vector)
        predicted_angle = atan2(population_vector_y, population_vector_x);
        
        % Store the predicted angle for this trial and angle
        predicted_angles(trial_idx, angle_idx) = predicted_angle;
    end
end

% Plot the predicted angles for all trials
figure; hold on;

% Plot all trials in the same figure with different colors
for trial_idx = 1:num_trials
    plot(reach_angles, predicted_angles(trial_idx, :), 'o-', 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0]);
end

% Add labels and title
xlabel('Movement Direction (radians)');
ylabel('Predicted Movement Direction (radians)');
title('Predicted Movement Directions Using Population Vector');
hold off;

