%% Load Data
clear; clc; close all;
load('monkeydata_training.mat');

%% Preprocess Data
num_trials = size(trial,1);
num_angles = size(trial,2);
num_neurons = size(trial(1,1).spikes,1);

% Find minimum time length across all trials (for truncation)
T_min = inf; % Initialize with a large value
for angle = 1:num_angles
    for t = 1:num_trials
        T_min = min(T_min, size(trial(t, angle).spikes, 2));
    end
end

% Initialize matrices
X_data = [];  % Feature matrix
Y_labels = []; % Labels (movement direction)

for angle = 1:num_angles
    for t = 1:num_trials
        % Extract spike train and hand movement direction
        spikes = trial(t, angle).spikes;  % (98 x T) binary matrix
        spikes = spikes(:, 1:T_min);  % Truncate to match T_min
        
        % Flatten spike train into a feature vector
        spike_vector = reshape(spikes, [], 1)'; % (98*T_min) x 1

        % Store the data
        X_data = [X_data; spike_vector]; % Feature matrix
        Y_labels = [Y_labels; angle]; % Movement direction as class label
    end
end

%% Train LDA Classifier
lda_model = fitcdiscr(X_data, Y_labels); % Train LDA model

%% Evaluate Model Performance (Predict Directions)
Y_pred_labels = predict(lda_model, X_data);

% Compute accuracy
accuracy = sum(Y_pred_labels == Y_labels) / length(Y_labels) * 100;
fprintf('LDA Classification Accuracy: %.2f%%\n', accuracy);

%% Predict Hand Trajectories Using LDA
% Since LDA predicts movement directions, we estimate a trajectory template
% for each direction and use it to reconstruct predicted movements.

% Compute average trajectory for each direction
mean_trajectories = zeros(2, T_min, num_angles); % Store avg (X,Y) for each direction

for angle = 1:num_angles
    traj_sum = zeros(2, T_min);
    count = 0;
    
    for t = 1:num_trials
        if Y_labels(t) == angle  % Check if the trial belongs to this class
            handPos = trial(t, angle).handPos(1:2, 1:T_min); % Extract (X, Y)
            traj_sum = traj_sum + handPos;
            count = count + 1;
        end
    end
    
    % Compute average trajectory for this direction
    if count > 0
        mean_trajectories(:,:,angle) = traj_sum / count;
    end
end

%% Reconstruct Predicted Trajectories
Y_pred_trajectory = zeros(2, T_min, length(Y_pred_labels));

for i = 1:length(Y_pred_labels)
    predicted_angle = Y_pred_labels(i);
    Y_pred_trajectory(:,:,i) = mean_trajectories(:,:,predicted_angle);
end

%% Plot Full Trajectories
figure; hold on; grid on;
title('LDA Prediction Results');
xlabel('x direction'); ylabel('y direction');

for trial_idx = 1:size(Y_pred_trajectory, 3)
    % Get actual trajectory
    actual_angle = Y_labels(trial_idx);
    actual_traj = mean_trajectories(:,:,actual_angle);
    
    % Plot actual trajectory in blue
    plot(actual_traj(1,:), actual_traj(2,:), 'b', 'LineWidth', 1);
    
    % Plot predicted trajectory in red
    plot(Y_pred_trajectory(1,:,trial_idx), Y_pred_trajectory(2,:,trial_idx), 'r', 'LineWidth', 1);
end

legend({'Actual Position', 'Predicted Position'});
