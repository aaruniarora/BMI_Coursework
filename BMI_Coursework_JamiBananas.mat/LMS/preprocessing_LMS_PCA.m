%% Load the Data
clear; clc; close all;
addpath('..\'); load('monkeydata_training.mat');

%% Preprocess Data
num_trials = size(trial,1);
num_angles = size(trial,2);
num_neurons = size(trial(1,1).spikes,1);

% Find the minimum time length across all trials (for truncation)
T_min = inf; % Initialize with a large value
for angle = 1:num_angles
    for t = 1:num_trials
        T_min = min(T_min, size(trial(t, angle).spikes, 2));
    end
end

% Initialize matrices for spike trains and hand trajectories
X_data = [];
Y_data = [];

for angle = 1:num_angles
    for t = 1:num_trials
        % Extract spike train and corresponding hand positions
        spikes = trial(t, angle).spikes;   % 98 x T binary matrix
        handPos = trial(t, angle).handPos; % 3 x T position matrix (X, Y, Z)

        % Truncate to match T_min
        spikes = spikes(:, 1:T_min);
        handPos = handPos(1:2, 1:T_min); % Only take X and Y

        % Flatten spike train into a feature vector
        spike_vector = reshape(spikes, [], 1)'; % (98*T_min) x 1

        % Store the data
        X_data = [X_data; spike_vector]; % Feature matrix
        Y_data = [Y_data; reshape(handPos, 1, [])]; % Flatten trajectory
    end
end

%% Apply PCA Using Singular Value Decomposition (SVD)
% Normalize data before SVD
X_mean = mean(X_data, 1);
X_norm = X_data - X_mean; % Center data (zero mean)

% Compute SVD
[U, S, V] = svd(X_norm, 'econ'); % 'econ' mode for efficiency

% Compute variance explained
singular_values = diag(S);
explained_variance = (singular_values.^2) / sum(singular_values.^2);
cum_variance = cumsum(explained_variance);

% Choose number of principal components to retain 95% variance
variance_threshold = 0.95;
num_PC = find(cum_variance >= variance_threshold, 1);

% Reduce data dimensionality
X_reduced = X_norm * V(:,1:num_PC); % Project onto principal components

%% Least Mean Squares (LMS) for Regression
% LMS parameters
mu = 0.001; % Learning rate
num_epochs = 1000; % Iterations

% Initialize weights
[~, num_features] = size(X_reduced);
W = zeros(num_features, size(Y_data, 2)); % Weights for full trajectory

% Train LMS using Gradient Descent
for epoch = 1:num_epochs
    for i = 1:size(X_reduced,1)
        % Prediction
        Y_pred = X_reduced(i,:) * W;

        % Error
        error = Y_data(i,:) - Y_pred;

        % Update weights
        W = W + mu * (X_reduced(i,:)' * error);
    end
end

%% Evaluate Model Performance
% Predict full hand trajectories using trained model
Y_pred = X_reduced * W;

% Reshape predictions back to (X, Y) trajectory format
Y_data_reshaped = reshape(Y_data', 2, T_min, []);
Y_pred_reshaped = reshape(Y_pred', 2, T_min, []);

%% Plot Full Trajectories
figure; hold on; grid on;
title('LMS Prediction Results with PCA');
xlabel('x direction'); ylabel('y direction');

for trial_idx = 1:size(Y_data_reshaped, 3)
    % Plot actual trajectory in blue
    plot(Y_data_reshaped(1,:,trial_idx), Y_data_reshaped(2,:,trial_idx), 'b', 'LineWidth', 1);
    
    % Plot predicted trajectory in red
    plot(Y_pred_reshaped(1,:,trial_idx), Y_pred_reshaped(2,:,trial_idx), 'r', 'LineWidth', 1);
end

legend({'Actual Position', 'Decoded Position'});

%%
% Compute RMSE per time step
rmse_X = sqrt(mean((Y_pred_reshaped(1,:,:) - Y_data_reshaped(1,:,:)).^2, 'all'));
rmse_Y = sqrt(mean((Y_pred_reshaped(2,:,:) - Y_data_reshaped(2,:,:)).^2, 'all'));
total_rmse = mean([rmse_X, rmse_Y]);

% Display RMSE
fprintf('RMSE for X: %.4f cm\n', rmse_X);
fprintf('RMSE for Y: %.4f cm\n', rmse_Y);
fprintf('Total RMSE: %.4f cm\n', total_rmse);