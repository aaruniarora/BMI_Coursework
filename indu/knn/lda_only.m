clc
clear all

load('monkeydata_training.mat');
num = 2;
trainingData = trial(1:num, :);   % First 10 trials for training
testData = trial(num+1:end, :);  
trial = trainingData; % Remaining trials for testing

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

% for angle = 1:num_angles
%     for t = 1:num_trials
%         % Extract spike train and corresponding hand positions
%         spikes = trial(t, angle).spikes;   % 98 x T binary matrix
%         handPos = trial(t, angle).handPos; % 3 x T position matrix (X, Y, Z)
% 
%         % Truncate to match T_min
%         spikes = spikes(:, 1:T_min);
%         handPos = handPos(1:2, 1:T_min); % Only take X and Y
% 
%         % Flatten spike train into a feature vector
%         spike_vector = reshape(spikes, [], 1)'; % (98*T_min) x 1
% 
%         % Store the data
%         X_data = [X_data; spike_vector]; % Feature matrix
%         Y_data = [Y_data; reshape(handPos, 1, [])]; % Flatten trajectory
%     end
% end

% Define parameters
bin_size = 20; % 20 ms binning
sigma = 20; % Standard deviation of Gaussian window (in ms)
window = fspecial('gaussian', [1, 5*sigma], sigma); % Gaussian kernel

X_data = []; % Initialize feature matrix
Y_data = []; % Initialize labels (hand positions)

for angle = 1:num_angles
    for t = 1:num_trials
        % Extract spike train and corresponding hand positions
        spikes = trial(t, angle).spikes;   % 98 x T binary matrix
        handPos = trial(t, angle).handPos; % 3 x T position matrix (X, Y, Z)

        % Truncate to match T_min
        spikes = spikes(:, 301:T_min);
        handPos = handPos(1:2, 301:T_min); % Only take X and Y

        % Convolve each neuron's spike train with Gaussian kernel
        smoothed_spikes = conv2(spikes, window, 'same'); % 98 x T_min

        % Downsample to 20ms bins
        num_bins = floor((T_min - 300) / bin_size);
        firing_rates = zeros(size(spikes,1), num_bins); % 98 x num_bins

        for bin = 1:num_bins
            idx_start = (bin - 1) * bin_size + 1;
            idx_end = bin * bin_size;
            firing_rates(:, bin) = mean(smoothed_spikes(:, idx_start:idx_end), 2);
        end

        % Flatten firing rate matrix into a feature vector
        feature_vector = reshape(firing_rates, [], 1)'; % (98*num_bins) x 1

        % Store the data
        X_data = [X_data; feature_vector]; % Feature matrix
        Y_data = [Y_data; reshape(handPos, 1, [])]; % Flatten trajectory
    end
end
X_data = X_data*1000;

 %%
 % Manual LDA Implementation
% Define class labels (use angles as class labels)
angle_labels = repmat(1:num_angles, num_trials, 1);
angle_labels = angle_labels(:); % Flatten into a column vector

X_reduced = X_data;

num_features = size(X_data, 2);

% Separate data by class (angle)
X_class = cell(num_angles, 1);
for i = 1:num_angles
    X_class{i} = X_reduced(angle_labels == i, :);
end

% Compute class means and overall mean
mean_overall = mean(X_reduced, 1);
mean_class = zeros(num_angles, num_features);
for i = 1:num_angles
    mean_class(i, :) = mean(X_class{i}, 1);
end

% Compute the Between-Class Scatter Matrix (S_B)
S_B = zeros(num_features, num_features);
for i = 1:num_angles
    n_i = size(X_class{i}, 1);
    mean_diff = mean_class(i, :) - mean_overall;
    S_B = S_B + n_i * (mean_diff' * mean_diff);
end

% Compute the Within-Class Scatter Matrix (S_W)
S_W = zeros(num_features, num_features);
for i = 1:num_angles
    scatter_matrix = cov(X_class{i});
    S_W = S_W + scatter_matrix;
end

% Solve for the eigenvectors and eigenvalues of inv(S_W) * S_B
[W, D] = eig(inv(S_W) * S_B);

% Sort the eigenvectors by eigenvalues in descending order
[eigenvalues, idx] = sort(diag(D), 'descend');
W = W(:, idx);

% Project the data onto the first few eigenvectors (the most significant ones)
num_LDA_components = 2; % We choose 2 components for visualization
X_lda = X_reduced * W(:, 1:num_LDA_components);

%%
%% Classify using Nearest Mean Classifier
% Compute the mean of each class in the LDA space
class_means = zeros(num_angles, num_LDA_components);
for i = 1:num_angles
    class_means(i, :) = mean(X_lda(angle_labels == i, :), 1);
end

% Predict labels for each data point based on nearest class mean
predicted_labels = zeros(size(X_lda, 1), 1);
for i = 1:size(X_lda, 1)
    distances = vecnorm(X_lda(i, :) - class_means, 2, 2); % Euclidean distance
    [~, predicted_labels(i)] = min(distances);
end

% Visualize the classification in the LDA space
figure;
scatter(X_lda(:,1), X_lda(:,2), 50, predicted_labels, 'filled');
xlabel('LDA Component 1');
ylabel('LDA Component 2');
title('LDA Classification of PCA-Reduced Data');
colormap jet;
colorbar;
grid on;