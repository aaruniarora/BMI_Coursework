function model_params = positionEstimatorTraining(training_data)
%% Initialization
clc
model_params = struct;
num_trials = size(training_data, 1);
num_angles = size(training_data, 2);
model_params.trial_id = 0;
model_params.iterations = 0;
model_params.reach_angles = [0, 45, 90, 135, 180, 225, 270, 315]; 

bin_size = 20;  % Size of the time bin in ms

% Preprocessing: get firing rate features and positions
[X_data, Y_data, num_bins] = preprocessing(training_data, num_trials, num_angles, bin_size);
model_params.bin_size = bin_size;
model_params.num_bins = num_bins;
model_params.num_angles = num_angles;
model_params.firing_data = X_data;

% Train Random Forest models for angle prediction
num_trees = 50;
for t = 1:num_bins
    % Prepare data for Random Forest
    X_bin = X_data(:, (t-1)*98+1:t*98);  % Extract features for current time bin
    Y_angle = Y_data(:, (t-1)*2 + 1 : t*2);  % Extract angle labels (X, Y positions)
    % Train Random Forest for angle prediction
    model_params.rf_angle{t} = trainRandomForest(X_bin, Y_angle, num_trees);
end

% Train Linear Regression models for X and Y position prediction
for angle = 1:num_angles
    for t = 1:num_bins

        angle_start = (angle - 1) * num_trials + 1;   % Start index for the current angle
        angle_end = angle * num_trials;                % End index for the current angle
        
        % Slice the X_data and Y_data for the current angle
        X_bin = X_data(angle_start:angle_end, (t-1)*98+1:t*98);  % Firing rates for the current angle and time bin
        Y_x = Y_data(angle_start:angle_end, t*2-1);               % X positions for the current angle and time bin
        Y_y = Y_data(angle_start:angle_end, t*2);                 % Y positions for the current angle and time bin
        assignin('base',"x",X_bin)
        assignin('base',"y_x",Y_x)
        % Train Linear Regression for X and Y position prediction
        model_params.regression_x{angle, t} = trainLinearRegression(X_bin, Y_x);
        model_params.regression_y{angle, t} = trainLinearRegression(X_bin, Y_y);
        
    end
end

%% Function to train Random Forest
function forest = trainRandomForest(X, Y, num_trees)
    forest = struct;
    max_depth = 10;
    min_samples_split = 2;
    min_samples_leaf = 1;

    for i = 1:num_trees
        idx = randi(size(X, 1), size(X, 1), 1);  % Bootstrap
        X_sample = X(idx, :);
        Y_sample = Y(idx, :);
        forest(i).tree = trainDecisionTree(X_sample, Y_sample, 0, max_depth, min_samples_split, min_samples_leaf);
    end
end

%% Function to train Decision Tree
function tree = trainDecisionTree(X, Y, depth, max_depth, min_samples_split, min_samples_leaf)
    tree = struct;
    if depth >= max_depth || size(X, 1) < min_samples_split || numel(unique(Y)) == 1
        tree.value = mean(Y(:));
        return;
    end

    tree.split_feature = randi(size(X, 2));
    tree.split_value = median(X(:, tree.split_feature));

    left_idx = X(:, tree.split_feature) <= tree.split_value;
    right_idx = ~left_idx;

    if sum(left_idx) >= min_samples_leaf && sum(right_idx) >= min_samples_leaf
        tree.left = trainDecisionTree(X(left_idx, :), Y(left_idx), depth + 1, max_depth, min_samples_split, min_samples_leaf);
        tree.right = trainDecisionTree(X(right_idx, :), Y(right_idx), depth + 1, max_depth, min_samples_split, min_samples_leaf);
    else
        tree.value = mean(Y(:));
    end
end

%% Function to train Linear Regression (OLS)
function model = trainLinearRegression(X, Y)
    % Add a bias column (ones) for the intercept term
    X_augmented = [ones(size(X, 1), 1), X];
    X_augmented = (X_augmented - mean(X_augmented)) ./ std(X_augmented);
    
    % Linear regression: Y = X * beta + intercept
    beta = (X_augmented' * X_augmented) \ (X_augmented' * Y); % OLS solution
    model.beta = beta(2:end);  % Coefficients
    model.intercept = beta(1);  % Intercept
end

end


%%

function [X_data, Y_data, num_bins] = preprocessing(t_data, num_trials, num_angles, bin_size)
start = 301;
end_remove = 100;

% Find the maximum time length across all trials
T_max = -inf; % Initialize with a small value
for angle = 1:num_angles
    for t = 1:num_trials
        T_max = max(T_max, size(t_data(t, angle).spikes, 2));
    end
end

T_max = T_max - end_remove - start;
T_max = ceil(T_max / 20) * 20;
% Downsample to 20ms bins
num_bins = floor(T_max / bin_size);

% Define parameters
sigma = 20; % Standard deviation of Gaussian window (in ms)
window = fspecial('gaussian', [1, 5*sigma], sigma); % Gaussian kernel

X_data = []; % Initialize feature matrix
Y_data = []; % Initialize combined Y matrix

for angle = 1:num_angles
    for t = 1:num_trials
        % Extract spike train and corresponding hand positions
        spikes = t_data(t, angle).spikes;   % 98 x T binary matrix
        handPos = t_data(t, angle).handPos; % 3 x T position matrix (X, Y, Z)

        % Truncate from 301:T_max
        spikes = spikes(:, start:end-end_remove);
        handPos = handPos(1:2, start:end-end_remove); % Only take X and Y

        % Determine the current length
        T_current = size(spikes, 2);

        % Pad spikes with zeros if shorter than T_max
        if T_current < T_max
            spikes = [spikes, zeros(size(spikes, 1), T_max - T_current)];
        end

        % Pad handPos with the last available position if shorter than T_max
        if T_current < T_max
            last_pos = handPos(:, end);
            handPos = [handPos, repmat(last_pos, 1, T_max - T_current)];
        end

        % Convolve each neuron's spike train with Gaussian kernel
        smoothed_spikes = conv2(spikes, window, 'same'); % 98 x T_max


        firing_rates = zeros(size(spikes, 1), num_bins); % 98 x num_bins

        for bin = 1:num_bins
            idx_start = (bin - 1) * bin_size + 1;
            idx_end = min(bin * bin_size, T_max);
            firing_rates(:, bin) = mean(smoothed_spikes(:, idx_start:idx_end), 2);
        end

        % Downsample X and Y hand positions separately
        Y_x_binned = zeros(1, num_bins); % 1 x num_bins (for X)
        Y_y_binned = zeros(1, num_bins); % 1 x num_bins (for Y)
        
        for bin = 1:num_bins
            idx_start = (bin - 1) * bin_size + 1;
            idx_end = min(bin * bin_size, T_max);
            Y_x_binned(:, bin) = mean(handPos(1, idx_start:idx_end), 2); % X coordinate
            Y_y_binned(:, bin) = mean(handPos(2, idx_start:idx_end), 2); % Y coordinate
        end

        % Flatten firing rate matrix into a feature vector
        feature_vector = reshape(firing_rates, [], 1)'; % (98*num_bins) x 1

        % Store the data
        X_data = [X_data; feature_vector]; % Feature matrix

        % Arrange X and Y alternately in Y_data
        Y_row = reshape([Y_x_binned; Y_y_binned], 1, []);  % 1 x (2*num_bins)
        Y_data = [Y_data; Y_row];

    end
end
X_data = X_data * 1000; % Scale firing rates
end
%%
function [score, coeff] = covPCA(X, numPC)
    mu = mean(X,1);
    Xc = X - mu;
    C = cov(Xc);
    [V, D] = eig(C);
    [~, idx] = sort(diag(D), 'descend');
    coeff = V(:, idx(1:numPC));
    score = Xc * coeff;
end