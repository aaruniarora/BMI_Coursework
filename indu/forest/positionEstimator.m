function [x, y, modelParameters] = positionEstimator(testData, modelParameters)
%% Initialization
trialDuration = size(testData.spikes, 2);
if modelParameters.trial_id == 0
    modelParameters.trial_id = testData.trialId;
else 
    if modelParameters.trial_id ~= testData.trialId
        modelParameters.iterations = 0;
        modelParameters.trial_id = testData.trialId;
    end
end
modelParameters.iterations = modelParameters.iterations + 1;
bin_size = modelParameters.bin_size;
num_bins = modelParameters.num_bins;
[X_new_processed] = preprocessing(testData, bin_size,num_bins);

% Ensure X_new_processed is the right size
expected_length = 98 * num_bins;
if length(X_new_processed) < expected_length
    X_new_processed = [X_new_processed, zeros(1, expected_length - length(X_new_processed))];
end

% 1. Use Random Forest to predict the angle
decoded_angle = zeros(1, num_bins);  % To store predicted angles

% For each time bin, predict the angle using the Random Forest model
for t = 1:num_bins
    % Extract features for the current time bin

    X_current = X_new_processed(:, (t - 1) * 98 + 1 : t * 98);

    % Predict the angle for this time bin
    decoded_angle(t) = predictRandomForest(modelParameters.rf_angle{t}, X_current);
end

% 2. Use Linear Regression to predict X and Y positions for each time bin
decoded_x = zeros(1, num_bins);
decoded_y = zeros(1, num_bins);

% Identify the predicted angle (use the most common angle across time bins)
predicted_angle = mode(decoded_angle);

% Find the corresponding angle index
[~, angle_idx] = min(abs(modelParameters.reach_angles - predicted_angle));

modelParameters.actualLabel = angle_idx;

% Use linear regression coefficients to predict X and Y positions based on predicted angle
for t = 1:num_bins
    % Extract features for the current time bin
    X_current = X_new_processed(:, (t - 1) * 98 + 1 : t * 98);

    % Predict X and Y positions using the corresponding regression models
    decoded_x(t) = predictLinearRegression(modelParameters.regression_x{angle_idx, t}, X_current);
    decoded_y(t) = predictLinearRegression(modelParameters.regression_y{angle_idx, t}, X_current);
end

% The final predicted X and Y positions (last time bin)
x = decoded_x(end);  % The final predicted X position
y = decoded_y(end);  % The final predicted Y position

%% Function to predict using Random Forest
function y_pred = predictRandomForest(forest, X_new)
    % Predict using the Random Forest (average tree predictions)
    num_trees = length(forest);
    predictions = zeros(num_trees, size(X_new, 1));

    for i = 1:num_trees
        predictions(i, :) = predictDecisionTree(forest(i).tree, X_new);
    end

    y_pred = mean(predictions, 1); % Average predictions
end

%% Function to predict using the manually trained decision tree
function y_pred = predictDecisionTree(tree, X)
    % Predict using the manually trained decision tree
    y_pred = zeros(size(X, 1), 1);

    for i = 1:size(X, 1)
        node = tree;
        while isfield(node, 'left') && isfield(node, 'right')
            if X(i, node.split_feature) <= node.split_value
                node = node.left;
            else
                node = node.right;
            end
        end
        y_pred(i) = node.value;
    end
end

%% Function to predict using Linear Regression (OLS)
function y_pred = predictLinearRegression(model, X_new)
    % Predict using the Linear Regression model (y = X * beta)
    y_pred = X_new * model.beta + model.intercept;  % Linear model: y = X * beta + intercept
end

%% Preprocess Data - padding for handpos and spikes
    function [X_data] = preprocessing(t_data, bin_size,num_b)
    start = 301;
    end_remove = 100;

    % Define parameters
    sigma = 20; % Standard deviation of Gaussian window (in ms)
    window = fspecial('gaussian', [1, 5*sigma], sigma); % Gaussian kernel

    % Extract spike train and corresponding hand positions
    spikes = t_data.spikes;   % 98 x T binary matrix

    len = length(spikes);

    % Convolve each neuron's spike train with Gaussian kernel
    smoothed_spikes = conv2(spikes, window, 'same'); % 98 x T_max

    % Downsample to 20ms bins
    % num_b = floor(len / bin_size);
    firing_rates = zeros(size(spikes, 1), num_b); % 98 x num_bins

    for bin = 1:num_b
        idx_start = (bin - 1) * bin_size + 1;
        idx_end = min(bin * bin_size, size(smoothed_spikes, 2));  % prevent overflow
        if idx_start <= idx_end
            firing_rates(:, bin) = mean(smoothed_spikes(:, idx_start:idx_end), 2);
        else
            firing_rates(:, bin) = zeros(size(spikes, 1), 1);  % pad with zeros if needed
        end
    end
    
    % Flatten firing rate matrix into a feature vector
    feature_vector = reshape(firing_rates, [], 1)'; % (98*num_bins) x 1

    % Store the data
    X_data = feature_vector; % Feature matrix
    X_data = X_data * 1000; % Scale firing rates
end

end
