
% clc
% clear all
% 
% load('monkeydata_training.mat');
% num = 80;
% training_data = trial(1:num, :);   % First 10 trials for training
% testData = trial(num+1:end, :);  
function [x, y, modelParameters] = positionEstimator(testData, modelParameters)
%%
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
[X_new_processed] = preprocessing(testData,bin_size);

% 2. Apply PCA on the new data using the stored PCA coefficients
% Define num_features based on the PCA coefficients (rows of coeff_hand)
num_features = size(modelParameters.coeff_hand, 1);  % Get the number of original features

% Ensure X_new_processed has the same number of features as the training data
if size(X_new_processed, 2) < num_features
    % Pad with zeros if new data has fewer features
    X_new_processed = [X_new_processed, zeros(1, num_features - size(X_new_processed, 2))];
elseif size(X_new_processed, 2) > num_features
    % Truncate if new data has more features (though this should not happen)
    X_new_processed = X_new_processed(:, 1:num_features);
end

d_reduced_new = X_new_processed * modelParameters.coeff_hand;

lda_component_truncated = modelParameters.lda_component(1:size(d_reduced_new, 2), :);

% 3. Apply LDA using the stored LDA components
lda_transformed_new = d_reduced_new * lda_component_truncated;

% Step 4: Manually compute the Euclidean distance between the new data and each centroid
num_centroids = size(modelParameters.centroids, 1);  % Number of centroids (8 in your case)
distances = zeros(num_centroids, 1);  % Initialize the distances array

% Compute the Euclidean distance between lda_transformed_new and each centroid
for i = 1:num_centroids
    distance = sqrt(sum((lda_transformed_new - modelParameters.centroids(i, :)).^2));
    distances(i) = distance;  % Store the distance
end

% Step 5: Find the index of the closest centroid
[~, predicted_cluster_idx] = min(distances);

cluster_angle_mapping = modelParameters.cluster_angle_mapping;

% Step 6: Map the predicted cluster index back to the corresponding reach angle
predicted_reach_angle = modelParameters.cluster_angle_mapping(predicted_cluster_idx);
modelParameters.prev_angle = predicted_reach_angle;

% Display the predicted reach angle
%disp('Predicted Reach Angle:');
%disp(predicted_reach_angle);

reach_angles = modelParameters.reach_angles;

% Identify movement direction (use a classifier or predefined method)
[~, angle_idx] = min(abs(reach_angles - predicted_reach_angle));
modelParameters.actualLabel = angle_idx;

decoded_x = zeros(1, num_bins);
decoded_y = zeros(1, num_bins);

for t = 1:num_bins
    % Extract features for current time bin
    X_current = X_new_processed(:, (t-1) * 98 + 1 : t * 98);

    % Predict using the corresponding model
    decoded_x(t) = predictRandomForest(modelParameters.rf_x{angle_idx, t}, X_current);
    decoded_y(t) = predictRandomForest(modelParameters.rf_y{angle_idx, t}, X_current);
end

x = decoded_x(end);  % The final predicted x position
y = decoded_y(end);  % The final predicted y position

function y_pred = predictRandomForest(forest, X_new)
    % Predict using the Random Forest (average tree predictions)
    num_trees = length(forest);
    predictions = zeros(num_trees, size(X_new,1));

    for i = 1:num_trees
        predictions(i, :) = predictDecisionTree(forest(i).tree, X_new);
    end

    y_pred = mean(predictions, 1); % Average predictions
end

function y_pred = predictDecisionTree(tree, X)
    % Predict using the manually trained decision tree
    y_pred = zeros(size(X,1),1);

    for i = 1:size(X,1)
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



%% Preprocess Data - padding for handpos and spikes
function [X_data] = preprocessing(t_data,bin_size)
start = 301;
end_remove = 100;

% Define parameters
sigma = 20; % Standard deviation of Gaussian window (in ms)
window = fspecial('gaussian', [1, 5*sigma], sigma); % Gaussian kernel

% Extract spike train and corresponding hand positions
spikes = t_data.spikes;   % 98 x T binary matrix

% Truncate from 301:T_max
%spikes = spikes(:, start:end-end_remove)

len = length(spikes);

% Convolve each neuron's spike train with Gaussian kernel
smoothed_spikes = conv2(spikes, window, 'same'); % 98 x T_max

% Downsample to 20ms bins
num_b = floor(len/ bin_size);
firing_rates = zeros(size(spikes,1), num_b); % 98 x num_bins

for bin = 1:num_b
    idx_start = (bin - 1) * bin_size + 1;
    idx_end = bin * bin_size;
    firing_rates(:, bin) = mean(smoothed_spikes(:, idx_start:idx_end), 2);
end
% Flatten firing rate matrix into a feature vector
feature_vector = reshape(firing_rates, [], 1)'; % (98*num_bins) x 1

% Store the data
X_data = feature_vector; % Feature matrix
X_data = X_data * 1000; % Scale firing rates
end
%% 
function predicted_direction = predict_direction(lda_component_test, cluster_centroids, cluster_angle_mapping)
    % This function predicts the reaching direction for a new test data point
    % lda_component_test: The reduced test data (one or more data points in LDA space)
    % cluster_centroids: Centroids of the clusters from K-means
    % cluster_angle_mapping: The angle mapping for each cluster (the most common angle for each cluster)
    
    % Number of clusters
    num_clusters = size(cluster_centroids, 1);
    
    % Compute the Euclidean distance from the test data to each centroid
    distances = zeros(size(lda_component_test, 1), num_clusters);
    
    % Calculate the Euclidean distance to each centroid
    for k = 1:num_clusters
        distances(:, k) = sqrt(sum((lda_component_test - cluster_centroids(k, :)).^2, 2)); % Distance to centroid k
    end
    
    % Find the closest centroid (smallest distance)
    [~, predicted_cluster_idx] = min(distances, [], 2);  % Find the index of the closest centroid
    
    % Map the predicted cluster index to the corresponding angle using the cluster_angle_mapping
    predicted_direction = cluster_angle_mapping(predicted_cluster_idx);  % Predicted direction in radians
    
    % If you want to display it as a human-readable angle, you can convert it to degrees:
    % predicted_direction_deg = rad2deg(predicted_direction);  % Convert to degrees if needed
end
%%

end


