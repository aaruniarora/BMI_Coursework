
% clc
% clear all
% 
% load('monkeydata_training.mat');
% num = 80;
% training_data = trial(1:num, :);   % First 10 trials for training
% testData = trial(num+1:end, :);  
function [x, y, modelParameters] = positionEstimator(testData, modelParameters)
%%
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

% Assuming:
% lda_transformed_new is a 1 x 3 matrix (transformed data point after LDA)
% modelParameters.centroids is a 8 x 3 matrix (learned centroids from k-means)
% modelParameters.cluster_angle_mapping is a mapping of cluster indices to angles

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

beta_x_all = modelParameters.b_x;
beta_y_all = modelParameters.b_y;
unique_angles = modelParameters.unique_angles;

mean_X_pos= modelParameters.meanx_pos;
mean_Y_pos = modelParameters.meany_pos;

if modelParameters.iterations == 1
    x = testData.startHandPos(1);
    y = testData.startHandPos(2);
elseif modelParameters.iterations > 1
[x,y] = predictHandPositionByAngleXY(d_reduced_new, predicted_reach_angle, beta_x_all, beta_y_all, cluster_angle_mapping,mean_X_pos, mean_Y_pos);
elseif modelParameters.iterations > 5
[x,y] = predictHandPositionByAngleXY(d_reduced_new,modelParameters.prev_angle, beta_x_all, beta_y_all, cluster_angle_mapping,mean_X_pos, mean_Y_pos);
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
num_bins = floor(len/ bin_size);
firing_rates = zeros(size(spikes,1), num_bins); % 98 x num_bins

for bin = 1:num_bins
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
    function [x,y] = predictHandPositionByAngleXY(d_reduced_new, predicted_angle, beta_x_all, beta_y_all,cluster_angle_mapping,mean_X_pos, mean_Y_pos)
    % Find which model to use based on the predicted angle
    angle_idx = find(cluster_angle_mapping == predicted_angle, 1);
    
    if isempty(angle_idx)
        error('Predicted angle not found in trained models.');
    end
    
    % Select the appropriate regression models for X and Y
    beta_x = beta_x_all{angle_idx};
    beta_y = beta_y_all{angle_idx};

    % Center the new data by subtracting the training mean
    % X_new = d_reduced_new - mean_X(:,1:length(d_reduced_new));
    X_new = d_reduced_new;
    
    % Predict X and Y positions separately (No Bias Term)
    x = (X_new * beta_x) + mean_X_pos{angle_idx};
    y = (X_new * beta_y) + mean_Y_pos{angle_idx};
end

end
