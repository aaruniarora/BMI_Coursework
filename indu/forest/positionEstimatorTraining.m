% clc
% clear all
% 
% load('monkeydata_training.mat');
% num = 80;
% training_data = trial(1:num, :);   % First 10 trials for training
% testData = trial(num+1:end, :);  
function [model_params, firing_data] = positionEstimatorTraining(training_data)
%%
clc
model_params = struct;
num_trials = size(training_data,1);
num_angles = size(training_data,2);
model_params.trial_id = 0;
model_params.iterations = 0;

bin_size = 20;
[X_data,Y_data,num_bins] = preprocessing(training_data,num_trials,num_angles,bin_size);
model_params.firing_data = X_data;
model_params.bin_size = bin_size;
model_params.num_bins = num_bins;

num_PC=50;
% [d_reduced,coeff_hand,num_PC,V_hand] = applyPCA(X_data);
[d_reduced, coeff_hand] = covPCA(X_data, num_PC);
model_params.num_PC = num_PC;
model_params.coeff_hand = coeff_hand;
% model_params.V_hand = V_hand;

lda_no = 3;
[lda_component,predicted_labels,angle_labels] = applyLDA(d_reduced,num_trials,num_angles,num_PC,lda_no);
model_params.lda_no = lda_no;
model_params.lda_component = lda_component;

[cluster_centroids, cluster_idx, cluster_angle_mapping, reach_angles] = applykmeans(lda_component,predicted_labels,angle_labels);
model_params.cluster_idx = cluster_idx;
model_params.centroids = cluster_centroids;
model_params.cluster_angle_mapping = cluster_angle_mapping;
model_params.reach_angles = reach_angles;

num_trees = 10; % Number of trees

for angle = 1:num_angles
    for t = 1:num_bins
        % Get indices of trials for this angle
        angle_indices = (1:num_trials) + (angle - 1) * num_trials;
        
        % Extract features and labels
        X_train = X_data(angle_indices, (t-1) * 98 + 1 : t * 98);
        Y_train_x = Y_data(angle_indices * 2 - 1, t);
        Y_train_y = Y_data(angle_indices * 2, t);

        % Train separate Random Forest models
        model_params.rf_x{angle, t} = trainRandomForest(X_train, Y_train_x, num_trees);
        model_params.rf_y{angle, t} = trainRandomForest(X_train, Y_train_y, num_trees);
    end
end

assignin('base','rf_x',model_params.rf_x)

% Store metadata
model_params.bin_size = bin_size;
model_params.num_bins = num_bins;
model_params.num_angles = num_angles;


function forest = trainRandomForest(X, Y, num_trees)
    forest = struct;
    
    % RF parameters
    max_depth = 10;
    min_samples_split = 2;
    min_samples_leaf = 1;

    for i = 1:num_trees
        idx = randi(size(X,1), size(X,1), 1);  % Bootstrap sample
        X_sample = X(idx, :);
        Y_sample = Y(idx, :);

        forest(i).tree = trainDecisionTree(X_sample, Y_sample, 0, max_depth, min_samples_split, min_samples_leaf);
    end
end

 function tree = trainDecisionTree(X, Y, depth, max_depth, min_samples_split, min_samples_leaf)
    tree = struct;

    % Stopping conditions
    if depth >= max_depth || size(X,1) < min_samples_split || numel(unique(Y)) == 1
        tree.value = mean(Y(:));
        return;
    end

    % Choose split
    tree.split_feature = randi(size(X,2));
    tree.split_value = median(X(:, tree.split_feature));

    % Split data
    left_idx = X(:, tree.split_feature) <= tree.split_value;
    right_idx = ~left_idx;

    % Ensure both child nodes meet min_samples_leaf requirement
    if sum(left_idx) >= min_samples_leaf && sum(right_idx) >= min_samples_leaf
        tree.left = trainDecisionTree(X(left_idx, :), Y(left_idx, :), depth + 1, max_depth, min_samples_split, min_samples_leaf);
        tree.right = trainDecisionTree(X(right_idx, :), Y(right_idx, :), depth + 1, max_depth, min_samples_split, min_samples_leaf);
    else
        tree.value = mean(Y(:));
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
        Y_data = [Y_data; Y_x_binned; Y_y_binned]; 
    end
end
X_data = X_data * 1000; % Scale firing rates
end
%%
function [score, coeff] = covPCA(X, numPC)
    % Compute PCA: center the data, get covariance, then the top nPC eigenvectors.
    mu = mean(X,1);
    Xc = X - mu;
    C = cov(Xc);
    [V, D] = eig(C);
    [d, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    coeff = V(:, 1:numPC);
    score = Xc * coeff;
    % Normalize each principal component (each column) to have unit norm
    % normFactors = sqrt(sum(score.^2, 1));  % 1 x nPC vector, norm of each column
    % score = score ./ repmat(normFactors, size(score, 1), 1);
end
    function [lda_component,predicted_labels,angle_labels] = applyLDA(d_reduced,num_trials,num_angles,num_PC,lda_no)
%Manual LDA Implementation
% Define class labels (use angles as class labels)
angle_labels = repmat(1:num_angles, num_trials, 1);
angle_labels = angle_labels(:); % Flatten into a column vector

% Separate data by class (angle)
X_class = cell(num_angles, 1);
for i = 1:num_angles
    X_class{i} = d_reduced(angle_labels == i, :);
end

% Compute class means and overall mean
mean_overall = mean(d_reduced, 1);
mean_class = zeros(num_angles, num_PC);
for i = 1:num_angles
    mean_class(i, :) = mean(X_class{i}, 1);
end

% Compute the Between-Class Scatter Matrix (S_B)
S_B = zeros(num_PC, num_PC);
for i = 1:num_angles
    n_i = size(X_class{i}, 1);
    mean_diff = mean_class(i, :) - mean_overall;
    S_B = S_B + n_i * (mean_diff' * mean_diff);
end

% Compute the Within-Class Scatter Matrix (S_W)
S_W = zeros(num_PC, num_PC);
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
lda_component = d_reduced * W(:, 1:lda_no);

%% Classify using Nearest Mean Classifier
% Compute the mean of each class in the LDA space
class_means = zeros(num_angles,lda_no);
for i = 1:num_angles
    class_means(i, :) = mean(lda_component(angle_labels == i, :), 1);
end

% Predict labels for each data point based on nearest class mean
predicted_labels = zeros(size(lda_component, 1), 1);
for i = 1:size(lda_component, 1)
    distances = vecnorm(lda_component(i, :) - class_means, 2, 2); % Euclidean distance
    [~, predicted_labels(i)] = min(distances);
end

% Visualize the classification in the LDA space
% figure;
% scatter3(lda_component(:,1), lda_component(:,2),lda_component(:,3), 50, predicted_labels, 'filled');
% xlabel('LDA Component 1');
% ylabel('LDA Component 2');
% title('LDA Classification of PCA-Reduced Data');
% colormap jet;
% colorbar;
% grid on;

end

%%
function[cluster_centroids, cluster_idx, cluster_angle_mapping, reach_angles] = applykmeans(X_lda,predicted_labels,angle_labels)
%Perform K-means clustering manually (without using kmeans function or pdist2)
num_clusters = 8; % Define the number of clusters
max_iter = 1000; % Maximum number of iterations
tol = 1e-1; % Tolerance for convergence

% Define reaching angles in radians
reach_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23/18, 31/18, 35/18] * pi; 

% Initialize centroids by taking the mean position of each angle's data
centroids = zeros(num_clusters, size(X_lda, 2)); % Allocate space for centroids

for k = 1:num_clusters
    % Find all points in X_lda corresponding to the k-th reaching angle
    cluster_points = X_lda(predicted_labels == k, :);
    
    % Set initial centroid as the mean of these points
    if ~isempty(cluster_points)
        centroids(k, :) = mean(cluster_points, 1);
    else
        % If no data points exist for this angle, assign a random point (fallback)
        centroids(k, :) = X_lda(randi(size(X_lda, 1)), :);
    end
end


for iter = 1:max_iter
    % Assign each point to the nearest centroid
    distances = zeros(size(X_lda, 1), num_clusters); % Initialize distance matrix

    % Calculate Euclidean distance manually
    for k = 1:num_clusters
        distances(:, k) = sqrt(sum((X_lda - centroids(k, :)).^2, 2)); % Distance to centroid k
    end

    % Assign clusters based on the closest centroid
    [~, cluster_idx] = min(distances, [], 2); % Assign clusters

    % Save the previous centroids to check for convergence
    prev_centroids = centroids;

    % Recompute the centroids
    for k = 1:num_clusters
        centroids(k, :) = mean(X_lda(cluster_idx == k, :), 1);
    end

    % Check for convergence (if centroids have not changed)
    if norm(centroids - prev_centroids) < tol
        break;
    end
end

% Final cluster centers
cluster_centroids = centroids;

%%
% Map clusters to reaching angles
cluster_angle_mapping = zeros(num_clusters, 1);

for k = 1:num_clusters
    % Find data points in this cluster
    cluster_trials = find(cluster_idx == k);
    
    % Get the predicted angles for this cluster
    cluster_predicted_labels = predicted_labels(cluster_trials);
    
    % Convert labels to reaching angles
    cluster_angles = reach_angles(cluster_predicted_labels);
    
    % Get the most common reaching angle for this cluster
    most_common_angle = mode(cluster_angles);
    
    % Assign the most common angle to this cluster
    cluster_angle_mapping(k) = most_common_angle;
end

%%
% Visualize the LDA projection with clusters colored by the predicted angle mapping
% figure;
% 
% % Use different colors for each cluster
% colors = lines(num_clusters); % Automatically generate distinct colors

% % Plot each data point in the LDA space, colored by cluster assignment
% for k = 1:num_clusters
%     cluster_points = X_lda(cluster_idx == k, :);
%     scatter3(cluster_points(:, 1), cluster_points(:, 2),cluster_points(:,3), 50, 'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', 'k');
%     hold on
% end
% 
% xlabel('LDA Component 1');
% ylabel('LDA Component 2');
% title('LDA Projection with Cluster Assignments');
% grid on;
% colorbar;
% 
% % Update legend to show the assigned reaching angles
% legend(arrayfun(@(k) sprintf('Cluster %d (%.2f rad)', k, cluster_angle_mapping(k)), 1:num_clusters, 'UniformOutput', false));


%%
% Calculate accuracy of cluster-to-angle mapping
correct_count = 0;

for i = 1:length(angle_labels)
    % Convert ground truth angle label (index) to its corresponding radian value
    true_angle = reach_angles(angle_labels(i));  
    
    % Retrieve predicted angle from the mapped cluster
    predicted_angle = cluster_angle_mapping(cluster_idx(i));  

    % Compare the angles in radians
    if abs(predicted_angle - true_angle) < 1e-3  % Allow small tolerance for numerical precision
        correct_count = correct_count + 1;
    end
end
end

 end