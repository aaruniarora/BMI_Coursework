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

bin_size = 20;
[X_data,Y_data] = preprocessing(training_data,num_trials,num_angles,bin_size);
model_params.firing_data = X_data;
model_params.bin_size = bin_size;

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

[beta_x_all, beta_y_all, unique_angles, mean_X,mean_x,mean_y] = trainAngleWiseRegressionXY(d_reduced, Y_data, angle_labels,num_trials);
model_params.b_x = beta_x_all;
model_params.b_y = beta_y_all;
model_params.meanX = mean_X;
model_params.unique_angles = unique_angles;
model_params.meanx_pos = mean_x;
model_params.meany_pos = mean_y;

model_params.iterations = 0;
model_params.trial_id = 0;
%%
function [beta_x_all, beta_y_all, unique_angles, mean_X,mean_x,mean_y] = trainAngleWiseRegressionXY(d_reduced, Y_data, angle_labels,num_trials)
    unique_angles = unique(angle_labels); % Find all unique angles
    num_angles = length(unique_angles);
    
    beta_x_all = cell(num_angles, 1); % Store regression models for X
    beta_y_all = cell(num_angles, 1); % Store regression models for Y
    
    % Compute the mean of PCA-transformed data
    mean_X = mean(d_reduced, 1); % Compute mean across trials (row-wise)
    
    % Center the data (subtract mean)
    % d_reduced_centered = d_reduced - mean_X;

    mean_x = cell(num_angles, 1);
    mean_y = cell(num_angles, 1);

    angles = 1:8;  % The unique angle labels
    angle_labels_e = repelem(angles,num_trials*2); % Repeat each number 4 times
    
    for i = 1:num_angles
        % Extract trials corresponding to this angle
        angle_idx = (angle_labels == unique_angles(i));
        angle_idx_e = (angle_labels_e == unique_angles(i));
        
        X = d_reduced(angle_idx, :); % Use centered data
        
        % Extract the corresponding hand positions
        Y_selected = Y_data(angle_idx_e, :); % This has shape (num_trials × 2)

        assignin('base', 'y_angle', Y_selected);
        % Separate into X and Y coordinates
        Y_x = Y_selected(1:2:end, :); % Take the first column → X coordinates
        Y_y = Y_selected(2:2:end, :); % Take the second column → Y coordinates

        Y_x = mean(Y_selected(1:2:end, :), 2);  % Collapse across time
        Y_y = mean(Y_selected(2:2:end, :), 2);
        
        assignin('base', 'y_x', Y_x);
        % Compute the means for X and Y hand positions
        mean_X_a = mean(Y_x, 'all'); % Mean of all X positions
        mean_Y_a = mean(Y_y, 'all') ; % Mean of all Y positions
        
        assignin('base', 'y_x_mean', mean_X_a);

        % Y_x = Y_x - mean_X_a;
        % Y_y = Y_y - mean_Y_a;

        % Ensure column vectors for correct matrix multiplication
        Y_x = Y_x(:);
        Y_y = Y_y(:);
        
        % Debugging: Print sizes (optional)
        % size(X)
        % size(Y_x)
        % size(Y_y)

        % Solve for beta_x and beta_y using normal equation (No Bias Term)
        % beta_x_all{i} = (X' * X) \ (X' * Y_x);
        % beta_y_all{i} = (X' * X) \ (X' * Y_y);
        size(X)
        size(Y_x)
        beta_x_all{i} = X \ Y_x;  % Using \ to perform the least squares fit (50 x 1 coefficients)
        beta_y_all{i} = X \ Y_y; 
        mean_x{i} = mean_X_a;
        mean_y{i} = mean_Y_a;
    end
end



%% Preprocess Data - padding for handpos and spikes
    function [X_data,Y_data] = preprocessing(t_data,num_trials,num_angles,bin_size)
start = 301;
end_remove = 100;

% Find the maximum time length across all trials
T_max = -inf; % Initialize with a small value
for angle = 1:num_angles
    for t = 1:num_trials
        T_max= max(T_max, size(t_data(t, angle).spikes, 2));
    end
end

T_max = T_max - end_remove;

% Define parameters
sigma = 20; % Standard deviation of Gaussian window (in ms)
window = fspecial('gaussian', [1, 5*sigma], sigma); % Gaussian kernel

X_data = []; % Initialize feature matrix
Y_data = []; % Initialize labels (hand positions)

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

        % Downsample to 20ms bins
        num_bins = floor(T_max /bin_size);
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
        Y_data = [Y_data; handPos]; % Flatten trajectory
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
%%
% %%
% function [d_reduced,coeff,num_PC,V] = applyPCA(data)
% d_mean = mean(data, 1);
% d_norm = data - d_mean; % Center data (zero mean)
% 
% % Compute SVD
% [U, S, V] = svd(d_norm, 'econ');
% 
% % Compute variance explained
% singular_values = diag(S);
% explained_variance = (singular_values.^2) / sum(singular_values.^2);
% cum_variance = cumsum(explained_variance);
% 
% % Choose number of principal components to retain 95% variance
% variance_threshold = 0.95;
% num_PC = find(cum_variance >= variance_threshold, 1);
% 
% % Reduce data dimensionality
% coeff = V(:,1:num_PC);
% d_reduced = d_norm * V(:,1:num_PC); % Project onto principal components
% end
%%
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