function model_params = positionEstimatorTraining(training_data)
    % Preprocess the training data
    [X, Y, min_length] = preprocessData(training_data);  % Assuming preprocessData returns X, Y, min_length
    
    % Apply PCA
    [X_pca, pca_params, V, X_centered] = applyPCA(X);
    
    % Apply LDA
    [X_lda, lda_params] = applyLDA(X_pca, Y);
    
    % Initialize the model_params structure to store all parameters
    model_params = struct;

    % Train Linear Regression Model (Beta is the solution to X_lda \ Y)
    model_params.regression.beta = X_lda \ Y;  % Store regression coefficients
    
    % Compute the average starting position across all trials (first handpos)
    starting_positions = zeros(length(training_data), 2);  % Store x, y positions
    for trial_idx = 1:length(training_data)
        starting_positions(trial_idx, :) = training_data(trial_idx).handPos(1:2,1);  % First hand position of each trial
    end
    
    % Calculate the average starting position
    avg_starting_position = mean(starting_positions, 1);
    
    % Add parameters to the model_params struct
    model_params.pca = struct;
    model_params.pca.params = pca_params;  % Store PCA parameters (mean, eigenvectors, eigenvalues)
    model_params.pca.V = V;  % Store eigenvectors
    model_params.pca.X_centered = X_centered;  % Store centered X data
    
    model_params.lda = struct;
    model_params.lda.params = lda_params;  % Store LDA parameters
    
    model_params.avg_starting_position = avg_starting_position;  % Add the avg starting position
    
    model_params.min_length = min_length;  % Store min_length (as provided by preprocessData)
    
    % Return the model parameters with all necessary data
    return

function [X, Y, min_length] = preprocessData(training_data)
    % Preprocesses the neural data for PCA, LDA, and linear regression.
    % INPUT:
    %   training_data: A struct array where each entry corresponds to a trial.
    %                  Each trial contains a matrix of size 8 x 1 where each column contains handPos (Nx2) and spikes (NxM).
    % OUTPUT:
    %   X: Feature matrix (each row is a feature vector).
    %   Y: Target labels (for supervised tasks).
    %   min_length: The minimum number of time points after removing 300ms and 100ms for each trial.

    % Constants
    first_300ms_idx = 300;  % First 300 ms to remove (assuming 1 ms per time step)
    last_100ms_idx = 100;   % Last 100 ms to remove (assuming 1 ms per time step)
    min_firing_rate = 0.1;   % Minimum firing rate (Hz) to keep neurons
    window_size = 50;        % Standard deviation for the Gaussian window for spike count smoothing
    
    % Reach angles (target labels for each of the 8 conditions)
    reach_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23/18, 31/18, 35/18] * pi;
    
    % Initialize variables
    num_trials = size(training_data);
    num_trials = num_trials(1); % Number of trials
    hand_positions = cell(num_trials, 8);  % To store hand positions (x, y) for each angle
    spikes_data = cell(num_trials, 8);     % To store spike data (for neurons) for each angle
    
    % Step 1: Extract hand positions (only x, y) and spike data for all trials and angles
    for trial_idx = 1:num_trials
        for angle_idx = 1:8
            % Access hand positions and spikes for each angle (stored in columns)
            hand_positions{trial_idx, angle_idx} = training_data(trial_idx, angle_idx).handPos(1:2,:);  % x and y only
            spikes_data{trial_idx, angle_idx} = training_data(trial_idx, angle_idx).spikes;
        end
    end
    
    % Step 2: Truncate the first 300 ms and the last 100 ms for each trial
    for trial_idx = 1:num_trials
        for angle_idx = 1:8
            total_time_points = size(spikes_data{trial_idx, angle_idx}, 2);  % Get total time points from columns of spikes
            
            % Ensure there are enough time points to truncate 300ms at the start and 100ms at the end
            if total_time_points > first_300ms_idx + last_100ms_idx
                % Truncate the first 300 ms and the last 100 ms
                truncated_handpos = hand_positions{trial_idx, angle_idx}(:, first_300ms_idx + 1:end - last_100ms_idx);
                truncated_spikes = spikes_data{trial_idx, angle_idx}(:, first_300ms_idx + 1:end - last_100ms_idx);
            else
                % If not enough data, skip this trial (or handle it based on your requirements)
                error('Not enough time points for truncation after removing 300ms and 100ms.');
            end
            
            % Update hand_positions and spikes_data with the truncated data
            hand_positions{trial_idx, angle_idx} = truncated_handpos;
            spikes_data{trial_idx, angle_idx} = truncated_spikes;
        end
    end
    
    % Step 3: Compute the minimum length of the data (after truncation)
    min_length = min(min(cellfun(@(spikes) size(spikes, 2), spikes_data))); % Using columns (time points)

    % Prepare the output feature matrix and target labels (Y)
    X = [];
    Y = [];

    % Step 4: Preprocess each trial and angle
    for trial_idx = 1:num_trials
        for angle_idx = 1:8
            % Extract truncated hand positions and spikes, use the computed min_length
            trial_handpos = hand_positions{trial_idx, angle_idx}(:, 1:min_length);  % Truncate to min_length
            trial_spikes = spikes_data{trial_idx, angle_idx}(:, 1:min_length);  % Truncate to min_length
            
            % Step 5: Smooth spike counts with a Gaussian filter (manual implementation)
            smoothed_spikes = gaussian_filter(trial_spikes, window_size);
            
            % Step 6: Compute spike count for each neuron
            spike_counts = sum(smoothed_spikes, 1);
            
            % Step 7: Remove low firing neurons (firing rate below min_firing_rate)
            firing_rates = spike_counts / min_length;  % Firing rate in Hz
            good_neurons = firing_rates >= min_firing_rate;  % Neurons to keep
            
            % Step 8: Keep only the good neurons
            trial_spikes_filtered = trial_spikes(:, good_neurons);
            
            % Step 9: Concatenate hand positions and spike data for feature vector
            trial_features = [trial_handpos', trial_spikes_filtered'];  % Transpose to get correct alignment
            
            % Step 10: Append to feature matrix X and target labels Y
            % Replicate the angles for each time point in the trial
            trial_angle = reach_angles(angle_idx);  % Angle for the current condition
            
            % Append features and angles to X and Y
            X = [X; trial_features];
            Y = [Y; repmat(trial_angle, size(trial_features, 1), 1)];  % The angle as label
        end
    end

function smoothed_data = gaussian_filter(data, window_size)
    % Implements a 1D Gaussian smoothing filter on the data
    N = size(data, 2);  % Number of time points (columns)
    M = size(data, 1);  % Number of neurons (rows)
    
    % Create a Gaussian window (standard deviation is window_size)
    t = -floor(3*window_size):floor(3*window_size);
    gaussian_kernel = exp(-0.5 * (t / window_size).^2);
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);  % Normalize the kernel
    
    % Apply the Gaussian kernel to each row (neuron) of the data
    smoothed_data = zeros(M, N);
    for neuron = 1:M
        smoothed_data(neuron, :) = conv(data(neuron, :), gaussian_kernel, 'same');
    end
end
end
    

function [X_pca, pca_params,V,X_centered] = applyPCA(X)
    % Center the data
    mu = mean(X, 1);
    X_centered = X - mu;
    
    % Compute covariance matrix manually
    C = (X_centered' * X_centered) / (size(X_centered, 1) - 1);
    
    % Eigen decomposition (no toolbox)
    [V, D] = eig(C);
    
    % Sort eigenvectors by eigenvalues
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Project data onto the principal components
    X_pca = X_centered * V;
    
    pca_params.mean = mu;
    pca_params.eigenvectors = V;
end

function [X_lda, lda_params] = applyLDA(X, Y)
    unique_classes = unique(Y, 'rows');  % Unique class labels
    num_classes = size(unique_classes, 1);
    
    % Compute class means
    class_means = zeros(num_classes, size(X, 2));
    for i = 1:num_classes
        class_means(i, :) = mean(X(all(bsxfun(@eq, Y, unique_classes(i, :)), 2), :), 1);
    end
    
    % Compute between-class and within-class scatter matrices
    Sb = zeros(size(X, 2));
    Sw = zeros(size(X, 2));
    
    for i = 1:num_classes
        class_samples = X(all(bsxfun(@eq, Y, unique_classes(i, :)), 2), :);
        class_mean = class_means(i, :);
        
        % Fix: Correctly compute between-class scatter matrix
        Sb = Sb + size(class_samples, 1) * (class_mean - mean(X, 1))' * (class_mean - mean(X, 1));
        
        % Compute within-class scatter matrix
        class_diff = class_samples - repmat(class_mean, size(class_samples, 1), 1);
        Sw = Sw + class_diff' * class_diff;
    end
    
    % Solve generalized eigenvalue problem: Sb * v = lambda * Sw * v
    [V, D] = eig(Sb, Sw);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    
    % Project data onto the LDA space
    X_lda = X * V;
    
    lda_params.eigenvectors = V;
end

function model_params = trainLinearRegression(X, Y)
    % Use MATLAB's mldivide for linear regression (optimized least squares)
    model_params.beta = X \ Y;
end

end
