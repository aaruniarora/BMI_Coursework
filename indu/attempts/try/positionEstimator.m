function [x, y] = posestimate(test_data, modelParameters)


    % Get the min_length from the model parameters
    min_length = modelParameters.min_length;
    
    % Preprocess the test data (same steps as training)
    [X_test, ~] = preprocessData(test_data);  % Ignore the min_length output from here
    
    % If test data has more time points than the min_length, truncate it
    if size(X_test, 1) > min_length
        X_test = X_test(1:min_length, :);
    end
    
    % Apply PCA (using the PCA parameters from the trained model)
    X_test_centered = X_test - modelParameters.pca.params.mean;  % Use the PCA mean from training
    X_test_pca = X_test_centered * modelParameters.pca.V;  % Apply PCA transformation using eigenvectors
    
    % Apply LDA (using the LDA parameters from the trained model)
    X_test_lda = X_test_pca * modelParameters.lda.params.eigenvectors;  % Use LDA eigenvectors
    
    % Predict positions using the linear regression model (beta)
    Y_pred = X_test_lda * modelParameters.regression.beta;  % Apply regression (beta) to LDA output
    
    % Use the average starting position from the training data
    initial_position = modelParameters.avg_starting_position;  % Get the average starting position from training
    
    % Predicted angles (in radians) are mapped to the unit circle for 2D positions
    x_pred = cos(Y_pred);  % Convert predicted angles to x-coordinates
    y_pred = sin(Y_pred);  % Convert predicted angles to y-coordinates
    
    % Add the initial position to the predicted coordinates
    x = x_pred + initial_position(1);  % Shift x predictions by the initial position x-component
    y = y_pred + initial_position(2);  % Shift y predictions by the initial position y-component


function [X, min_length] = preprocessData(test_data)
    % Preprocesses the neural data for PCA, LDA, and linear regression for test data.
    % INPUT:
    %   test_data: A struct array where each entry corresponds to a trial.
    %              Each trial contains spike data (NxM) and other trial-specific data.
    % OUTPUT:
    %   X: Feature matrix (each row is a feature vector).
    %   min_length: The minimum number of time points after removing 300ms and 100ms for each trial.

    % Constants
    first_300ms_idx = 300;  % First 300 ms to remove (assuming 1 ms per time step)
    last_100ms_idx = 100;   % Last 100 ms to remove (assuming 1 ms per time step)
    min_firing_rate = 0.1;   % Minimum firing rate (Hz) to keep neurons
    window_size = 50;        % Standard deviation for the Gaussian window for spike count smoothing
    
    % Initialize variables
    num_trials = size(test_data, 1);  % Number of trials
    spikes_data = cell(num_trials, 1); % To store spike data for each trial (no angles in test data)
    
    % Step 1: Extract spike data for all trials
    for trial_idx = 1:num_trials
        % Access spike data for each trial (assuming only 1 condition per trial in test data)
        spikes_data{trial_idx} = test_data(trial_idx).spikes;  % No angle, just trial-specific spike data
    end
    
    % Step 2: Truncate the first 300 ms and the last 100 ms for each trial
    for trial_idx = 1:num_trials
        total_time_points = size(spikes_data{trial_idx}, 2);  % Get total time points from columns of spikes
        
        % Ensure there are enough time points to truncate 300ms at the start and 100ms at the end
        if total_time_points > first_300ms_idx + last_100ms_idx
            % Truncate the first 300 ms and the last 100 ms
            truncated_spikes = spikes_data{trial_idx}(:, first_300ms_idx + 1:end - last_100ms_idx);
        else
            % If not enough data, skip this trial (or handle it based on your requirements)
            %error('Not enough time points for truncation after removing 300ms and 100ms.');
            truncated_spikes = spikes_data;
        end
        
        % Update spikes_data with the truncated data
        spikes_data{trial_idx} = truncated_spikes;
    end
    
    % Step 3: Compute the minimum length of the data (after truncation)
    min_length = min(cellfun(@(spikes) size(spikes, 2), spikes_data)); % Using columns (time points)

    % Prepare the output feature matrix
    X = [];

    % Step 4: Preprocess each trial
    for trial_idx = 1:num_trials
        % Extract truncated spike data, use the computed min_length
        trial_spikes = spikes_data{trial_idx}(:, 1:min_length);  % Truncate to min_length
        
        % Step 5: Smooth spike counts with a Gaussian filter (manual implementation)
        smoothed_spikes = gaussian_filter(trial_spikes, window_size);
        
        % Step 6: Compute spike count for each neuron
        spike_counts = sum(smoothed_spikes, 1);
        
        % Step 7: Remove low firing neurons (firing rate below min_firing_rate)
        firing_rates = spike_counts / min_length;  % Firing rate in Hz
        good_neurons = firing_rates >= min_firing_rate;  % Neurons to keep
        
        % Step 8: Keep only the good neurons
        trial_spikes_filtered = trial_spikes(:, good_neurons);
        
        % Step 9: Append the spike data for feature vector
        trial_features = trial_spikes_filtered';  % Transpose to get correct alignment
        
        % Step 10: Append to feature matrix X
        X = [X; trial_features];
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