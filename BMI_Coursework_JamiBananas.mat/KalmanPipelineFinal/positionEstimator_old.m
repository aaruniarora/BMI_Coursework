function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSITION ESTIMATOR (KF Version with Time‐Window Indexing)
%
% Decodes the current hand position from a test trial using the pre-trained
% model parameters. This version uses the time-bin indexing method from the
% kNN_PCR code to select the appropriate Kalman filter (trained for a given
% window length) based on the amount of data available. It then performs a KF
% update using the measurement obtained from the test spike data.
%
% Outputs:
%   x, y            - Estimated hand position (in mm)
%   modelParameters - Updated with current KF state and decoded trajectory.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% 1. Set parameters
    bin_group = 20; % must match training
    alpha = 0.3;
    sigma = 50;
    
    start_idx  = modelParameters.start_idx; % e.g., 320 ms
    stop_idx   = modelParameters.stop_idx;  % end of training window
    directions = modelParameters.directions;
    nWindows   = modelParameters.nWindows;  % number of KF models stored
    
    % Current time in test trial (ms)
    curr_time = size(test_data.spikes, 2);
    % Determine current number of complete bins in test data.
    curr_bins = floor(curr_time / bin_group);
    % Determine the window index as follows:
    % We assume that training started at start_idx, which corresponds to bin index:
    start_bin_idx = ceil(start_idx/bin_group);
    % The window index (win_idx) is how many bins beyond start_idx we have:
    win_idx = curr_bins - start_bin_idx + 1;
    % Clip win_idx to be within [1, nWindows]
    win_idx = min(max(win_idx, 1), nWindows);
    
    %% 2. Preprocess test data
    preprocessed_test = preprocessing(test_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
    neuron_len = size(preprocessed_test(1,1).rate, 1);
    
    % Extract test neural data over the current window:
    start_bin = start_bin_idx;
    end_bin = start_bin_idx + win_idx - 1;
    if size(preprocessed_test(1,1).rate,2) < end_bin
        % Not enough bins; return last known state.
        d = modelParameters.actualLabel;
        x_update = modelParameters.kf(win_idx,d).state;
        x = x_update(1);
        y = x_update(2);
        return;
    end
    rate = preprocessed_test(1,1).rate(:, start_bin:end_bin);
    % Remove low-firing neurons
    rate(modelParameters.removeneurons, :) = [];
    
    % For KF measurement, we follow the same procedure as in training:
    % Each test sample is the sequence in the current window.
    % We form the measurement vector by flattening (column-wise).
    test_spk_vec = reshape(rate, [], 1);
    
    %% 3. Apply the measurement model (PCA) from the matching training window.
    % For each direction d, we will compute the KF innovation and select the one
    % with the lowest Mahalanobis distance.
    best_d = 1;
    best_d_score = Inf;
    for d = 1:directions
        % Retrieve PCA model for current window & direction.
        pca_coeff = modelParameters.kf(win_idx,d).pca_coeff;
        mean_spk  = modelParameters.kf(win_idx,d).mean_spk;
        
        % Center and project the test feature vector.
        test_spk_centered = test_spk_vec - mean_spk;
        z_test = pca_coeff' * test_spk_centered;  % measurement vector
        
        % Retrieve KF parameters for direction d.
        A = modelParameters.kf(win_idx,d).A;
        H = modelParameters.kf(win_idx,d).H;
        Q = modelParameters.kf(win_idx,d).Q;
        R = modelParameters.kf(win_idx,d).R;
        x_prev = modelParameters.kf(win_idx,d).state;
        P_prev = modelParameters.kf(win_idx,d).P;
        
        % Prediction step.
        x_pred = A * x_prev;
        P_pred = A * P_prev * A' + Q;
        
        % Innovation and covariance.
        innov = z_test - H * x_pred;
        S = H * P_pred * H' + R;
        
        % Compute Mahalanobis distance.
        d_score = innov' / S * innov;
        
        if d_score < best_d_score
            best_d_score = d_score;
            best_d = d;
        end
    end
    
    % Use the best direction's KF for update.
    d = best_d;
    A = modelParameters.kf(win_idx,d).A;
    H = modelParameters.kf(win_idx,d).H;
    Q = modelParameters.kf(win_idx,d).Q;
    R = modelParameters.kf(win_idx,d).R;
    x_prev = modelParameters.kf(win_idx,d).state;
    P_prev = modelParameters.kf(win_idx,d).P;
    
    x_pred = A * x_prev;
    P_pred = A * P_prev * A' + Q;
    K = P_pred * H' / (H * P_pred * H' + R);
    x_update = x_pred + K * (z_test - H * x_pred);
    P_update = (eye(size(K,1)) - K * H) * P_pred;
    
    modelParameters.kf(win_idx,d).state = x_update;
    modelParameters.kf(win_idx,d).P = P_update;
    modelParameters.actualLabel = d;
    
    %% 4. Output estimated position (x, y)
    x = x_update(1);
    y = x_update(2);
    
    if ~isfield(modelParameters, 'decodedHandPos') || isempty(modelParameters.decodedHandPos)
        modelParameters.decodedHandPos = [x; y];
    else
        modelParameters.decodedHandPos(:, end+1) = [x; y];
    end
end
