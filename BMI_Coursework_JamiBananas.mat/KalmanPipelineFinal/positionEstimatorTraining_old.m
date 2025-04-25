function modelParameters = positionEstimatorTraining(training_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSITION ESTIMATOR TRAINING FUNCTION (KF Version with Time‐Window Indexing)
%
% This function trains a continuous position estimator using neural spikes.
% It first pre-processes the data (binning, sqrt transform, smoothing), then
% removes low‐firing neurons, and finally – for a series of time windows – it
% extracts features, computes a PCA-based measurement model, and trains a
% Kalman filter for each movement direction.
%
% The time-bin indexing is the same as in the provided kNN_PCR code: the
% training window starts at start_idx and ends at stop_idx. For each time
% window (of length win, in bins) a separate KF is trained and stored.
%
% Outputs:
%   modelParameters - structure that contains:
%      .start_idx, .stop_idx, .directions, .nWindows
%      .removeneurons - indices of neurons excluded for low firing
%      .kf(win, d) - for each time window (win=1:nWindows) and each direction d,
%                     the following fields are stored:
%                          A, H, Q, R, state, P,
%                          pca_coeff (for measurement projection),
%                          mean_spk (mean neural feature vector).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% 1. Set parameters
    [training_length, directions] = size(training_data); 
    bin_group = 20;         % bin width (ms)
    alpha = 0.35;           % EMA smoothing constant
    sigma = 50;             % Gaussian filter sigma (ms)
    start_idx = 300 + bin_group;  % e.g. 320 ms
    
    % Determine the minimum trial length (in ms)
    spike_cells = {training_data.spikes};
    min_time_length = min(cellfun(@(sp) size(sp, 2), spike_cells));
    clear spike_cells;
    
    % Define end of training window: from start_idx up to the largest multiple
    % of bin_group available below min_time_length.
    stop_idx = floor((min_time_length - start_idx) / bin_group) * bin_group + start_idx;
    % Define the discrete time bins used in training.
    time_bins = start_idx:bin_group:stop_idx;
    nWindows = length(time_bins);
    
    % Store basic parameters.
    modelParameters.start_idx   = start_idx;
    modelParameters.stop_idx    = stop_idx;
    modelParameters.directions  = directions;
    modelParameters.nWindows    = nWindows;
    modelParameters.trial_id    = 0;
    modelParameters.iterations  = 0;

    %% 2. Preprocess neural data (binning, sqrt transform, smoothing)
    preprocessed_data = preprocessing(training_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
    orig_neurons = size(preprocessed_data(1,1).rate, 1);
    
    % Determine low-firing neurons using the full window (for consistency).
    full_bins = stop_idx / bin_group;  
    [spikes_full, ~] = extract_features(preprocessed_data, orig_neurons, full_bins, 'nodebug');
    removeneurons = remove_neurons(spikes_full, orig_neurons, 'nodebug');
    modelParameters.removeneurons = removeneurons;
    
    %% 3. Train Kalman filter models indexed by time window
    % For each time window win (i.e. using win bins starting at start_idx)
    % we will accumulate training samples from each trial and direction.
    for win = 1:nWindows
        % Current window length in bins
        curr_bins = win;
        for d = 1:directions
            X_cell = {};  % will store state sequences for trials
            Z_cell = {};  % will store corresponding neural measurement sequences
            for tr = 1:training_length
                % Check if the trial is long enough
                if size(training_data(tr,d).handPos,2) < (start_idx + (curr_bins-1)*bin_group)
                    continue;
                end
                % Extract hand position from start_idx with spacing = bin_group, for curr_bins bins
                pos = training_data(tr,d).handPos(1:2, start_idx:bin_group:(start_idx+(curr_bins-1)*bin_group));
                if size(pos,2) < 2
                    continue;
                end
                % Compute velocity (difference) and replicate last column
                vel = diff(pos, 1, 2);
                vel(:,end+1) = vel(:,end);
                X_curr = [pos; vel];  % dimension: 4 x curr_bins
                
                % Extract neural data for this trial & direction:
                % In preprocessed_data, each column is one bin.
                start_bin_idx = ceil(start_idx/bin_group);
                end_bin_idx = start_bin_idx + curr_bins - 1;
                rate = preprocessed_data(tr,d).rate;
                if size(rate,2) < end_bin_idx
                    continue;
                end
                spk = rate(:, start_bin_idx:end_bin_idx);  % dimension: orig_neurons x curr_bins
                % Remove low-firing neurons
                spk(removeneurons, :) = [];
                if size(spk,2) ~= size(X_curr,2)
                    continue;
                end
                % For KF training we treat each time step in the window as one measurement.
                % Store the entire sequence (each column is one time point).
                X_cell{end+1} = X_curr;    % 4 x curr_bins
                Z_cell{end+1} = spk;         % n_used x curr_bins
            end
            if isempty(X_cell) || isempty(Z_cell)
                continue;
            end
            % Concatenate training samples from all trials along time axis.
            % That is, assume each trial provides a sequence; we concatenate them
            % so that columns correspond to sequential time steps from all trials.
            X_all = cat(2, X_cell{:});        % 4 x (total_samples)
            Z_all = cat(2, Z_cell{:});        % (n_used) x (total_samples)
            
            % Compute PCA on the neural measurements for this window.
            [pca_coeff, ~, ~] = perform_PCA(Z_all, 0.44, 'nodebug');
            mean_spk = mean(Z_all,2);
            % Project measurements to obtain the observation sequence.
            Z_proj = pca_coeff' * (Z_all - repmat(mean_spk,1,size(Z_all,2)));
            
            % For Kalman training, we use successive time steps.
            if size(X_all,2) < 2, continue; end
            X0 = X_all(:, 1:end-1);
            X1 = X_all(:, 2:end);
            Z0 = Z_proj(:, 2:end);  % associate each state X1 with measurement from same time
            % Estimate state transition matrix A (4x4)
            A = X1 * X0' / (X0 * X0');
            % Estimate measurement matrix H (maps state X to observation z)
            H = Z0 * X1' / (X1 * X1');
            % Process noise covariance Q and measurement noise covariance R.
            Q = cov((X1 - A*X0)');  % 4x4
            R = cov((Z0 - H*X1)');
            
            % Store KF parameters for this time window and direction.
            modelParameters.kf(win,d).A = A;
            modelParameters.kf(win,d).H = H;
            modelParameters.kf(win,d).Q = Q;
            modelParameters.kf(win,d).R = R;
            % Initialize KF state for each direction as zero state.
            modelParameters.kf(win,d).state = zeros(4,1);
            modelParameters.kf(win,d).P = eye(4);
            % Also store measurement model parameters (for PCA projection).
            modelParameters.kf(win,d).pca_coeff = pca_coeff;
            modelParameters.kf(win,d).mean_spk = mean_spk;
        end % for d
    end % for win
end
