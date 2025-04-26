function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSITION ESTIMATOR
%
% Uses trained model parameters to:
%   1. Preprocess incoming spike data (binning, sqrt, smoothing)
%   2. Extract and reshape features from the current trial
%   3. Classify intended movement direction using soft kNN
%   4. Predict x and y hand position using regression coefficients
%
% Inputs:
%   test_data        - A struct representing a single trial
%   modelParameters  - Learned parameters from training
%
% Outputs:
%   x, y             - Estimated hand position (in mm)
%   modelParameters  - Updated with current predicted direction label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Parameters
    bin_group = 20; % Time bin width in ms
    alpha = 0.3; % EMA smoothing factor
    sigma = 50; % Std. deviation for Gaussian filter

    start_idx = modelParameters.start_idx;
    stop_idx = modelParameters.stop_idx;
    directions = modelParameters.directions; % get the number of angles
    polyDegree = modelParameters.polyd;

    if ~isfield(modelParameters, 'actLabel')
        modelParameters.actLabel = []; % Default label
    end

    %% Soft kNN parameters
    k = 20;    % Number of neighbors for kNN (8 for hard kNN and 20 for soft)
    pow = 1;   % Power factor for distance-based weighting
    alp = 1e-6; % Scaling for exponential weighting

   %% Preprocess the trial data
   preprocessed_test = preprocessing(test_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
   neuron_len = size(preprocessed_test(1,1).rate, 1);

   %% Use indexing based on data given
   curr_bin = size(test_data.spikes, 2);
   idx = min( max( floor( ( curr_bin - start_idx ) / bin_group) + 1, 1), length(modelParameters.class));

   %%  Remove low firing neurons for better accuracy
   spikes_test = extract_features(preprocessed_test, neuron_len, curr_bin/bin_group, 'nodebug');
   removed_neurons = modelParameters.removeneurons;
   spikes_test(removed_neurons, :) = [];

   %% Reshape dataset: Flatten spike data into column vector
   spikes_test = reshape(spikes_test, [], 1);

   %% Predict movement direction using kNN classification
   if curr_bin <= stop_idx 
       % Extract LDA projections and mean firing for the current bin
       train_weight = modelParameters.class(idx).lda_weights;
       test_weight =  modelParameters.class(idx).lda_outputs;
       curr_firing_mean = modelParameters.class(idx).mean_firing;
       
       % Project test spike vector to LDA space
       test_weight = test_weight' * (spikes_test(:) - curr_firing_mean(:));

       % Classify using hard or soft kNN. Soft kNN can be distance (dist) or exponential (exp) based weighting
       output_label = KNN_classifier(directions, test_weight, train_weight, k, pow, alp, 'soft', 'dist');

   else 
       % After max time window, retain previous classification
       % output_label = mode(modelParameters.actLabel);
       output_label = modelParameters.actualLabel;
   end

    % if modelParameters.trial_id == 0
    % modelParameters.trial_id = test_data.trialId;
    % else 
    % if modelParameters.trial_id ~= test_data.trialId
    %     modelParameters.iterations = 0;
    %     modelParameters.trial_id = test_data.trialId;
    %     modelParameters.actLabel = [];
    % end
    % end
    % modelParameters.iterations = modelParameters.iterations + 1;
    % 
    % % disp(modelParameters.actualLabel)
    % 
    % %% Reset `actualLabel` if there are repeated inconsistencies
    % if ~isempty(modelParameters.actLabel)
    %     if modelParameters.actLabel(end) ~= output_label
    %         if length(modelParameters.actLabel) > 10 && sum(modelParameters.actLabel(end-4:end) ~= output_label) >= 5
    %             % If the last 5 classifications contain at least 3 mismatches, reset
    %             modelParameters.actLabel = [];
    %         end
    %     end
    % end
    % 
    % len_b_mode = 7;
    % % Update the actual label in model parameters
    % if ~isempty(modelParameters.actLabel)
    % 
    % % Accumulate stable labels before following the mode
    % if length(modelParameters.actLabel) > len_b_mode  % Wait until there are at least 5 labels
    %     output_label = mode(modelParameters.actLabel);
    % end
    % modelParameters.actLabel(end+1) = output_label;
    % modelParameters.actLabel(:) = output_label;  % Ensure all entries are consistent
    % else
    %     % For the very first classification, just set the label
    %     modelParameters.actLabel(end+1) = output_label;
    % end 
    % 
    % output_label = modelParameters.actLabel(end);
    % modelParameters.actualLabel = modelParameters.actLabel(end); 

    modelParameters.actualLabel = output_label; 
    
   %% -----------  Kalman decoding  ----------------------------------------
    % Which direction’s filter?
    dir = output_label;
    
    % Keep a persistent Kalman state per trial
    if ~isfield(modelParameters,'KF') ...
          || test_data.trialId ~= modelParameters.KF.trial_id
        % --- first call of a new trial: initialise ---
        modelParameters.KF.trial_id = test_data.trialId;
        modelParameters.KF.x_hat    = modelParameters.kalman(dir).Pi;
        modelParameters.KF.P        = modelParameters.kalman(dir).Vi;
    end
    
    Kpar = modelParameters.kalman(dir);          % shorthand
    A = Kpar.A;  C = Kpar.C;  Q = Kpar.Q;  R = Kpar.R;
    
    % Build current observation vector z(k) exactly like during training
    zk = spikes_test;   % already binned / EMA / neurons‑removed & vectorised
    
    % ---------- Predict step ----------------------------------------------
    x_pred = A * modelParameters.KF.x_hat;
    P_pred = A * modelParameters.KF.P * A' + Q;
    
    % ---------- Update step -----------------------------------------------
    S   = C * P_pred * C' + R;                  % innovation covariance
    K   = P_pred * C' / S;                      % Kalman gain  (pinv ok too)
    x_upd = x_pred + K * (zk - C * x_pred);
    P_upd = (eye(size(A)) - K*C) * P_pred;
    
    % ----------  Save back & output position ------------------------------
    modelParameters.KF.x_hat = x_upd;
    modelParameters.KF.P     = P_upd;
    
    x = x_upd(1);
    y = x_upd(2);

end