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
    % polyDegree = modelParameters.polyd;

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
   modelParameters.actualLabel = output_label; 
    
    %% Estimate hand position (x, y) using PCR model
    % av_X = modelParameters.averages(idx).av_X(:, output_label);
    % av_Y =  modelParameters.averages(idx).av_Y(:, output_label);
    % meanFiring = modelParameters.pcr(output_label, idx).f_mean;
    % bx = modelParameters.pcr(output_label, idx).bx;
    % by = modelParameters.pcr(output_label, idx).by;
    % reg_meth = modelParameters.reg_meth;
    % 
    % x = position_calc(spikes_test, meanFiring, bx, av_X, curr_bin,reg_meth,polyDegree);
    % y = position_calc(spikes_test, meanFiring, by, av_Y, curr_bin,reg_meth,polyDegree);

    %% Kalman Filtering for (x,y) prediction
    
    predicted_angle = output_label(1);
    
    nb_states = 4; % (x,y,Vx,Vy) states coordinates 
    % NB: testing phase revealed that the inclusion of acceleration components in the state vector did not improved the performance of the decoder. 
    % Therefore, acceleration was excluded (only 4 states are used).
    I = eye(nb_states); 

    % get Kalman filter parameters for predicted angle
    H = modelParameters.H{predicted_angle};
    Q = modelParameters.Q{predicted_angle};
    A = modelParameters.A{predicted_angle};
    W = modelParameters.W{predicted_angle};
    selected_neurons = modelParameters.selected_neurons;

    
    % Compute firing rate 
    zk = test_data.spikes(selected_neurons, start_idx+1:end);

    if isempty(zk) % t=320 ms (no motion)
        zk = [];
    else %(motion starts)
        zk = zk(:,end-bin_group+1:end); % only keep the last 20 ms for the prediction 
        zk = (sum(zk,2)/bin_group); % firing rate 
       
    end
    
    %  Kalman filter initialization
    if isempty(zk)
        prior = zeros(nb_states); 
        xk = zeros(nb_states,1);
        xk(1:2) = test_data.startHandPos; % first estimate: actual initial position
        x = xk(1);
        y = xk(2);
        % Kk = prior*H'/(H*prior*H'+Q);
        Kk = prior*H' * pinv(H*prior*H' + Q + eps*eye(size(Q)));
        modelParameters.Kk = Kk;
        modelParameters.posterior = prior;
        modelParameters.decodedHandPos = xk;

    % Prediction for next time steps
    else
        xk_previous_estimate = modelParameters.decodedHandPos;
        posterior = modelParameters.posterior;
        prior = A*posterior*A'+W;

        % Kalman filter parameters update
        % Reference: W. Wu, M. Black, Y. Gao, E. Bienenstock, M. Serruya,
        % and J. Donoghue, "Inferring hand motion from multi-cell
        % recordings in motor cortex using a kalman filter" (2002)

        % Kk = prior*H'/(H*prior*H'+Q);
        Kk = prior*H' * pinv(H*prior*H' + Q + eps*eye(size(Q)));
        xk_estimate = A*xk_previous_estimate;
        xk = xk_estimate+Kk*(zk-H*xk_estimate);
        modelParameters.posterior = (I-Kk*H)*prior;
        modelParameters.decodedHandPos = xk;
        x = xk(1);
        y = xk(2);
    end
end