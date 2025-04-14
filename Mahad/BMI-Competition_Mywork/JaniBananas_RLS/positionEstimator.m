function [x, y, modelParameters] = positionEstimator(test_trial, modelParameters)
% positionEstimator_RLS  Use the trained PCA+RLS model to decode X-Y positions
%                        for a single trial.
%
% Usage:
%   [x, y, modelParameters] = positionEstimator_RLS(test_trial, modelParameters)
%
% Inputs:
%   - test_trial: A struct with field .spikes (num_neurons x T)
%   - modelParameters: The structure returned by positionEstimatorTraining_RLS.
%
% Outputs:
%   - x, y: Each is 1 x T_min containing the decoded hand trajectory (T_min is the
%           minimum trial length used in training).
%   - modelParameters: Optionally updated if you wish to implement adaptive updates.

    % Retrieve training parameters
    T_min      = modelParameters.T_min;     % required trial length
    X_mean     = modelParameters.X_mean;      % 1 x (num_neurons*T_min)
    V_reduced  = modelParameters.V_reduced;   % PCA projection matrix
    W_RLS      = modelParameters.W_RLS;       % RLS weight matrix

    % 1) Get spikes from test trial
    spikes = test_trial.spikes; % size: (num_neurons x T_test)
    [num_neurons, T_test] = size(spikes);
    
    % If the test trial is shorter than T_min, pad it with the last column.
    if T_test < T_min
        pad_length = T_min - T_test;
        pad_spikes = repmat(spikes(:,T_test), 1, pad_length);
        spikes = [spikes, pad_spikes];
    elseif T_test > T_min
        % Truncate if the trial is longer than T_min.
        spikes = spikes(:,1:T_min);
    end
    % Now spikes is (num_neurons x T_min)
    
    % 2) Flatten spikes into a single row vector
    spike_vector = reshape(spikes, 1, []); % size: 1 x (num_neurons*T_min)
    
    % 3) Project onto PCA space (center using training X_mean)
    X_norm = spike_vector - X_mean;  % should now be compatible in size
    X_proj = X_norm * V_reduced;       % 1 x num_PC
    
    % 4) Augment with bias and predict via RLS
    x_aug = [X_proj, 1];               % 1 x (num_PC+1)
    Y_pred = x_aug * W_RLS;            % 1 x (2*T_min)
    
    % 5) Reshape predictions to 2 x T_min
    Y_pred_reshaped = reshape(Y_pred, 2, T_min);
    x = Y_pred_reshaped(1, :);
    y = Y_pred_reshaped(2, :);
    
    % (Optional) Update modelParameters if doing online adaptation
end
