function modelParameters = positionEstimatorTraining(trial)
% positionEstimatorTraining_RLS  Train a PCA+RLS model for neural decoding
%
% Usage:
%   modelParameters = positionEstimatorTraining_RLS(trial)
%
% Steps:
%   1) Determine T_min (the shortest trial duration).
%   2) Flatten spike trains (X_data) and hand positions (Y_data).
%   3) Perform SVD-based PCA on X_data to reduce dimensionality.
%   4) Train a Recursive Least Squares (RLS) model to map from PCA-reduced
%      features to the flattened hand positions.
%   5) Store all parameters needed for decoding in modelParameters.

    % 1) Determine T_min across all trials/angles
    num_trials = size(trial,1);
    num_angles = size(trial,2);
    num_neurons = size(trial(1,1).spikes,1);

    T_min = inf;
    for angle = 1:num_angles
        for t = 1:num_trials
            T_min = min(T_min, size(trial(t, angle).spikes, 2));
        end
    end

    % 2) Build flattened X_data (spikes) and Y_data (positions)
    X_data = [];  % (#trials total) x (num_neurons * T_min)
    Y_data = [];  % (#trials total) x (2 * T_min)
    for angle = 1:num_angles
        for t = 1:num_trials
            % Extract & truncate
            spikes  = trial(t, angle).spikes(:, 1:T_min);    % 98 x T_min
            handPos = trial(t, angle).handPos(1:2, 1:T_min); % 2 x T_min

            % Flatten
            spike_vector = reshape(spikes, 1, []);         % 1 x (98*T_min)
            handPos_vec  = reshape(handPos, 1, []);        % 1 x (2*T_min)

            X_data = [X_data; spike_vector];
            Y_data = [Y_data; handPos_vec];
        end
    end

    % 3) PCA via SVD (same as your snippet)
    X_mean = mean(X_data, 1);
    X_norm = X_data - X_mean;  % center data
    [U, S, V] = svd(X_norm, 'econ');

    singular_values = diag(S);
    explained_variance = (singular_values.^2) / sum(singular_values.^2);
    cum_variance = cumsum(explained_variance);

    % Choose #PCs to retain 95% variance
    variance_threshold = 0.95;
    num_PC = find(cum_variance >= variance_threshold, 1);

    % Project onto principal components
    V_reduced = V(:, 1:num_PC);
    X_reduced = X_norm * V_reduced;  % (#trials) x (num_PC)

    % 4) Train RLS to map X_reduced -> Y_data
    %
    % We'll do a one-pass RLS, augmenting with a bias term. So each sample is:
    %    x_aug(i,:) = [X_reduced(i,:), 1]
    % and we want to predict Y_data(i,:) of size 1 x (2*T_min).
    %
    % RLS update:
    %   k     = P*x / (lambda + x'*P*x)
    %   err   = y - W'*x
    %   W     = W + k * err'
    %   P     = (P - k*x'*P)/lambda
    %
    % We'll set lambda=1 (no forgetting) or something close to 1 if you want adaptation.

    N = size(X_reduced,1);        % total #samples
    dim_input = num_PC + 1;       % +1 for bias
    dim_output = size(Y_data, 2); % 2*T_min

    lambda = 1.0;     % no forgetting (you can try ~0.99 for adaptive)
    delta  = 1e2;     % controls initial covariance scale

    % Initialize
    W = zeros(dim_input, dim_output);         % weight matrix
    P = (1/delta) * eye(dim_input);           % inverse covariance

    for i = 1:N
        x_aug = [X_reduced(i,:), 1]';  % (dim_input x 1)
        y_tar = Y_data(i,:)';         % (dim_output x 1)

        gain = P * x_aug / (lambda + x_aug' * P * x_aug);
        err  = y_tar - (W' * x_aug);
        W    = W + gain * err';
        P    = (P - gain * x_aug' * P) / lambda;
    end

    % 5) Store parameters in modelParameters
    modelParameters.T_min      = T_min;
    modelParameters.X_mean     = X_mean;     % mean of original data
    modelParameters.V_reduced  = V_reduced;  % PCA projection
    modelParameters.num_PC     = num_PC;
    modelParameters.W_RLS      = W;          % RLS weights
end
