function [x, y, modelParameters] = positionEstimator(past_current_trial, modelParameters)
% positionEstimator   Apply a Kalman filter in PCA space to decode hand position.
%
% Usage:
%   [x, y, modelParameters] = positionEstimator(past_current_trial, modelParameters)
%
% Steps:
%   1) Preprocess test data (bin, sqrt, smooth).
%   2) Remove low-firing neurons (same as training).
%   3) Project neural data into PCA space.
%   4) Kalman predict-update for each time bin.
%   5) Output final (x, y).

    KF = modelParameters.kalman;

    % Retrieve the stored parameters
    A  = KF.A;
    C  = KF.C;
    Q  = KF.Q;
    R  = KF.R;
    x_hat = KF.x0;   % initial state
    P     = KF.P0;   % initial covariance

    lowFirers = KF.lowFirers;
    mu_pca    = KF.PCAmean;   % M x 1
    W_pca     = KF.PCAcoeff;  % M x K
    K_dim     = KF.K;         % # of principal components

    group = 20;  % must match training
    win   = 50;

    % -------------------------------------------------------------------------
    % 1) Preprocess the incoming single trial
    % -------------------------------------------------------------------------
    trialProcess = bin_and_sqrt(past_current_trial, group, 1);
    trialFinal   = get_firing_rates(trialProcess, group, win);

    % rates: (#neurons x #bins)
    rates = trialFinal.rates;
    % 2) Remove the same low-firing neurons
    rates(lowFirers, :) = [];

    T_end = size(rates, 2);
    regFactor = 1e-5;  % small diagonal for numerical stability

    % -------------------------------------------------------------------------
    % 3) Run Kalman recursion
    % -------------------------------------------------------------------------
    for t = 1:T_end
        % observation in original (M) space
        z_raw = rates(:, t);

        % Project to PCA space
        z_centered = z_raw - mu_pca;       % M x 1
        z_pca      = W_pca' * z_centered;  % (K x M) * (M x 1) => K x 1

        % (a) Predict
        x_pred = A * x_hat;
        P_pred = A * P * A' + Q;

        % (b) Compute Kalman gain
        S = C * P_pred * C' + R + regFactor*eye(K_dim);
        K_t = P_pred * C' / S;

        % (c) Update
        x_hat = x_pred + K_t*(z_pca - C*x_pred);
        P = (eye(size(P_pred)) - K_t * C)*P_pred;
    end

    % Final position is (x,y)
    x = x_hat(1);
    y = x_hat(2);

    % Optionally store new state if you call this repeatedly in streaming
    modelParameters.kalman.x0 = x_hat;
    modelParameters.kalman.P0 = P;
end
