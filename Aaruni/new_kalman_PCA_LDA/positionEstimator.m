function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% POSITIONESTIMATOR predicts the current hand position using enhanced features.
%
%   [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%
%   This function applies the same preprocessing as training:
%     - Low-pass filtering and Gaussian smoothing
%     - Temporal feature extraction (current and previous bins)
%     - Feature normalization and PCA projection
%     - Ridge regression prediction followed by a Kalman filter update

trial_spikes = test_data.spikes;
[num_neurons, T] = size(trial_spikes);

if num_neurons ~= modelParameters.numNeurons
    error('Mismatch in number of neurons between test and training data.');
end

% Process spikes with low-pass filter and Gaussian smoothing.
processedSpikes = zeros(num_neurons, T);
for i = 1:num_neurons
    x_spike = double(trial_spikes(i,:));
    y_lp = simpleLowPass(x_spike, modelParameters.alpha);
    y_sm = conv(y_lp, modelParameters.gKernel, 'same');
    processedSpikes(i,:) = y_sm;
end

% Use current time T for prediction.
bin_window = modelParameters.bin_window;
history_steps = modelParameters.history_steps;
feature_temp = [];
for h = 0:(history_steps-1)
    t_idx = T - h*bin_window;
    if t_idx < 1
        feature_temp = [feature_temp; zeros(num_neurons,1)];
    else
        feature_temp = [feature_temp; processedSpikes(:, t_idx)];
    end
end

% Construct the feature vector and append starting hand position.
feature = [feature_temp; test_data.startHandPos];  % (numNeurons*history_steps+2 x 1)

% Normalize using training mean and std.
feature_norm = (feature' - modelParameters.meanX) ./ (modelParameters.stdX + eps);

% Project features using PCA.
feature_reduced = feature_norm * modelParameters.V;

% Add bias term.
feature_design = [1, feature_reduced];

% Predict hand position via ridge regression.
prediction = feature_design * modelParameters.W;  % yields 1x2 vector

% --- Kalman Filter Update ---
if ~modelParameters.kalmanInitialized
    % Initialize state: [x; y; vx; vy] using starting hand position.
    state = [test_data.startHandPos; 0; 0];
    P = eye(4);
    modelParameters.kalmanInitialized = true;
else
    state = modelParameters.kalmanState;
    P = modelParameters.kalmanCov;
end

A = modelParameters.A;
Q = modelParameters.Q;
state_pred = A * state;
P_pred = A * P * A' + Q;

H = modelParameters.H;
R = modelParameters.R;
z = prediction';  % measurement as a column vector
K = P_pred * H' / (H * P_pred * H' + R);
state_new = state_pred + K * (z - H * state_pred);
P_new = (eye(4) - K * H) * P_pred;

pos_est = state_new(1:2);
x = pos_est(1);
y = pos_est(2);

% Update Kalman filter state.
modelParameters.kalmanState = state_new;
modelParameters.kalmanCov = P_new;
newModelParameters = modelParameters;

end

%% Helper Function

function y = simpleLowPass(x, alpha)
% SIMPLELOWPASS applies a recursive low-pass filter.
y = zeros(size(x));
y(1) = alpha * x(1);
for n = 2:length(x)
    y(n) = alpha * x(n) + (1 - alpha) * y(n-1);
end
end
