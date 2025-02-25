function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
% POSITIONESTIMATOR returns the predicted hand position (x,y) at the current time.
%
%   [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
%
%   Input:
%     test_data - a structure with fields:
%                   .trialId, .spikes (neurons x current T), .handPos, 
%                   .decodedHandPos, and .startHandPos (the initial hand position)
%     modelParameters - the structure returned by positionEstimatorTraining.
%
%   The function:
%     1. Processes the current spike train with the same low-pass filter and
%        Gaussian smoothing as in training.
%     2. Extracts a feature vector from the last time point: the neural value
%        at time T (for each neuron) and the average starting hand position.
%     3. Applies the PCA transformation (using training mean and eigenvectors)
%        and then adds a bias column.
%     4. Uses the learned ridge regression weights to predict the hand position.
%     5. Feeds the prediction into a simple Kalman filter to obtain a smooth estimate.
%
%   Output:
%     x, y - predicted hand position coordinates.
%     newModelParameters - updated modelParameters (with updated Kalman state).

trial_spikes = test_data.spikes;
[num_neurons, T] = size(trial_spikes);

% Verify that the number of neurons matches the training data.
if num_neurons ~= modelParameters.numNeurons
    error('Mismatch in number of neurons between test and training data.');
end

% Process each neuron’s spike train using the same preprocessing.
processedSpikes = zeros(num_neurons, T);
for i = 1:num_neurons
    x_spike = double(trial_spikes(i,:));
    y_lp = simpleLowPass(x_spike, modelParameters.alpha);
    y_sm = conv(y_lp, modelParameters.gKernel, 'same');
    processedSpikes(i,:) = y_sm;
end

% Use the last time point (T) for this prediction.
% Feature vector: [neural firing rates at time T; starting hand position].
feature = [processedSpikes(:, T); test_data.startHandPos];  % (num_neurons+2 x 1)

% Apply the PCA transformation: center the feature and project.
feature_centered = feature' - modelParameters.meanX;
feature_reduced = feature_centered * modelParameters.V;
  
% Add a bias term.
feature_design = [1, feature_reduced];

% Predict hand position via ridge regression.
prediction = feature_design * modelParameters.W;  % yields a 1x2 vector

% --- Kalman Filter Update ---
% Initialize the Kalman filter if not already done.
if ~modelParameters.kalmanInitialized
    % Initialize state: [x; y; vx; vy] with position = starting hand position, velocity = 0.
    state = [test_data.startHandPos; 0; 0];
    P = eye(4);
    modelParameters.kalmanInitialized = true;
else
    state = modelParameters.kalmanState;
    P = modelParameters.kalmanCov;
end

% Predict step.
A = modelParameters.A;
Q = modelParameters.Q;
state_pred = A * state;
P_pred = A * P * A' + Q;

% Measurement update.
H = modelParameters.H;
R = modelParameters.R;
z = prediction';  % convert to column vector (2x1)
K = P_pred * H' / (H * P_pred * H' + R);
state_new = state_pred + K * (z - H * state_pred);
P_new = (eye(4) - K * H) * P_pred;

% The filtered (smoothed) hand position is the first two state components.
pos_est = state_new(1:2);
x = pos_est(1);
y = pos_est(2);

% Update the Kalman filter state in the modelParameters.
modelParameters.kalmanState = state_new;
modelParameters.kalmanCov = P_new;
newModelParameters = modelParameters;

end

%% Helper function (same as in training)
function y = simpleLowPass(x, alpha)
y = zeros(size(x));
y(1) = alpha * x(1);
for n = 2:length(x)
    y(n) = alpha * x(n) + (1 - alpha) * y(n-1);
end
end
