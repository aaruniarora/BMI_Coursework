function modelParameters = positionEstimatorTraining(training_data)
% POSITIONESTIMATORTRAINING trains the decoder.
%
%   modelParameters = positionEstimatorTraining(training_data)
%
%   Input:
%     training_data - a [nTrials x nDirections] structure array where each
%                     element contains:
%                        .trialId, .spikes (neurons x T) and .handPos (3 x T)
%
%   The pipeline:
%     1. Bins the spikes (here we sample every 20 ms, starting at 320 ms).
%     2. Applies a simple recursive low-pass filter (cutoff ~100 Hz).
%     3. Smooths with a Gaussian kernel.
%     4. Pads trials to a common maximum length (implicitly by using the
%        sampled time index in each trial).
%     5. Computes the average starting hand position (0–300 ms) per trial.
%     6. Collects a feature vector from each sample: [neural features; avg start].
%     7. Performs PCA on the design matrix and reduces its dimensionality.
%     8. Adds a bias (ones) column.
%     9. Uses ridge regression to compute weights mapping features to (x,y).
%    10. Sets up a simple Kalman filter (constant-velocity model) for smoothing.
%
%   Output:
%     modelParameters - a structure containing the learned regression weights,
%                       PCA parameters, Kalman filter parameters, and others.
  
% Determine maximum trial length (for potential padding)
maxT = 0;
[nTrials, nDirections] = size(training_data);
for n = 1:nTrials
    for k = 1:nDirections
        T = size(training_data(n,k).spikes,2);
        if T > maxT
            maxT = T;
        end
    end
end

% Initialize matrices for training features and targets.
X_train = [];
Y_train = [];

% Preprocessing parameters
alpha = 0.385;        % low-pass filter parameter (approx. for 100 Hz cutoff with 1 ms dt)
sigma = 5;            % standard deviation (ms) for Gaussian smoothing
kernel_radius = round(3*sigma);
gKernel = gaussianKernel(sigma, kernel_radius);

% Define the time window for creating training examples.
time_start = 320; % start decoding at 320 ms
time_step  = 10;  % use steps of 20 ms

% Loop over every trial and reaching angle.
for n = 1:nTrials
    for k = 1:nDirections
        trial_spikes  = training_data(n,k).spikes;   % (neurons x T)
        trial_handPos = training_data(n,k).handPos;    % (3 x T), use first two rows
        [num_neurons, T] = size(trial_spikes);
        
        % Compute average starting hand position (first 0-300 ms)
        t_avg = min(300, T);
        avgStart = mean(trial_handPos(1:2,1:t_avg), 2);  % 2x1 vector
        
        % Preprocess each neuron’s spike train.
        processedSpikes = zeros(num_neurons, T);
        for i = 1:num_neurons
            x = double(trial_spikes(i,:));            % convert to double
            y_lp = simpleLowPass(x, alpha);             % low-pass filtering
            y_sm = conv(y_lp, gKernel, 'same');          % Gaussian smoothing
            processedSpikes(i,:) = y_sm;
        end
        
        % Create training samples at discrete time points.
        for t = time_start:time_step:T
            % Feature vector: neural firing rates at time t, concatenated with
            % the average starting hand position.
            feature = [processedSpikes(:, t); avgStart]; % (num_neurons+2 x 1)
            target  = trial_handPos(1:2, t);              % hand position (x,y)
            X_train = [X_train; feature'];
            Y_train = [Y_train; target'];
        end
    end
end

% --- PCA for dimensionality reduction ---
% Center the data.
meanX = mean(X_train,1);
X_centered = X_train - meanX;
% Compute covariance matrix and its eigen decomposition.
C = (X_centered' * X_centered) / size(X_centered,1);
[V, D] = eig(C);
% Sort eigenvalues in descending order.
[eigvals, idx] = sort(diag(D), 'descend');
V = V(:, idx);
% Choose the top d principal components (e.g., d = 10).
d = 10;
V_reduced = V(:, 1:d);
X_reduced = X_centered * V_reduced;

% Add a bias (intercept) column.
X_design = [ones(size(X_reduced,1),1) X_reduced];

% --- Ridge Regression ---
lambda = 1; % regularization parameter
[n_samples, n_features] = size(X_design);
W = (X_design' * X_design + lambda * eye(n_features)) \ (X_design' * Y_train);

% --- Kalman Filter Setup ---
% Here we use a simple constant-velocity model.
dt = 0.02; % time step (20 ms in seconds)
A = [1 0 dt 0; 
     0 1 0 dt; 
     0 0 1 0; 
     0 0 0 1];
H = [1 0 0 0; 
     0 1 0 0];
Q = 0.01 * eye(4); % process noise covariance
R = 0.1  * eye(2); % measurement noise covariance

% (A branch for LDA could be inserted here if you wish to discriminate reaching angle.)

% Save all learned parameters.
modelParameters.meanX = meanX;
modelParameters.V = V_reduced;
modelParameters.W = W;
modelParameters.lambda = lambda;
modelParameters.pca_dim = d;
modelParameters.alpha = alpha;
modelParameters.gKernel = gKernel;
modelParameters.numNeurons = size(training_data(1,1).spikes,1);
modelParameters.dt = dt;
modelParameters.A = A;
modelParameters.H = H;
modelParameters.Q = Q;
modelParameters.R = R;
modelParameters.kalmanInitialized = false;  % flag for Kalman initialization
modelParameters.kalmanState = [];
modelParameters.kalmanCov = [];
  
end

%% Helper Functions

function y = simpleLowPass(x, alpha)
% SIMPLELOWPASS implements a simple recursive low-pass filter.
y = zeros(size(x));
y(1) = alpha * x(1);
for n = 2:length(x)
    y(n) = alpha * x(n) + (1 - alpha) * y(n-1);
end
end

function g = gaussianKernel(sigma, radius)
% GAUSSIANKERNEL creates a normalized Gaussian kernel.
t = -radius:radius;
g = exp(-t.^2/(2*sigma^2));
g = g / sum(g);
end
