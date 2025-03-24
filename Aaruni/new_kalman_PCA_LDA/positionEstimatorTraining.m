function modelParameters = positionEstimatorTraining(training_data)
% POSITIONESTIMATORTRAINING trains the decoder with enhanced features.
%
%   modelParameters = positionEstimatorTraining(training_data)
%
%   Pipeline changes:
%     - Binning with additional temporal history (current, t-20, t-40 ms)
%     - Low-pass filtering with updated alpha and Gaussian smoothing (sigma reduced)
%     - Feature normalization (zero mean, unit variance)
%     - PCA with increased dimensions (d = 15)
%     - Ridge regression with updated regularization (lambda = 0.1)
%     - Kalman filter with updated noise parameters for smoothing

% Parameters:
bin_window = 20;         % ms binning interval
history_steps = 3;       % use current time + 2 previous bins (t, t-20, t-40)
alpha = 0.45;            % updated low-pass filter parameter
sigma = 3;               % updated Gaussian smoothing sigma
kernel_radius = round(3*sigma);
gKernel = gaussianKernel(sigma, kernel_radius);

% Determine maximum trial length (for padding if needed)
maxT = 0;
[nTrials, nDirections] = size(training_data);
for n = 1:nTrials
    for k = 1:nDirections
        T = size(training_data(n,k).spikes,2);
        if T > maxT, maxT = T; end
    end
end

% Initialize matrices for training features and targets.
X_train = [];
Y_train = [];

% Define time window for training examples.
time_start = 320; % start decoding at 320 ms
time_step  = bin_window;  % use steps of 20 ms

for n = 1:nTrials
    for k = 1:nDirections
        trial_spikes  = training_data(n,k).spikes;   % (neurons x T)
        trial_handPos = training_data(n,k).handPos;    % (3 x T)
        [num_neurons, T] = size(trial_spikes);
        
        % Compute average starting hand position (first 0-300 ms)
        t_avg = min(300, T);
        avgStart = mean(trial_handPos(1:2,1:t_avg), 2);  % 2x1 vector
        
        % Preprocess each neuron's spike train.
        processedSpikes = zeros(num_neurons, T);
        for i = 1:num_neurons
            x = double(trial_spikes(i,:));
            y_lp = simpleLowPass(x, alpha);
            y_sm = conv(y_lp, gKernel, 'same');
            processedSpikes(i,:) = y_sm;
        end
        
        % Create training samples at discrete time points.
        for t = time_start:time_step:T
            % Build temporal features: current bin and previous two bins.
            feature_temp = [];
            for h = 0:(history_steps-1)
                t_idx = t - h*bin_window;
                if t_idx < 1
                    feature_temp = [feature_temp; zeros(num_neurons,1)];
                else
                    feature_temp = [feature_temp; processedSpikes(:, t_idx)];
                end
            end
            % Append the average starting hand position.
            feature = [feature_temp; avgStart]; % dimension: (num_neurons*history_steps + 2) x 1
            target  = trial_handPos(1:2, t);     % hand position (x,y)
            X_train = [X_train; feature'];
            Y_train = [Y_train; target'];
        end
    end
end

% --- Normalize features ---
meanX = mean(X_train, 1);
stdX = std(X_train, 0, 1);
X_norm = (X_train - meanX) ./ (stdX + eps);

% --- PCA for dimensionality reduction ---
C = (X_norm' * X_norm) / size(X_norm,1);
[V, D] = eig(C);
[eigvals, idx] = sort(diag(D), 'descend');
V = V(:, idx);
% Choose top d principal components (increased to 15).
d = 15;
V_reduced = V(:, 1:d);
X_reduced = X_norm * V_reduced;

% Add bias column.
X_design = [ones(size(X_reduced,1),1) X_reduced];

% --- Ridge Regression ---
lambda = 0.1; % updated regularization parameter
[n_samples, n_features] = size(X_design);
W = (X_design' * X_design + lambda * eye(n_features)) \ (X_design' * Y_train);

% --- Kalman Filter Setup ---
dt = bin_window/1000; % time step in seconds (20 ms)
A = [1 0 dt 0; 
     0 1 0 dt; 
     0 0 1 0; 
     0 0 0 1];
H = [1 0 0 0; 
     0 1 0 0];
Q = 0.005 * eye(4); % updated process noise covariance
R = 0.05  * eye(2); % updated measurement noise covariance

% Save all learned parameters.
modelParameters.meanX = meanX;
modelParameters.stdX = stdX;
modelParameters.V = V_reduced;
modelParameters.W = W;
modelParameters.lambda = lambda;
modelParameters.pca_dim = d;
modelParameters.alpha = alpha;
modelParameters.gKernel = gKernel;
modelParameters.bin_window = bin_window;
modelParameters.history_steps = history_steps;
modelParameters.numNeurons = num_neurons;
modelParameters.dt = dt;
modelParameters.A = A;
modelParameters.H = H;
modelParameters.Q = Q;
modelParameters.R = R;
modelParameters.kalmanInitialized = false;
modelParameters.kalmanState = [];
modelParameters.kalmanCov = [];

end

%% Helper Functions

function y = simpleLowPass(x, alpha)
% SIMPLELOWPASS applies a recursive low-pass filter.
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
