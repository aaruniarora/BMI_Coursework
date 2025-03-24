function modelParameters = positionEstimatorTraining(training_data)
% POSITIONESTIMATORTRAINING trains the decoder with extended features.
%
% Summary of methods:
%   1. Extended temporal history: features from t, t-20, t-40, t-60 ms.
%   2. Additional derivative feature: difference between current bin and previous bin.
%   3. Feature normalization and PCA (increased to 20 dimensions).
%   4. Ridge regression with a lower regularization (lambda = 0.05).
%   5. Kalman filter with reduced process and measurement noise.
%
%   modelParameters = positionEstimatorTraining(training_data)

% Parameters
bin_window = 10;           % ms binning interval
history_steps = 4;         % use current bin plus three previous bins: t, t-20, t-40, t-60 ms
alpha = 0.5;               % updated low-pass filter parameter
sigma = 3;                 % Gaussian smoothing sigma remains 3
kernel_radius = round(3*sigma);
gKernel = gaussianKernel(sigma, kernel_radius);

% Determine maximum trial length (if padding needed)
maxT = 0;
[nTrials, nDirections] = size(training_data);
for n = 1:nTrials
    for k = 1:nDirections
        T = size(training_data(n,k).spikes,2);
        if T > maxT, maxT = T; end
    end
end

% Initialize training matrices.
X_train = [];
Y_train = [];

% Time window for training examples.
time_start = 320;  % start at 320 ms
time_step  = bin_window;  % 20 ms increments

for n = 1:nTrials
    for k = 1:nDirections
        trial_spikes  = training_data(n,k).spikes;   % (neurons x T)
        trial_handPos = training_data(n,k).handPos;    % (3 x T)
        [num_neurons, T] = size(trial_spikes);
        
        % Compute average starting hand position (0–300 ms)
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
            % Collect extended history features from t, t-20, t-40, t-60.
            feat_history = [];
            for h = 0:(history_steps-1)
                t_idx = t - h*bin_window;
                if t_idx < 1
                    feat_history = [feat_history; zeros(num_neurons,1)];
                else
                    feat_history = [feat_history; processedSpikes(:, t_idx)];
                end
            end
            % Compute derivative for the current bin (difference with previous bin).
            if t - bin_window < 1
                deriv = zeros(num_neurons,1);
            else
                deriv = processedSpikes(:, t) - processedSpikes(:, t - bin_window);
            end
            
            % Concatenate history, derivative, and average starting hand position.
            feature = [feat_history; deriv; avgStart];  
            % Dimensions: (num_neurons*history_steps + num_neurons + 2) x 1
            
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
% Increase PCA dimensions to 20.
d = 20;
V_reduced = V(:, 1:d);
X_reduced = X_norm * V_reduced;

% Add bias (intercept) column.
X_design = [ones(size(X_reduced,1),1) X_reduced];

% --- Ridge Regression ---
lambda = 0.05; % reduced regularization parameter
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
Q = 0.003 * eye(4); % further reduced process noise covariance
R = 0.03  * eye(2); % further reduced measurement noise covariance

% Save learned parameters.
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
