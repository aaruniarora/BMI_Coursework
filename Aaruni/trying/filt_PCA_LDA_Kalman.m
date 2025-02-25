clc; clear; close all;
% training_data = load('monkeydata_training.mat');
% positionEstimatorTraining(training_data)

%%
function [modelParameters] = positionEstimatorTraining(training_data)
% positionEstimatorTraining trains a neural decoder using a combination of:
%   - Bandpass filtering (0.5–100 Hz) and Savitzky–Golay smoothing to preprocess spike trains.
%   - PCA for dimensionality reduction.
%   - LDA for discriminative feature extraction (using reaching angle labels).
%   - Kalman filter training with state defined as [pos_x; pos_y; vel_x; vel_y].
%
% INPUT:
%   training_data - a 100 x 8 struct array. Each element contains:
%         .trialId, .spikes (98 x T matrix), .handPos (3 x T matrix)
%
% OUTPUT:
%   modelParameters - structure containing all parameters for decoding.
%
% References:
%   - Bandpass filtering: Oppenheim & Schafer, Discrete-Time Signal Processing, 2009.
%   - PCA: Bishop, Pattern Recognition and Machine Learning, 2006.
%   - LDA: Duda, Hart & Stork, Pattern Classification, 2001.
%   - Kalman filtering: Welch & Bishop, An Introduction to the Kalman Filter, 1995.
%   - Savitzky–Golay: Savitzky & Golay, 1964.

dt = 0.02;  % update interval for Kalman filter (20 ms chunks)
fs = 1000;  % sampling frequency (1 ms bins)

%% 1. Define filter parameters
lowCut = 0.5;
highCut = 100;
[b_bp, a_bp] = butter(2, [lowCut/(fs/2), highCut/(fs/2)], 'bandpass');  % 2nd order Butterworth

% Savitzky–Golay parameters (window length must be odd)
sgWindow = 11;
sgOrder  = 3;

%% 2. Initialize accumulators for state and neural data
% X_all: state vector over time from all trials: [pos_x; pos_y; vel_x; vel_y]
% Y_all: neural features for every sample (raw: 98-d, before PCA/LDA)
X_all = [];
Y_all = [];
labels_all = [];  % corresponding reaching angle label for each sample

training_data = training_data.trial;
[numTrials, numAngles] = size(training_data);

for tr = 1:numTrials
    for ang = 1:numAngles
        trialStruct = training_data(tr, ang);
        spikes = trialStruct.spikes;    % 98 x T
        handPos = trialStruct.handPos;    % 3 x T (use first 2 for x,y)
        T = size(spikes, 2);
        
        % Preprocess: Filter and smooth each neuron's spike train.
        filteredSpikes = zeros(size(spikes));
        for neuron = 1:size(spikes, 1)
            spikeTrain = double(spikes(neuron, :));
            % Zero-phase bandpass filtering
            spikeBP = filtfilt(b_bp, a_bp, spikeTrain);
            % Savitzky–Golay smoothing (sgolayfilt is a built-in MATLAB function)
            spikeSmooth = sgolayfilt(spikeBP, sgOrder, sgWindow);
            filteredSpikes(neuron, :) = spikeSmooth;
        end
        
        % Compute state vector over time.
        % We'll use finite differences to estimate velocity (dt_sample = 0.001 sec).
        X_trial = zeros(4, T-1);
        for t = 2:T
            pos = handPos(1:2, t);
            prevPos = handPos(1:2, t-1);
            vel = (pos - prevPos) / 0.001;  % velocity in [units/sec]
            X_trial(:, t-1) = [pos; vel];
        end
        
        % Neural features: Use the filtered spike data at time steps 2:T.
        Y_trial = filteredSpikes(:, 2:T);  % 98 x (T-1)
        
        % Create label vector: reaching angle (from 1 to numAngles)
        labels_trial = ang * ones(1, T-1);
        
        % Append this trial's data to the overall dataset.
        X_all = [X_all, X_trial];
        Y_all = [Y_all, Y_trial];
        labels_all = [labels_all, labels_trial];
    end
end

%% 3. Dimensionality Reduction using PCA
% Choose a reduced dimension (e.g., r = 10)
r = 10;
[coeff, ~, ~] = pca(Y_all');  % note: pca expects observations in rows
U = coeff(:, 1:r);  % PCA projection matrix (98 x r)
Y_reduced = U' * Y_all;  % now Y_reduced is r x total_samples

%% 4. Apply LDA for Discrimination
% LDA can further reduce dimensionality up to (numClasses-1); here, max 7 dimensions.
classes = unique(labels_all);
numClasses = length(classes);
% Compute overall mean of Y_reduced
mu_overall = mean(Y_reduced, 2);

Sw = zeros(r, r);  % within-class scatter
Sb = zeros(r, r);  % between-class scatter

for i = 1:numClasses
    idx = (labels_all == classes(i));
    Yi = Y_reduced(:, idx);  % data for class i
    mu_class = mean(Yi, 2);
    % Within-class scatter: sum of covariances
    % Use (n-1) normalization by cov (or simply accumulate differences)
    Sw = Sw + cov(Yi');
    % Between-class scatter:
    ni = size(Yi, 2);
    diff = mu_class - mu_overall;
    Sb = Sb + ni * (diff * diff');
end

% Solve generalized eigenvalue problem: Sb*w = lambda*Sw*w
[W_lda, eigenvals] = eig(Sb, Sw);
% Sort eigenvectors by eigenvalue in descending order
[~, sortIdx] = sort(diag(eigenvals), 'descend');
W_lda = W_lda(:, sortIdx);
% Choose the top LDA dimensions; maximum is numClasses-1
ldaDim = min(numClasses - 1, r);  
W_lda = W_lda(:, 1:ldaDim);

% Project the PCA-reduced data further using LDA.
Y_final = W_lda' * Y_reduced;  % final neural features (dimension: ldaDim x total_samples)

%% 5. Kalman Filter Training (Linear Regression)
% We now want to learn an observation matrix C such that:
%    Y_final ≈ C * X_all.
C = Y_final * X_all' / (X_all * X_all');

% Estimate measurement noise covariance R:
residuals = Y_final - C * X_all;
R_cov = cov(residuals');  % dimensions: ldaDim x ldaDim

% Estimate process noise covariance Q using state dynamics:
% Let X_all be our state sequence; approximate Q from state prediction errors.
X_next = X_all(:, 2:end);
X_prev = X_all(:, 1:end-1);
% Define state transition matrix A for a constant-velocity model:
A = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];
X_predicted = A * X_prev;
Q_cov = cov((X_next - X_predicted)');

% Set an initial state and covariance (from first sample)
initialState = X_all(:, 1);
initialP = eye(4);

%% 6. Package all parameters into modelParameters structure.
modelParameters.A = A;
modelParameters.C = C;
modelParameters.Q = Q_cov;
modelParameters.R = R_cov;
modelParameters.U = U;            % PCA projection matrix (98 x r)
modelParameters.W_lda = W_lda;      % LDA projection matrix (r x ldaDim)
modelParameters.b_bp = b_bp;
modelParameters.a_bp = a_bp;
modelParameters.sgOrder = sgOrder;
modelParameters.sgWindow = sgWindow;
modelParameters.dt = dt;
modelParameters.initialState = initialState;
modelParameters.initialP = initialP;

% Also store dimensions for use in the estimator.
modelParameters.pcaDim = r;
modelParameters.ldaDim = ldaDim;

end
