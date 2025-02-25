clc; close all; clear;

%% 1. Load Training Data and Extract Features
% Assumes "monkeydata_training.mat" is in the working directory.
load('monkeydata_training.mat');  
% 'trial' is a 100x8 struct array.
[nTrials, nAngles] = size(trial);

allFeatures = [];   % Each row will be a feature vector (1x98) per trial.
allLabels   = [];   % Reaching angle label for each trial (1 to 8).
allHandTraj = {};   % Each cell will hold a 2 x T matrix (x and y hand positions).

for ang = 1:nAngles
    for tr = 1:nTrials
        % Extract spike data: 98 x T binary matrix for current trial.
        spikes = trial(tr, ang).spikes;
        % Compute firing rate per neuron (spike count normalized by trial duration).
        firingRate = sum(spikes, 2) / size(spikes,2);
        allFeatures = [allFeatures; firingRate'];
        allLabels   = [allLabels; ang];
        % Extract hand position (3 x T), keep only x and y coordinates.
        handPos = trial(tr, ang).handPos;
        allHandTraj{end+1} = handPos(1:2, :);
    end
end

%% 2. Data Truncation: Truncate All Hand Trajectories to a Common Length
% Find the minimum trajectory length among all trials.
minT = inf;
for i = 1:length(allHandTraj)
    currentT = size(allHandTraj{i}, 2);
    if currentT < minT
        minT = currentT;
    end
end

% Truncate every trajectory to the first minT time points.
for i = 1:length(allHandTraj)
    allHandTraj{i} = allHandTraj{i}(:, 1:minT);
end

%% 3. Dimensionality Reduction via PCA (using SVD)
% Center the feature matrix.
X = allFeatures;  % Dimensions: (numTrials x 98)
meanX = mean(X, 1);
X_centered = X - repmat(meanX, size(X,1), 1);

% Compute SVD on the centered data.
[U, S, V] = svd(X_centered, 'econ');

% Choose number of principal components (e.g., 3).
nComponents = 3;
% Project data onto the top nComponents.
featuresPCA = X_centered * V(:,1:nComponents);
% featuresPCA is now (totalTrials x nComponents).

%% 4. Manual LDA Implementation
% Our goal is to compute the discriminant function for each class.
% For each class k, we calculate the mean vector mu_k and use the pooled covariance.
K = nAngles;   % Number of classes (8 reaching angles)
n = size(featuresPCA, 1);
d = nComponents;  % Dimensionality after PCA

classMeans = zeros(K, d);
classCounts = zeros(K, 1);
Sigma = zeros(d, d);

for k = 1:K
    idx = find(allLabels == k);
    Xk = featuresPCA(idx, :);  % Data for class k.
    nk = size(Xk, 1);
    classCounts(k) = nk;
    mu_k = mean(Xk, 1);
    classMeans(k, :) = mu_k;
    
    % Compute class covariance (unbiased estimate).
    if nk > 1
        Xk_centered = Xk - repmat(mu_k, nk, 1);
        cov_k = (Xk_centered' * Xk_centered) / (nk - 1);
    else
        cov_k = zeros(d,d);
    end
    Sigma = Sigma + (nk - 1) * cov_k;
end

% Compute the pooled covariance matrix.
Sigma = Sigma / (n - K);

% Class priors.
priors = classCounts / n;

% Precompute the inverse of Sigma.
invSigma = inv(Sigma);

% Classify each trial using the LDA discriminant function.
predictedLabels = zeros(n, 1);
for i = 1:n
    x = featuresPCA(i,:)';  % Column vector.
    maxScore = -inf;
    label = 1;
    for k = 1:K
        mu_k = classMeans(k,:)';
        % Discriminant function:
        score = x' * invSigma * mu_k - 0.5 * (mu_k' * invSigma * mu_k) + log(priors(k));
        if score > maxScore
            maxScore = score;
            label = k;
        end
    end
    predictedLabels(i) = label;
end

%% 5. Compute Average Hand Trajectory for Each Reaching Angle
% For each reaching angle, average the (truncated) x and y trajectories over all trials.
T = minT;  % Common trajectory length.
avgTraj = cell(K, 1);
for ang = 1:K
    idx = find(allLabels == ang);
    nTrials_ang = length(idx);
    traj_x = zeros(nTrials_ang, T);
    traj_y = zeros(nTrials_ang, T);
    for i = 1:nTrials_ang
        traj = allHandTraj{idx(i)};  % 2 x T matrix.
        traj_x(i, :) = traj(1, :);
        traj_y(i, :) = traj(2, :);
    end
    avgTraj{ang} = [mean(traj_x, 1); mean(traj_y, 1)];  % 2 x T average trajectory.
end

%% 6. Assign Predicted Trajectories to Each Trial
% Each trial's predicted trajectory is the average trajectory corresponding to its LDA-predicted reaching angle.
predictedTraj = cell(n, 1);
for i = 1:n
    ang = predictedLabels(i);
    predictedTraj{i} = avgTraj{ang};
end

%% 7. Compute Overall RMSE (in mm and cm)
% For each trial, compute the difference between the actual and predicted trajectory.
% The error is computed over both x and y dimensions and averaged over time and trials.
allErrors = [];
for i = 1:n
    diffTraj = allHandTraj{i} - predictedTraj{i};  % 2 x T difference matrix.
    allErrors = [allErrors; diffTraj(:)];  % Stack as a column vector.
end
rmse_mm = sqrt(mean(allErrors.^2));  % RMSE in mm.
rmse_cm = rmse_mm / 10;              % Convert mm to cm.
fprintf('Overall RMSE: %.2f mm (%.2f cm)\n', rmse_mm, rmse_cm);

%% 8. Visualize Actual vs. Predicted Trajectories for All Trials and All Reaching Angles
% % Create one figure with 8 subplots (one per reaching angle).
% figure; hold on;
% for ang = 1:K
%     % subplot(2, 4, ang);
%     idx = find(allLabels == ang);
%     hold on;
%     % Loop over all trials for this reaching angle.
%     for j = 1:length(idx)
%         actual = allHandTraj{idx(j)};
%         predicted = predictedTraj{idx(j)};
%         % Plot actual trajectory (blue solid line).
%         plot(actual(1,:), actual(2,:), 'b-', 'LineWidth', 1, 'DisplayName', 'Actual');
%         % Plot predicted trajectory (red dashed line).
%         plot(predicted(1,:), predicted(2,:), 'r--', 'LineWidth', 1, 'DisplayName', 'Predicted');
%     end
%     % title(sprintf('Angle %d', ang));
%     title('LDA PCA')
%     xlabel('X Position (mm)');
%     ylabel('Y Position (mm)');
%     hold off;
% end
% legend ({'Actual', 'Predicted'}) ;

%% 8. Visualize Actual vs. Predicted Trajectories for All Trials (Single Plot)
figure; hold on;

% Create two "placeholder" lines so the legend shows only two entries.
p1 = plot(NaN, NaN, 'b', 'LineWidth', 1.5);
p2 = plot(NaN, NaN, 'r', 'LineWidth', 1.5);

% Now plot all actual and predicted trajectories
for i = 1:n
    actual = allHandTraj{i};
    predicted = predictedTraj{i};
    plot(actual(1,:), actual(2,:), 'b', 'LineWidth', 1);
    plot(predicted(1,:), predicted(2,:), 'r', 'LineWidth', 1);
end

% Axis labels and legend
xlabel('x direction (mm)');
ylabel('y direction (mm)');
title('All Trials: Actual vs. Decoded Position');
legend([p1 p2], {'Actual Position', 'Decoded Position'});
grid on;

%%
% clear; close all; clc;
% 
% %% 1. Load Data
% load('monkeydata_training.mat'); 
% % 'trial' is a 100x8 struct array:
% %   trial(n, k).spikes   -> 98 x T (binary)
% %   trial(n, k).handPos -> 3 x T  (we'll use only x & y, i.e. rows 1 & 2)
% 
% [nTrials, nAngles] = size(trial);
% 
% %% 2. Build a Single Design Matrix for All Trials, All Time Bins
% % We'll gather:
% %   X: (totalTimeBins x 98) = each row is one time bin from one trial
% %   Y: (totalTimeBins x 2)  = each row is the (x,y) position for that time bin
% 
% X = [];  % neural data across all trials/time bins
% Y = [];  % position data across all trials/time bins
% 
% % We'll also keep track of each (trial, angle)'s length T so we can reconstruct later.
% trialLengths = zeros(nTrials, nAngles);
% 
% for ang = 1:nAngles
%     for tr = 1:nTrials
%         spk = trial(tr, ang).spikes;    % 98 x T
%         pos = trial(tr, ang).handPos;   % 3 x T (we only use rows 1:2)
%         T   = size(spk, 2);
% 
%         % Append to X and Y
%         %   spk' => T x 98  (transpose so each row is a time bin)
%         %   pos(1:2,:)' => T x 2
%         X = [X; spk'];
%         Y = [Y; pos(1:2, :)'];
% 
%         trialLengths(tr, ang) = T;
%     end
% end
% 
% %% 3. Fit a Linear Regression Decoder
% % We'll do Y = X*W + b by augmenting X with a column of ones for the intercept.
% X_design = [X, ones(size(X,1), 1)];  % (N x 99), if N = totalTimeBins across all trials
% % Solve for W in a least-squares sense: (X'X)^(-1) X'Y
% % W will be a (99 x 2) matrix (98 + 1 intercept) -> 2 outputs (x,y).
% W = (X_design' * X_design) \ (X_design' * Y);
% 
% %% 4. Predict Positions for All Time Bins
% Y_pred = X_design * W;  % (N x 2)
% 
% %% 5. Compute Overall RMSE
% diffPos = Y_pred - Y;               % (N x 2)
% rmse_mm = sqrt(mean(diffPos(:).^2));% overall RMSE in mm
% rmse_cm = rmse_mm / 10;             % convert to cm
% fprintf('Overall RMSE = %.2f mm (%.2f cm)\n', rmse_mm, rmse_cm);
% 
% %% 6. Reconstruct Predicted Trajectories for Each Trial
% % We'll go back through trialLengths so we know how many time bins each trial had.
% predTraj = cell(nTrials, nAngles);
% 
% indexStart = 1;
% for ang = 1:nAngles
%     for tr = 1:nTrials
%         T = trialLengths(tr, ang);
%         indexEnd = indexStart + T - 1;
%         % Extract predicted positions for this trial
%         predThis = Y_pred(indexStart:indexEnd, :);  % (T x 2)
%         predTraj{tr, ang} = predThis';              % store as (2 x T)
% 
%         indexStart = indexEnd + 1;
%     end
% end
% 
% %% 7. Plot All Trials in One Figure
% figure; hold on;
% for ang = 1:nAngles
%     for tr = 1:nTrials
%         actual = trial(tr, ang).handPos(1:2, :);  % (2 x T)
%         predicted = predTraj{tr, ang};            % (2 x T)
% 
%         % Plot actual (blue) and predicted (red)
%         plot(actual(1,:), actual(2,:), 'b', 'LineWidth', 1);
%         plot(predicted(1,:), predicted(2,:), 'r', 'LineWidth', 1);
%     end
% end
% xlabel('x direction (mm)');
% ylabel('y direction (mm)');
% title('All Trials, All Angles, All Positions, All Neurons (Linear Regression)');
% legend({'Actual Position','Decoded Position'});
% grid on;
% hold off; 
