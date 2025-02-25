clc; clear; close all;

%% 1) Load the data
load('monkeydata_training.mat'); % trial is 100 x 8 struct: trial(n, k).spikes and trial(n, k).handPos

[numTrials, numAngles]  = size(trial); % 100 x 8
numNeurons = size(trial(1,1).spikes, 1); % 98

% We'll build X and Y by concatenating all trials & angles
X_all = [];  % Design matrix (#timepoints_total x #neurons)
Y_all = [];  % Target matrix (#timepoints_total x 2) for (x,y)

for angleIdx = 1:numAngles
    for tr = 1:numTrials
        
        % Extract spikes and hand positions for this trial
        spikes = trial(tr, angleIdx).spikes;   % (98 x T)
        handPos = trial(tr, angleIdx).handPos; % (3 x T)
        
        % Number of time bins in this trial
        T = size(spikes, 2);
        
        % We'll only use x and y from handPos
        xPositions = handPos(1, :);  % (1 x T)
        yPositions = handPos(2, :);  % (1 x T)
        
        % Transpose so that each row is one time bin
        % spikes_for_regression: T x 98
        spikes_for_regression = spikes';  
        
        % targets_for_regression: T x 2
        targets_for_regression = [xPositions', yPositions'];
        
        % Concatenate
        X_all = [X_all; spikes_for_regression];
        Y_all = [Y_all; targets_for_regression];
    end
end

% At this point:
%  - X_all is (#all_timepoints x 98)
%  - Y_all is (#all_timepoints x 2)

%% 2) Fit a linear regression model
%    We want W in R^(98 x 2), so that X_all * W = Y_all
%    Solve via ordinary least squares: W = (X^T X)^(-1) X^T Y
W = (X_all' * X_all) \ (X_all' * Y_all);

% W is a 98 x 2 matrix of regression weights

disp('Trained linear regression model: W is [98 x 2]');

%% 3) Evaluate on a single trial (as a naive check)
%    We'll just pick trial(1,1) for demonstration
spikes_test = trial(1,1).spikes;    % (98 x Ttest)
handPos_test = trial(1,1).handPos;  % (3 x Ttest)
Ttest = size(spikes_test, 2);

% Build test design matrix for each time bin
X_test = spikes_test';  % Ttest x 98
% True positions (x,y) for comparison
Y_true = [handPos_test(1,:)' handPos_test(2,:)'];  % Ttest x 2

% Predict
Y_pred = X_test * W;  % Ttest x 2

%% 4) Compute a simple RMSE
mse_val = mean(sum((Y_true - Y_pred).^2,2));  % average squared error
rmse_val = sqrt(mse_val);

fprintf('RMSE on trial(1,1) is %.3f\n', rmse_val);

%% 5) (Optional) Plot predicted vs. actual trajectory for this trial
figure; hold on;
plot(Y_true(:,1), Y_true(:,2), 'b', 'LineWidth', 1.5);  % actual
plot(Y_pred(:,1), Y_pred(:,2), 'r--', 'LineWidth', 1.5); % predicted
legend({'Actual', 'Predicted'});
xlabel('X-position (mm)'); ylabel('Y-position (mm)');
title('Naive Linear Regression: Trial(1,1)');
axis equal;
grid on;
