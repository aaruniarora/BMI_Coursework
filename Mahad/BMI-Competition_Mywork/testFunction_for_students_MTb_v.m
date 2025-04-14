% Test Script to give to the students, March 2015
% clc; clear; close all;
%% Continuous Position Estimator Test Script with Validation Set
% This function first calls "positionEstimatorTraining" using the training set
% to get the modelParameters. Then it uses a validation set and a test set to
% decode the trajectory and compute RMSEs. 

function [RMSE_test, RMSE_val] = testFunction_for_students_MTb_v(teamName)
% Start timing
tic

load monkeydata_training.mat  % load data (e.g. monkeydata0.mat)

% Set random number generator for reproducibility
rng(2013);
ix = randperm(length(trial));

% --- Data Split --- 
% Option 1: Use fixed indices (example: 40 training, 10 validation, remainder test)
% trainingData   = trial(ix(1:40),:);
% validationData = trial(ix(41:50),:);
% testData       = trial(ix(51:end),:);

% Option 2: Use percentage splits (60% training, 20% validation, 20% test)
nTotal = length(trial);
nTrain = round(0.6* nTotal);
nVal   = round(0.2 * nTotal);
% The remaining trials are assigned to test
trainingData   = trial(ix(1:nTrain),:);
validationData = trial(ix(nTrain+1:nTrain+nVal),:);
testData       = trial(ix(nTrain+nVal+1:end),:);

addpath(teamName);

fprintf('Testing the continuous position estimator...\n')

%% Train Model using Training Data
modelParameters = positionEstimatorTraining(trainingData);

% Save copies of the model parameters for independent validation and test evaluation
modelParametersVal  = modelParameters; 
modelParametersTest = modelParameters;

%% Evaluate on Validation Set
fprintf('Evaluating on the validation set...\n')
meanSqError_val = 0;
n_predictions_val = 0;  

figure
hold on
axis square
grid on
title('Validation Set Decoded vs Actual Position')

for tr = 1:size(validationData,1)
    disp(['Decoding validation block ', num2str(tr), ' out of ', num2str(size(validationData,1))]);
    pause(0.001)
    for direc = randperm(8)
        decodedHandPos = [];
        times = 320:20:size(validationData(tr,direc).spikes,2);
        for t = times
            past_current_trial.trialId = validationData(tr,direc).trialId;
            past_current_trial.spikes = validationData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = validationData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParametersVal);
                modelParametersVal = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParametersVal);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError_val = meanSqError_val + norm(validationData(tr,direc).handPos(1:2,t) - decodedPos)^2;
        end
        n_predictions_val = n_predictions_val + length(times);
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r');
        plot(validationData(tr,direc).handPos(1,times), validationData(tr,direc).handPos(2,times), 'b')
    end
end

RMSE_val = sqrt(meanSqError_val / n_predictions_val);
fprintf('Validation RMSE: %.2f\n', RMSE_val);

%% Evaluate on Test Set
fprintf('Evaluating on the test set...\n')
meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid on
title('Test Set Decoded vs Actual Position')

for tr = 1:size(testData,1)
    disp(['Decoding test block ', num2str(tr), ' out of ', num2str(size(testData,1))]);
    pause(0.001)
    for direc = randperm(8)
        decodedHandPos = [];
        times = 320:20:size(testData(tr,direc).spikes,2);
        for t = times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParametersTest);
                modelParametersTest = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParametersTest);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
        end
        n_predictions = n_predictions + length(times);
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'b')
    end
end

RMSE_test = sqrt(meanSqError / n_predictions);
fprintf('Test RMSE: %.2f\n', RMSE_test);

%% End timing and clean up
elapsedTime = toc;  
rmpath(genpath(teamName))

fprintf('Execution time: %.2f seconds\n', elapsedTime);
fprintf('Weighted Rank: %.2f\n', 0.9*RMSE_test + 0.1*elapsedTime);

end
