% Fjona Lutaj, Soranna Bacanu, Sakshi Singh, Cristina López Ruiz 


% % This script uses k-fold cross-validation to evaluate the position estimator.
% 
% function RMSE = testFunction_for_students_MTb_kFold()
% 
% load monkeydata_training.mat
% 
% % Set random number generator
% rng(2013);
% ix = randperm(length(trial));
% 
% % Define the number of folds for cross-validation
% numFolds = 5;
% cv = cvpartition(length(trial), 'KFold', numFolds);
% 
% % Initialize variables to accumulate the errors
% totalMeanSqError = 0;
% totalPredictions = 0;  
% 
% 
% for fold = 1:numFolds
%     fprintf('Testing fold %d of %d...\n', fold, numFolds);
% 
%     % Split data into training and testing sets for the current fold
%     trainingData = trial(ix(training(cv, fold)),:);
%     testData = trial(ix(test(cv, fold)),:);
% 
% 
%     modelParameters = positionEstimatorTraining(trainingData);
% 
%     meanSqError = 0;
%     n_predictions = 0;  
% 
% 
%     for tr = 1:size(testData, 1)
%         for direc = 1:8
%             decodedHandPos = [];
%             times = 320:20:size(testData(tr,direc).spikes,2);
% 
%             for t = times
%                 past_current_trial.trialId = testData(tr,direc).trialId;
%                 past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
%                 past_current_trial.decodedHandPos = decodedHandPos;
%                 past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);
% 
%                 if nargout('positionEstimator') == 3
%                     [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
%                     modelParameters = newParameters;
%                 elseif nargout('positionEstimator') == 2
%                     [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
%                 end
% 
%                 decodedPos = [decodedPosX; decodedPosY];
%                 decodedHandPos = [decodedHandPos decodedPos];
% 
%                 meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
% 
%             end
%             n_predictions = n_predictions + length(times);
%         end
%     end
% 
%     % Update total error and prediction count
%     totalMeanSqError = totalMeanSqError + meanSqError;
%     totalPredictions = totalPredictions + n_predictions;
% end
% 
% % Calculate the overall RMSE across all folds
% RMSE = sqrt(totalMeanSqError / totalPredictions);
% fprintf('Overall RMSE for %d-fold cross-validation: %f\n', numFolds, RMSE);

% end


% This script uses k-fold cross-validation to evaluate the position estimator.

function RMSE = testFunction_for_students_MTb_kFold()

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% Define the number of folds for cross-validation
numFolds = 10;
cv = cvpartition(length(trial), 'KFold', numFolds);

% Initialize variables to accumulate the errors
totalMeanSqError = zeros(1, numFolds);
totalPredictions = zeros(1, numFolds);  
foldRMSE = zeros(1, numFolds);

figure;
hold on;

for fold = 1:numFolds
    fprintf('Testing fold %d of %d...\n', fold, numFolds);
    
    % Split data into training and testing sets for the current fold
    trainingData = trial(ix(training(cv, fold)),:);
    testData = trial(ix(test(cv, fold)),:);
    
  
    modelParameters = positionEstimatorTraining(trainingData);
    
    meanSqError = 0;
    n_predictions = 0;  
    

    for tr = 1:size(testData, 1)
        for direc = 1:8
            decodedHandPos = [];
            times = 320:20:size(testData(tr,direc).spikes,2);
            
            for t = times
                past_current_trial.trialId = testData(tr,direc).trialId;
                past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);
                
                if nargout('positionEstimator') == 3
                    [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                    modelParameters = newParameters;
                elseif nargout('positionEstimator') == 2
                    [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                end
                
                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos];
                
                meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                
            end
            n_predictions = n_predictions + length(times);
        end
    end
    
    % Update total error and prediction count
    totalMeanSqError(fold) = meanSqError;
    totalPredictions(fold) = n_predictions;
    
    % Calculate RMSE for the current fold
    foldRMSE(fold) = sqrt(meanSqError / n_predictions);
end

% Calculate the mean overall RMSE after 5-fold cross-validation
RMSE = sqrt(sum(totalMeanSqError) / sum(totalPredictions));
fprintf('Overall RMSE for %d-fold cross-validation: %f\n', numFolds, RMSE);

% Plot RMSE for each fold as a line graph
plot(1:numFolds, foldRMSE, 'bo-',LineWidth=1.5);

% Plot mean overall RMSE as a horizontal line
meanRMSE = mean(foldRMSE);
plot([1, numFolds], [meanRMSE, meanRMSE], 'r--',LineWidth=1.5);
text(numFolds + 0.1, meanRMSE, ['Mean RMSE: ', num2str(meanRMSE)], 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left');

xlabel('k-Folds');
ylabel('RMSE');
title('k-Fold Cross Validation');

hold off;

end

