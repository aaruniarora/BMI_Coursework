% % Test Script to give to the students, March 2015
% % clc; clear; close all;
% %% Continuous Position Estimator Test Script
% % This function first calls the function "positionEstimatorTraining" to get
% % the relevant modelParameters, and then calls the function
% % "positionEstimator" to decode the trajectory. 
% 
% function RMSE = testFunction_for_students_MTb(teamName)
% 
% % Star time
% tic
% 
% % load monkeydata_training.mat
% load monkeydata0.mat
% 
% % Set random number generator
% rng(2013);
% ix = randperm(length(trial));
% 
% addpath(teamName);
% 
% % Select training and testing data (you can choose to split your data in a different way if you wish)
% trainingData = trial(ix(1:50),:);
% testData = trial(ix(51:end),:);
% 
% fprintf('Testing the continuous position estimator...')
% 
% meanSqError = 0;
% n_predictions = 0;  
% 
% % Classification Accuracy
% correctCount   = 0;  % how many times we guessed the right direction
% totalCount     = 0;  % how many direction predictions we made
% predictedLabel = zeros(size(testData,1), 8); 
% 
% figure
% hold on
% axis square
% grid
% 
% % Train Model
% modelParameters = positionEstimatorTraining(trainingData);
% 
% for tr=1:size(testData,1)
%     display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
%     pause(0.001)
%     for direc=randperm(8) 
%         decodedHandPos = [];
% 
%         times=320:20:size(testData(tr,direc).spikes,2);
% 
%         for t=times
%             past_current_trial.trialId = testData(tr,direc).trialId;
%             past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
%             past_current_trial.decodedHandPos = decodedHandPos;
% 
%             past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
% 
%             if nargout('positionEstimator') == 3
%                 [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
%                 modelParameters = newParameters;
%             elseif nargout('positionEstimator') == 2
%                 [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
%             end
% 
%             decodedPos = [decodedPosX; decodedPosY];
%             decodedHandPos = [decodedHandPos decodedPos];
% 
%             meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
%         end
%         n_predictions = n_predictions+length(times);
%         hold on
%         plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
%         plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
% 
%         % Classification code
%         predictedDir = modelParameters.actualLabel;  % The label your code just assigned
%         predictedLabel(tr, direc) = predictedDir;
%         % Compare predictedDir to the true direction (= direc)
%         if predictedDir == direc
%             correctCount = correctCount + 1;
%         end
%         totalCount = totalCount + 1;
%         % -----END----- %
%     end
% end
% 
% legend('Decoded Position', 'Actual Position')
% 
% RMSE = sqrt(meanSqError/n_predictions); 
% 
% classificationAccuracy = correctCount / totalCount;
% 
% % End timing and store the result
% elapsedTime = toc;  
% 
% rmpath(genpath(teamName))
% 
% % Display the elapsed time
% fprintf('Execution time: %.2f seconds\n', elapsedTime);
% fprintf('RMSE: %.4f\n', RMSE);
% fprintf('Weighted Rank: %.2f\n', 0.9*RMSE + 0.1*elapsedTime);
% fprintf('Classification Accuracy = %.2f%% \n', classificationAccuracy * 100);
% 
% end

%% Use this to run in command window: [meanRMSE, meanAccuracy] = testFunction_for_students_MTb('PCR_kNN_08March2025', 10);
function [meanRMSE, meanAccuracy] = testFunction_for_students_MTb(teamName, k)
    % Start timing
    tic

    % Load dataset
    load monkeydata_training.mat

    % Set random number generator
    rng(2013);
    ix = randperm(length(trial));

    % Add path for model files
    addpath(teamName);

    % Define storage for results
    RMSEs = zeros(1, k);
    accuracies = zeros(1, k);

    % Create k-folds
    foldSize = floor(length(trial) / k);
    
    fprintf('Performing %d-fold cross-validation...\n', k);

    for fold = 1:k
        fprintf('Processing fold %d/%d...\n', fold, k);

        % Define training and testing sets
        testIdx = ix((fold-1)*foldSize + 1 : min(fold*foldSize, length(trial)));
        trainIdx = setdiff(ix, testIdx);

        trainingData = trial(trainIdx, :);
        testData = trial(testIdx, :);

        % Train Model
        modelParameters = positionEstimatorTraining(trainingData);

        % Initialize error metrics
        meanSqError = 0;
        n_predictions = 0;
        correctCount = 0;
        totalCount = 0;

        % Loop through test data
        for tr = 1:size(testData,1)
            for direc = randperm(8)
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

                    % Calculate mean squared error
                    meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                end

                n_predictions = n_predictions + length(times);

                % Classification
                predictedDir = modelParameters.actualLabel;
                if predictedDir == direc
                    correctCount = correctCount + 1;
                end
                totalCount = totalCount + 1;
            end
        end

        % Store fold results
        RMSEs(fold) = sqrt(meanSqError / n_predictions);
        accuracies(fold) = correctCount / totalCount;
    end

    % Compute average results
    meanRMSE = mean(RMSEs);
    meanAccuracy = mean(accuracies) * 100;

    % End timing
    elapsedTime = toc;
    
    % Remove path
    rmpath(genpath(teamName));

    % Display results
    fprintf('Average RMSE: %.4f\n', meanRMSE);
    fprintf('Average Classification Accuracy: %.2f%%\n', meanAccuracy);
    fprintf('Total Execution Time: %.2f seconds\n', elapsedTime);
end

