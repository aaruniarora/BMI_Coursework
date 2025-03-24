% Use this to run in command window: [meanRMSE, meanAccuracy] = testFunction_for_students_MTb_kfold('PCR_kNN_08March2025', 10);
function [meanRMSE, meanAccuracy] = testFunction_for_students_MTb_kfold(teamName, k, use_rng)
    % Start timing
    tic

    % Load dataset
    load monkeydata0.mat

    % Check if use_rng argument is provided, otherwise default to true
    if nargin < 2
        use_rng = false;
    end
    
    % Set random number generator
    if use_rng
        rng(2013);
        disp('rng set');
    else
        disp('No rng set');
    end

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
