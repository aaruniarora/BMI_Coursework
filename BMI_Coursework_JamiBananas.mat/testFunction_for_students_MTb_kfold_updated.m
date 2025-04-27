% Use this to run in command window: [meanRMSE, meanAccuracy] = testFunction_for_students_MTb_kfold('PCR_kNN_08March2025', 10);
function [meanRMSE, meanAccuracy] = testFunction_for_students_MTb_kfold(teamName, k, use_rng)
   
    % Check if use_rng argument is provided, otherwise default to true
    if nargin < 3
        use_rng = false;
    end
    
    % Set random number generator
    if use_rng
        rng(2013);
        disp('rng set to 2013');
    else
        disp('No rng set');
    end

    % Start timing
    tic

    % Load dataset
    load monkeydata0.mat

    ix = randperm(length(trial));

    % Add path for model files
    addpath(teamName);

    % Define storage for results
    RMSEs = zeros(1, k);
    RMSE_bin = zeros(1, k);
    accuracies = zeros(1, k);
    accuracies_bin = zeros(1, k);

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

        % Initialize storage for per-bin statistics
        timeBins = 320:20:560;  % same as 'times'
        nBins = length(timeBins);
        correctPerBin = zeros(1, nBins);
        totalPerBin = zeros(1, nBins);
        squaredErrorPerBin = zeros(1, nBins);
        countPerBin = zeros(1, nBins);

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
                    
                    % RMSE per bin
                    binIdx = find(timeBins == t);
                    squaredError = norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                    squaredErrorPerBin(binIdx) = squaredErrorPerBin(binIdx) + squaredError;
                    countPerBin(binIdx) = countPerBin(binIdx) + 1;
    
                     % Classification per bin
                    if modelParameters.actualLabel == direc
                        correctPerBin(binIdx) = correctPerBin(binIdx) + 1;
                    end
                    totalPerBin(binIdx) = totalPerBin(binIdx) + 1;
                    
                    end
    
                    n_predictions = n_predictions + length(times);

                    % Classification code
                    predictedDir = modelParameters.actualLabel;  % The label your code just assigned
                    predictedLabel(tr, direc) = predictedDir;
                    % Compare predictedDir to the true direction (= direc)
                    if predictedDir == direc
                        correctCount = correctCount + 1;
                    end
                    totalCount = totalCount + 1;

                end
        end
        meanRMSE_perBin = sqrt(squaredErrorPerBin ./ countPerBin);
        accuracy_perBin = correctPerBin ./ totalPerBin;
 
        % Store fold results
        RMSE_bin(fold) = mean(meanRMSE_perBin);
        RMSEs(fold) = sqrt(meanSqError / n_predictions);
        accuracies_bin(fold) = mean(accuracy_perBin)*100;
        accuracies(fold) = correctCount / totalCount;
    end

    % Compute average results
    meanRMSE_bin = mean(RMSE_bin);
    meanRMSE = mean(RMSEs);
    meanAccuracy_bin = mean(accuracies_bin);
    meanAccuracy = mean(accuracies) * 100;

    % End timing
    elapsedTime = toc;

    % Remove path
    rmpath(genpath(teamName));

    % Display results
    fprintf('Average RMSE: %.4f\n', meanRMSE);
    fprintf('Average RMSE over time bins: %.4f\n', meanRMSE_bin);
    fprintf('Average Classification Accuracy over time bins: %.2f%%\n', meanAccuracy_bin);
    fprintf('Average Classification Accuracy: %.2f%%\n', meanAccuracy);
    fprintf('Total Execution Time: %.2f seconds\n', elapsedTime);
    
end
