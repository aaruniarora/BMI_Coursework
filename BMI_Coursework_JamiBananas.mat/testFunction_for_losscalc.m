function RMSE = testFunction_for_losscalc(teamName, figname, use_rng)
% TESTFUNCTION_FOR_LOSSCALC
% Computes and plots train vs. test RMSE per time bin for the continuous
% position estimator. Also reports overall RMSE, classification accuracy, etc.

    % Handle inputs
    if nargin < 2, figname = ''; end
    if nargin < 3, use_rng = true; end

    % Seed RNG for reproducibility
    if use_rng
        rng(2013);
        disp('rng set to 2013');
    else
        disp('No rng set');
    end

    % Start timer
    tic

    % Load data
    load monkeydata0.mat

    % Random train/test split
    ix = randperm(length(trial));
    trainingData = trial(ix(1:50), :);
    testData     = trial(ix(51:end), :);

    addpath(teamName);
    fprintf('Testing the continuous position estimator...\n');

    % Define time bins (in ms)
    timeBins = 320:20:560;
    nBins    = numel(timeBins);
    times    = timeBins;  % index into handPos/spikes

    % Initialize accumulators
    sqrErrTrain   = zeros(1, nBins);
    countTrain    = zeros(1, nBins);
    sqrErrTest    = zeros(1, nBins);
    countTest     = zeros(1, nBins);
    correctPerBin = zeros(1, nBins);
    totalPerBin   = zeros(1, nBins);

    % For global RMSE & classification
    meanSqError   = 0;
    n_predictions = 0;
    correctCount  = 0;
    totalCount    = 0;
    predictedLabel = zeros(size(testData,1), 8);

    % Plot: trajectories
    figure; hold on; axis square; grid on;

    % 1) Train the model
    modelParameters = positionEstimatorTraining(trainingData);

    % 2) Compute TRAINING-set error per bin
    for tr = 1:size(trainingData,1)
        for direc = 1:8
            decodedHandPos = [];
            for b = 1:nBins
                t = times(b);
                % Build the partial-trial struct
                past_current_trial.trialId        = trainingData(tr,direc).trialId;
                past_current_trial.spikes         = trainingData(tr,direc).spikes(:,1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos   = trainingData(tr,direc).handPos(1:2,1);

                % Two-output call so modelParameters stays fixed
                [xHat, yHat] = positionEstimator(past_current_trial, modelParameters);
                decodedPos   = [xHat; yHat];

                % Accumulate train error
                sqrErrTrain(b) = sqrErrTrain(b) + norm(trainingData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                countTrain(b)  = countTrain(b)  + 1;

                decodedHandPos = [decodedHandPos, decodedPos];
            end
        end
    end

    % 3) Decode TEST set, accumulate TEST error & classification stats
    for tr = 1:size(testData,1)
        display(['Decoding block ', num2str(tr), ' out of ', num2str(size(testData,1))]);
        pause(0.001);
        for direc = randperm(8)
            decodedHandPos = [];
            for b = 1:nBins
                t = times(b);
                % Build the partial-trial struct
                past_current_trial.trialId        = testData(tr,direc).trialId;
                past_current_trial.spikes         = testData(tr,direc).spikes(:,1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos   = testData(tr,direc).handPos(1:2,1);

                % Two- or three-output depending on your estimator
                if nargout('positionEstimator') == 3
                    [xHat, yHat, modelParameters] = positionEstimator(past_current_trial, modelParameters);
                else
                    [xHat, yHat] = positionEstimator(past_current_trial, modelParameters);
                end
                decodedPos = [xHat; yHat];
                decodedHandPos = [decodedHandPos, decodedPos];

                % Global RMSE
                err2 = norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                meanSqError   = meanSqError   + err2;
                n_predictions = n_predictions + 1;

                % Per-bin TEST error
                sqrErrTest(b) = sqrErrTest(b) + err2;
                countTest(b)  = countTest(b)  + 1;

                % Per-bin classification
                if modelParameters.actualLabel == direc
                    correctPerBin(b) = correctPerBin(b) + 1;
                end
                totalPerBin(b) = totalPerBin(b) + 1;
            end

            % Plot decoded vs. actual trajectory
            plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r');
            plot( testData(tr,direc).handPos(1,times), ...
                  testData(tr,direc).handPos(2,times), 'b');

            % Global classification accuracy
            predictedDir = modelParameters.actualLabel;
            predictedLabel(tr, direc) = predictedDir;
            if predictedDir == direc
                correctCount = correctCount + 1;
            end
            totalCount = totalCount + 1;
        end
    end

    legend('Decoded Position', 'Actual Position');

    % 4) Compute global metrics
    RMSE = sqrt(meanSqError / n_predictions);
    classificationAccuracy = correctCount / totalCount;
    elapsedTime = toc;

    rmpath(genpath(teamName));

    fprintf('\nExecution time: %.2f seconds\n', elapsedTime);
    fprintf('Overall RMSE: %.4f\n', RMSE);
    fprintf('Weighted Rank: %.2f\n', 0.9*RMSE + 0.1*elapsedTime);
    fprintf('Classification Accuracy = %.2f%%\n', classificationAccuracy*100);

    % 5) Compute per-bin RMSE and accuracy
    trainRMSE = sqrt(sqrErrTrain ./ countTrain);
    testRMSE  = sqrt(sqrErrTest  ./ countTest);
    accPerBin = correctPerBin ./ totalPerBin;

    % Display as table
    fprintf('\nTime Bin (ms) | Accuracy (%%) | Train RMSE | Test RMSE\n');
    fprintf('--------------|---------------|------------|----------\n');
    for b = 1:nBins
        fprintf('%10d     |     %6.2f     |   %6.2f   |   %6.2f\n', ...
                timeBins(b), accPerBin(b)*100, trainRMSE(b), testRMSE(b));
    end

    % 6) Plot the train vs. test RMSE curve
    figure; hold on; grid on;
    plot(timeBins, trainRMSE, '-o', 'LineWidth',1.5, 'DisplayName','Train RMSE');
    plot(timeBins, testRMSE,  '-s', 'LineWidth',1.5, 'DisplayName','Test RMSE');
    xlabel('Time bin center (ms)');
    ylabel('RMSE (mm)');
    title('Train vs. Test RMSE per Time Bin');
    legend('Location','NorthEast');

    % 7) Optionally save the loss-curve figure
    if ~isempty(figname)
        save_figure(gcf, 'figures', figname, 'pdf', 'vector', false);
    end
end
