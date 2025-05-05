function RMSEs = testFunction_for_all_models(modelFolders)
% To run:
% modelFolders = {'Soft kNN-PCR', 'Soft kNN-Kalman', 'Soft kNN-LMS', 'SVM-PCR', 'Hard kNN-PCR', 'NN-PCR'};
% RMSEs = testFunction_for_all_models(modelFolders);

    load monkeydata0.mat
    rng(2013);
    ix = randperm(length(trial));

    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    nModels = length(modelFolders);
    modelParametersList = cell(1, nModels);
    testFuncs = cell(1, nModels);
    modelColors = lines(nModels);

    % Train each model using its own files
    for i = 1:nModels
        addpath(modelFolders{i});
        modelParametersList{i} = positionEstimatorTraining(trainingData);
        testFuncs{i} = @positionEstimator;
        rmpath(modelFolders{i});
    end

    RMSEs = zeros(1, nModels);
    totalError = zeros(1, nModels);
    n_predictions = 0;

    figure; hold on; axis square; grid;
    title('Actual vs Predicted Trajectories from Multiple Models');
    xlabel('X direction'); ylabel('Y direction');

    for tr = 1:2  % Change to full size(testData,1) to test all
        for direc = randperm(8)
            times = 320:20:size(testData(tr,direc).spikes,2);
            actualPos = testData(tr,direc).handPos(1:2, times);
            plot(actualPos(1,:), actualPos(2,:), 'k', 'LineWidth', 1.25);  % Actual trajectory

            for m = 1:nModels
                disp(['Model: ' modelFolders(m)])
                decodedHandPos = [];
                modelParams = modelParametersList{m};

                addpath(modelFolders{m});
                for t = times
                    past_current_trial.trialId = testData(tr,direc).trialId;
                    past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
                    past_current_trial.decodedHandPos = decodedHandPos;
                    past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

                    if nargout(testFuncs{m}) == 3
                        [decodedPosX, decodedPosY, modelParams] = testFuncs{m}(past_current_trial, modelParams);
                    else
                        [decodedPosX, decodedPosY] = testFuncs{m}(past_current_trial, modelParams);
                    end

                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos];

                    actualPt = testData(tr,direc).handPos(1:2,t);
                    totalError(m) = totalError(m) + norm(actualPt - decodedPos)^2;
                end
                rmpath(modelFolders{m});

                modelParametersList{m} = modelParams; % in case it's adaptive
                h = plot(decodedHandPos(1,:), decodedHandPos(2,:), 'LineWidth', 1);
                h.Color = [modelColors(m,:), 1 - (m-1)/nModels];  % Alpha decreases with model index

            end
            n_predictions = n_predictions + length(times);
        end
    end

    for m = 1:nModels
        RMSEs(m) = sqrt(totalError(m) / n_predictions);
    end

    legendEntries = [{'Actual trajectory'}, modelFolders];
    legend(legendEntries, 'Location', 'northeast');
    set(gcf, 'Renderer', 'opengl');  % Required for alpha blending
end
