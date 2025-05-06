function RMSEs = testFunction_for_all_models_avg(modelFolders)
% To run:
% modelFolders = {'Soft kNN-PCR', 'Hard kNN-PCR'};
% RMSEs = testFunction_for_all_models_avg(modelFolders);

    load monkeydata0.mat
    rng(2013);
    ix = randperm(length(trial));

    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    nModels = length(modelFolders);
    modelParametersList = cell(1, nModels);
    testFuncs = cell(1, nModels);
    modelColors = lines(nModels);

    times = 320:20:560;
    maxTimeLen = length(times);

    % Train each model
    for i = 1:nModels
        addpath(modelFolders{i});
        modelParametersList{i} = positionEstimatorTraining(trainingData);
        testFuncs{i} = @positionEstimator;
        rmpath(modelFolders{i});
    end

    % Store decoded and actual trajectories: [model x direction x 2 x time x trial]
    decodedAll = cell(nModels, 8);
    actualAll = cell(1, 8);

    for direc = 1:8
        for tr = 1:10%size(testData,1)
            actualAll{direc}(:,:,tr) = testData(tr,direc).handPos(1:2, times); % [2 x time]

            for m = 1:nModels
                decodedHandPos = [];
                modelParams = modelParametersList{m};
                addpath(modelFolders{m});

                for tIdx = 1:maxTimeLen
                    t = times(tIdx);
                    past_current_trial.trialId = testData(tr,direc).trialId;
                    past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
                    past_current_trial.decodedHandPos = decodedHandPos;
                    past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

                    if nargout(testFuncs{m}) == 3
                        [x, y, modelParams] = testFuncs{m}(past_current_trial, modelParams);
                    else
                        [x, y] = testFuncs{m}(past_current_trial, modelParams);
                    end

                    decodedHandPos = [decodedHandPos [x; y]];
                end

                rmpath(modelFolders{m});

                % Pad or truncate to maxTimeLen
                padded = nan(2, maxTimeLen);
                len = min(size(decodedHandPos, 2), maxTimeLen);
                padded(:,1:len) = decodedHandPos(:,1:len);
                decodedAll{m,direc}(:,:,tr) = padded; % [2 x time x trial]
            end
        end
    end

    % === Plot All Average Trajectories in One Plot ===
    figure; hold on; axis equal; grid on;
    % title('Actual vs Predicted Trajectories (Average per Direction)');
    % xlabel('X direction'); ylabel('Y direction');

    % Plot actual average trajectory (black) per direction
    for d = 1:8
        avgActual = nanmean(actualAll{d}, 3);  % [2 x time]
        plot(avgActual(1,:), avgActual(2,:), 'k', 'LineWidth', 2);
    end

    % Plot average prediction per model/direction
    for m = 1:nModels
        for d = 1:8
            avgPred = nanmean(decodedAll{m,d}, 3);  % [2 x time]
            plot(avgPred(1,:), avgPred(2,:), '-', 'Color', modelColors(m,:), 'LineWidth', 1.5);
        end
    end

    legendHandles = gobjects(1, nModels + 1);

    % Plot actual once (first)
    legendHandles(1) = plot(nan, nan, 'k', 'LineWidth', 2);  % dummy for legend
    
    % Then dummy handles for models with correct colors
    for m = 1:nModels
        legendHandles(m+1) = plot(nan, nan, '-', 'Color', modelColors(m,:), 'LineWidth', 1.5);
    end
    
    legendLabels = [{'Actual trajectory'}, modelFolders];
    legend(legendHandles, legendLabels, 'Location', 'southeast');

    % === Compute RMSE ===
    RMSEs = zeros(1, nModels);
    for m = 1:nModels
        totalSqErr = 0;
        totalCount = 0;
        for d = 1:8
            actual = actualAll{d};          % [2 x time x trials]
            pred = decodedAll{m,d};         % [2 x time x trials]

            if ~isempty(pred)
                err = pred - actual;        % [2 x time x trials]
                sqErr = nansum(err.^2, 1);  % [1 x time x trials]
                totalSqErr = totalSqErr + nansum(sqErr(:));
                totalCount = totalCount + sum(~isnan(sqErr(:)));
            end
        end
        RMSEs(m) = sqrt(totalSqErr / totalCount);
    end
    save_figure(gcf, '.', 'trajectories_comp', 'pdf')
end
