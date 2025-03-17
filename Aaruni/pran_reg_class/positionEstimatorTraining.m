function modelParameters = positionEstimatorTraining(trainingData)
<<<<<<< Updated upstream
    % Parameters
    classifierBinSize = 20;        % 20ms bins
    classifierWindowSize = 300;    % 300ms analysis window
    regressorBinSize = 20;       
    regressorWindowSize = 300;
    stepSize = 20;                 % 20ms window sliding step
    
    [numTrials, numAngles] = size(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);

    %% 1. Enhanced Classifier Training with Sliding Windows
    [classifierFeatures, validFeatures, featureMask] = preprocessSlidingWindows(trainingData, ...
        classifierBinSize, classifierWindowSize, stepSize);
    
    classifierLabels = repmat(repelem(1:numAngles, numTrials)', classifierWindowSize/stepSize, 1);
    
    % Train regularized discriminant classifier
    classifier = fitcdiscr(classifierFeatures, classifierLabels, ...
        'DiscrimType', 'pseudoLinear', ...
        'Gamma', 0.4, ...
        'Delta', 1e-4, ...
        'Prior', 'uniform');

    %% 2. Regressor Training (Unchanged)
    regressors = cell(1, numAngles);
    for dir = 1:numAngles
        [allFeat, allPos] = preprocessForRegression(trainingData(:, dir), regressorBinSize, regressorWindowSize);
        
        [coeff, score, ~, ~, explained, mu] = pca(allFeat);
        numComponents = find(cumsum(explained) >= 95, 1, 'first');
        Xb = [score(:,1:numComponents), ones(size(score,1),1)];
        Beta = Xb \ allPos;
        
        regressors{dir} = struct(...
            'projMatrix', coeff(:,1:numComponents), ...
            'Beta', Beta, ...
            'mu', mu, ...
            'binSize', regressorBinSize, ...
            'windowSize', regressorWindowSize);
    end

    %% 3. Model Packaging
    modelParameters = struct(...
        'classifier', classifier, ...
        'regressors', {regressors}, ...
        'featureMask', featureMask, ...
        'windowParams', struct('binSize', classifierBinSize, ...
                              'windowSize', classifierWindowSize, ...
                              'stepSize', stepSize), ...
        'trainingSamplesPerTrial', classifierWindowSize/stepSize);
end

function [features, validFeatures, featureMask] = preprocessSlidingWindows(data, binSize, windowSize, stepSize)
    % Extract overlapping windows from entire trial duration
    numNeurons = size(data(1,1).spikes, 1);
    features = [];
    
    windowBins = windowSize/binSize;
    stepBins = stepSize/binSize;
    
    for angle = 1:size(data,2)
        for trial = 1:size(data,1)
            spikes = data(trial,angle).spikes;
            [fr, totalBins] = preprocessSpikes(spikes, binSize);
            
            % Extract sliding windows
            for startBin = 1:stepBins:(totalBins - windowBins + 1)
                endBin = startBin + windowBins - 1;
                windowData = fr(:, startBin:endBin);
                features = [features; windowData(:)']; %#ok<AGROW>
            end
        end
    end
    
    % Feature selection
    featVars = var(features);
    validFeatures = featVars > 1e-4;
    features = features(:, validFeatures);
    
    % Create feature mask for real-time alignment
    featureMask = false(1, numNeurons*windowBins);
    featureMask(1:length(validFeatures)) = validFeatures;
end

function [fr, bins] = preprocessSpikes(spikes, binSize)
    T = size(spikes, 2);
    bins = floor(T / binSize);
    spikesBinned = sum(reshape(spikes(:,1:bins*binSize), size(spikes,1), binSize, []), 2);
    fr = (1000 / binSize) * permute(spikesBinned, [1,3,2]);
end
=======
    modelParameters.binSize = 30;
    modelParameters.historyBins = 12;
    modelParameters.normalize = false;

    % Xtrain - Feature matrix: (totalBins, spikingFeatures)
    [Xtrain, Ytrain] = buildFeatures(trainingData, modelParameters.binSize, ...
        modelParameters.historyBins, modelParameters.normalize);

    [Xtrain_pca, V_reduced, X_mean, X_std] = pcaReduction(Xtrain, 0.95);
    modelParameters.V_reduced = V_reduced;
    modelParameters.X_mean = X_mean;
    modelParameters.X_std = X_std;

    modelParameters.classificationModel = trainLDA(Xtrain_pca, Ytrain);

    % % Evaluate on training set
    trainPreds = predictLDA(modelParameters.classificationModel, Xtrain_pca);
    finalTrainAcc = mean(trainPreds == Ytrain);
    disp(['Final train accuracy: ', num2str(finalTrainAcc * 100, '%.2f'), '%']);
    
    % modelParameters.regressionModels = trainAngleSpecificSVRModels();
end

function [X, Y] = buildFeatures(data, binSize, historyBins, leftoverNormalize)
    %  For each trial, we create partial bins, each bin includes spiking data => (numNeurons x 1).
    %
    % Inputs:
    %   data              - Trial data structure
    %   binSize           - Size of bins in ms
    %   historyBins       - Number of history bins to include
    %   leftoverNormalize - Whether to normalize leftover bins
    %
    % Outputs:
    %   X - Feature matrix: (totalBins, spikingFeatures) 
    %           totalBins is across all trials
    %           spikingFeatures is historyBins * numNeurons -> mean spike rate
    %   Y - Angle labels: (totalBins, 1)

        [numTrials, numAngles] = size(data); % 50 trials x 8 angles
        numNeurons = size(data(1, 1).spikes, 1); % 98 neurons
    
        bigX = {};
        bigY = {};
        for angle = 1:numAngles
            for trial = 1:numTrials
                rawSpikes = data(trial, angle).spikes;
                timeMs = size(rawSpikes, 2); % time is around 600ms (no cropping)
    
                nFull = floor(timeMs / binSize); % no of full bins
                leftover = timeMs - nFull * binSize; % no of leftover time stamps 
                numBins = nFull + (leftover > 0); % total no of bins

                % For each bin, computes the mean no of spikes for each neuron - Mean Spike Rate
                binned = zeros(numNeurons, numBins, 'single');
                for b = 1:nFull
                    bin_start = (b-1) * binSize + 1;
                    bin_end = b * binSize;
                    seg = rawSpikes(:, bin_start:bin_end);
                    binned(:, b) = mean(seg, 2);
                end
                
                % Process leftover bin if exists and if greater than half the binSize
                if leftover > 0.5*binSize
                    bin_start = nFull * binSize + 1;
                    bin_end = timeMs;
                    seg = rawSpikes(:, bin_start:bin_end);
                    partialMean = mean(seg, 2);
                    
                    % Apply normalization if needed
                    if leftoverNormalize
                        ratio = binSize / leftover;
                        partialMean = partialMean * ratio;
                    end
                    
                    binned(:, numBins) = partialMean;
                end
    
                locX = [];
                locY = [];
                
                % Build sliding window features
                for b = 1:numBins
                    if b < historyBins
                        % Skip if can't form a full window
                        continue;
                    end
                    
                    % Get window of data
                    window = binned(:, b-historyBins+1 : b); % (98 neurons x history_bin_size)
                    featRow = reshape(window, 1, []);
    
                    % Store features and label
                    locX = [locX; featRow]; % Data to train for classification
                    locY = [locY; angle]; % Ground truth label for classification - my angles 
                end
                % locX is of size (no_of_bins x feat row) wherein no_of_bins varies as each trial is of different lengths
                % feat row is of size (1, 1176) wherein 1176 = number of neurons (98) * history_bin_size (12)
                bigX{end+1} = locX; % size is (1, numTrials*numAngles)
                bigY{end+1} = locY; % size is (1, numTrials*numAngles)
            end
        end
        
        % Combine all trials into single matrices
        totalBins = sum(cellfun(@(x) size(x, 1), bigX)); % Total no of bins across all trials = 4369
        dimFeat = size(bigX{1}, 2); % this 1176 (numNeurons*history_bin_size)
        X = zeros(totalBins, dimFeat, 'single');
        Y = zeros(totalBins, 1, 'int32');
        rowPos = 1;
        for i = 1:length(bigX)
            blockX = bigX{i};
            blockY = bigY{i};
            numBins = size(blockX, 1);
            X(rowPos:rowPos+numBins-1, :) = blockX;
            Y(rowPos:rowPos+numBins-1) = blockY;
            rowPos = rowPos + numBins;
        end
    end

    function [Xpca, V_reduced, X_mean, X_std] = pcaReduction(X, varianceThreshold)
        X_mean = mean(X, 1);
        X_std = std(X, [], 1);
        X_std(X_std == 0) = 1; % Wherever std is 0, make it 1 so no zero error
        X_norm = (X - X_mean)./X_std;

        % U - eigenvectors (PCs)
        [U, S, V] = svd(X_norm, 'econ');
        singular_values = diag(S);
        explained_var = (singular_values .^ 2) / sum(singular_values .^ 2);
        cum_var = cumsum(explained_var);
    
        numPCs = find(cum_var >= varianceThreshold, 1, 'first');
        V_reduced = V(:, 1:numPCs);
    
        disp("Size of X_norm: "), disp(size(X_norm));
        disp("Size of V_reduced: "), disp(size(V_reduced));
    
        Xpca = X_norm * V_reduced;
    
        disp("Original Feature Size:"), disp(size(X))
        disp("Reduced Feature Size:"), disp(size(Xpca))
        disp("Number of PCs Retained:"), disp(numPCs)
    
    end


function ldaModel = trainLDA(X, Y)
    % LDA - Linear Discriminant Analysis
    %
    % Inputs:
    %   X - Feature matrix: (totalBins, spikingFeatures)
    %   Y - Angle/class labels: (totalBins, 1)
    %
    % Outputs:
    %   ldaModel

    classes = unique(Y);
    numClasses = length(classes);
    numFeatures = size(X, 2);

    meanTotal = mean(X, 1);

    Sw = zeros(numFeatures, numFeatures); % Within-class scatter
    Sb = zeros(numFeatures, numFeatures); % Between-class scatter

    classMeans = zeros(numClasses, numFeatures);
    classPriors = zeros(numClasses, 1);
    for i = 1:numClasses
        classData = X(Y == classes(i), :);
        classMean = mean(classData, 1);
        classMeans(i, :) = classMean;
        classPriors(i) = size(classData, 1) / size(X, 1);

        % Within-class scatter
        classScatter = (classData - classMean)' * (classData - classMean);
        Sw = Sw + classScatter;
        
        % Between-class scatter
        meanDiff = (classMean - meanTotal)';
        Sb = Sb + size(classData, 1) * (meanDiff * meanDiff');
    end

    [eigVecs, eigVals] = eig(Sb, Sw);
    [~, sortedIdx] = sort(diag(eigVals), 'descend');
    W = eigVecs(:, sortedIdx); % Projection matrix

    % Store trained LDA model
    ldaModel.W = W;
    ldaModel.classMeans = classMeans;
    ldaModel.classPriors = classPriors;
    ldaModel.classes = classes;
end

function [predictions, predScores] = predictLDA(ldaModel, Xtest)
    % predictLDA - Uses trained LDA model to classify new samples and return raw LDA scores
    %
    % Inputs:
    %   ldaModel - Struct containing LDA projection matrix & class statistics
    %   Xtest - New data (samples x features)
    %
    % Outputs:
    %   predictions - Predicted class labels
    %   predScores - Raw LDA scores (negative Mahalanobis distance to each class mean)

    % Project data into LDA space
    projectedData = Xtest * ldaModel.W;

    % Compute Mahalanobis distance to class means
    numClasses = size(ldaModel.classMeans, 1);
    numSamples = size(Xtest, 1);
    distances = zeros(numSamples, numClasses);

    for i = 1:numClasses
        classMeanProj = ldaModel.classMeans(i, :) * ldaModel.W;
        distances(:, i) = sum((projectedData - classMeanProj).^2, 2);
    end

    % Assign class with the smallest distance
    [~, minIdx] = min(distances, [], 2);
    predictions = ldaModel.classes(minIdx);
    
    % Convert distances to scores (negative distances so that higher means better match)
    predScores = -distances;
    
    % Apply softmax normalization to predScores to avoid overfitting to train data
    predScores = exp(predScores - max(predScores, [], 2));  % Prevent numerical instability
    predScores = predScores ./ sum(predScores, 2);
end

%%% REGRESSION FUNCTIONS

function models = trainAngleSpecificSVRModels()
    global trainingData testData bestParams
    
    [~, numAngles] = size(trainingData);
    
    % Store models per angle
    models = cell(numAngles, 1);
    
    for angle = 1:numAngles
        models{angle} = trainSingleAngleSVRModel(angle);
    end
    
    % Evaluate models
    rmse = evaluateAngleSpecificSVR(testData, models);
    fprintf('Final RMSE with SVR: %.4f\n', rmse);
end

%% ===================== Train SVR for a Single Angle =====================
% function model = trainSingleAngleSVRModel(angle)
%     global trainingData bestParams
    
%     fprintf('Training SVR model for angle %d...\n', angle);
    
%     % Build features for this angle
%     [Xtrain, Ytrain] = buildFeatures(trainingData, bestParams.binSize, bestParams.historyBins, bestParams.normalize, angle);
    
%     % Train SVR regressors for (X, Y) separately
%     model.svrX = fitrsvm(Xtrain, Ytrain(:,1), 'KernelFunction', 'gaussian');
%     model.svrY = fitrsvm(Xtrain, Ytrain(:,2), 'KernelFunction', 'gaussian');
    
%     fprintf('SVR models for angle %d trained successfully.\n', angle);
% end

% function [X, Y] = buildFeatures(data, binSize, historyBins, normalize, angleFilter)
%     global pcaCoeff pcaMean numPCs

%     [numTrials, numAngles] = size(data);
%     numNeurons = size(data(1,1).spikes, 1);
    
%     if nargin < 5
%         useFilter = false;
%         targetAngles = 1:numAngles;
%     else
%         useFilter = true;
%         targetAngles = angleFilter;
%     end
    
%     % Preallocate based on estimated size
%     totalSamples = estimateSampleSize(data, binSize, historyBins, targetAngles);
%     X = zeros(totalSamples, numPCs * historyBins, 'single');
%     Y = zeros(totalSamples, 2, 'single');
    
%     sampleIdx = 1;
%     for angle = targetAngles
%         for trial = 1:numTrials
%             spikes = data(trial, angle).spikes;
%             Tms = size(spikes, 2);
%             nBins = ceil(Tms / binSize);
            
%             if nBins < historyBins
%                 continue;
%             end
            
%             % Binning
%             binned = binSpikes(spikes, binSize, nBins, normalize);
            
%             % PCA Transform - corrected dimensions
%             projectedBins = (binned - repmat(pcaMean, size(binned, 1), 1)) * pcaCoeff;
            
%             % Sliding window features
%             for binIdx = historyBins:nBins
%                 window = reshape(projectedBins(binIdx-historyBins+1:binIdx, :)', 1, []);
%                 X(sampleIdx, :) = window;
                
%                 % Get hand position at the correct time
%                 handPosIdx = min(binSize * binIdx, Tms);
%                 Y(sampleIdx, :) = data(trial, angle).handPos(1:2, handPosIdx)';
                
%                 sampleIdx = sampleIdx + 1;
%             end
%         end
%     end
    
%     % Trim excess
%     X = X(1:sampleIdx-1, :);
%     Y = Y(1:sampleIdx-1, :);
% end

% function nSamples = estimateSampleSize(data, binSize, historyBins, targetAngles)
%     % Estimates the number of total samples
%     nSamples = 0;
%     [numTrials, ~] = size(data);
    
%     for angle = targetAngles
%         for trial = 1:numTrials
%             Tms = size(data(trial, angle).spikes, 2);
%             nBins = ceil(Tms / binSize);
%             if nBins >= historyBins
%                 nSamples = nSamples + (nBins - historyBins + 1);
%             end
%         end
%     end
% end

% function binned = binSpikes(spikes, binSize, nBins, normalize)
%     % Bins the spikes into time windows
%     numNeurons = size(spikes, 1);
%     binned = zeros(nBins, numNeurons);
    
%     for bin = 1:nBins
%         idxStart = (bin-1)*binSize + 1;
%         idxEnd = min(bin*binSize, size(spikes, 2));
%         segment = spikes(:, idxStart:idxEnd);
%         binMean = mean(segment, 2);
        
%         if normalize && (idxEnd - idxStart + 1) < binSize
%             binMean = binMean * (binSize / (idxEnd - idxStart + 1));
%         end
        
%         binned(bin, :) = binMean';
%     end
% end

% function rmse = evaluateAngleSpecificSVR(data, models)
%     global bestParams
    
%     [~, numAngles] = size(data);
%     allPreds = [];
%     allTrueY = [];
    
%     for angle = 1:numAngles
%         [Xtest, Ytest] = buildFeatures(data, bestParams.binSize, bestParams.historyBins, bestParams.normalize, angle);
%         if isempty(Xtest), continue; end
        
%         predX = predict(models{angle}.svrX, Xtest);
%         predY = predict(models{angle}.svrY, Xtest);
        
%         allPreds = [allPreds; predX, predY];
%         allTrueY = [allTrueY; Ytest];
%     end
    
%     rmse = sqrt(mean(sum((allTrueY - allPreds).^2, 2)));
% end
>>>>>>> Stashed changes
