function [decodedPosX, decodedPosY, newParams] = positionEstimator(past_current_trial, modelParams)
    persistent lastPosition;
    
    %% 1. Feature Alignment
    % Get parameters from model
    mask = modelParams.featureMask;
    expectedFeatures = modelParams.expectedFeatures;
    classifierParams = modelParams.classifierParams;
    
    % Extract raw features
    currentSpikes = past_current_trial.spikes;
    [fr, ~] = preprocessSpikes(currentSpikes, classifierParams.binSize);
    rawFeatures = fr(:)';
    
    % Pad/crop features to match training dimensions
    if length(rawFeatures) > length(mask)
        processedFeatures = rawFeatures(1:length(mask));
    else
        paddedFeatures = zeros(1, length(mask));
        paddedFeatures(1:length(rawFeatures)) = rawFeatures;
        processedFeatures = paddedFeatures;
    end
    
    % Apply feature mask
    finalFeatures = processedFeatures(mask);
    
    %% 2. Continuous Classification
    try
        predictedDir = predict(modelParams.classifier, finalFeatures);
    catch
        predictedDir = 1; % Fallback to first direction
    end
    
    %% 3. Position Regression
    try
        regressor = modelParams.regressors{predictedDir};
        [fr, binCount] = preprocessSpikes(currentSpikes, regressor.binSize);
        
        windowBins = regressor.windowSize / regressor.binSize;
        t = max(windowBins, binCount);
        featVec = fr(:, t-windowBins+1:t);
        
        % Project and predict
        featCentered = featVec(:)' - regressor.mu;
        decodedPos = [featCentered * regressor.projMatrix, 1] * regressor.Beta;
        newPos = decodedPos';
    catch
        newPos = lastPosition; % Fallback to last position
    end
    
    %% 4. Maintain State
    if isempty(lastPosition)
        lastPosition = past_current_trial.startHandPos(1:2);
    else
        lastPosition = newPos;
    end
    
    decodedPosX = lastPosition(1);
    decodedPosY = lastPosition(2);
    newParams = modelParams;
end

function [features, validFeatures, featureMask] = preprocessClassifierFeatures(data, binSize, windowSize)
    % Feature extraction and selection
    numNeurons = size(data(1,1).spikes, 1);
    features = [];
    
    % 1. Extract raw features
    for angle = 1:size(data,2)
        for trial = 1:size(data,1)
            spikes = data(trial,angle).spikes(:,1:windowSize);
            [fr, ~] = preprocessSpikes(spikes, binSize);
            features = [features; fr(:)']; 
        end
    end
    
    % 2. Feature selection
    featVars = var(features);
    validFeatures = featVars > 1e-6; % Threshold for minimum variance
    features = features(:, validFeatures);
    
    % 3. Create full feature mask
    featureMask = false(1, numNeurons*(windowSize/binSize));
    featureMask(1:length(validFeatures)) = validFeatures;
end

function [fr, bins] = preprocessSpikes(spikes, binSize)
    T = size(spikes, 2);
    bins = floor(T / binSize);
    spikesBinned = sum(reshape(spikes(:,1:bins*binSize), size(spikes,1), binSize, []), 2);
    fr = (1000 / binSize) * permute(spikesBinned, [1,3,2]);
end

function [allFeat, allPos] = preprocessForRegression(trials, binSize, windowSize)
    allFeat = []; allPos = [];
    windowBins = windowSize / binSize;
    
    for tr = 1:numel(trials)
        [fr, binCount] = preprocessSpikes(trials(tr).spikes, binSize);
        handPos = trials(tr).handPos(1:2, :);
        
        for t = windowBins:binCount-1
            featVec = fr(:, t-windowBins+1:t);
            allFeat = [allFeat; featVec(:)'];
            allPos = [allPos; handPos(:, t*binSize)'];
        end
    end
end





