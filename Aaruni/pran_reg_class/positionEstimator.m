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
        lastPosition = 0.7*newPos + 0.3*lastPosition; % Smoothing
    end
    
    decodedPosX = lastPosition(1);
    decodedPosY = lastPosition(2);
    newParams = modelParams;
end