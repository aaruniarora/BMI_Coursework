function [predictedX, predictedY, updatedModelParameters] = positionEstimator(testTrialData, trainedModelParameters)
% POSITIONESTIMATOR_CONFIDENCE Decodes hand position from spike data using PCR,
% with a confidence-based update on the classifier output.
%
% Inputs:
%   testTrialData         - Structure containing spike data for the trial.
%   trainedModelParameters- Structure with model parameters from training.
%
% Outputs:
%   predictedX            - Decoded X position.
%   predictedY            - Decoded Y position.
%   updatedModelParameters- (Possibly updated) model parameters.

% Copy the trained model parameters
updatedModelParameters = trainedModelParameters;

% ----------------------- Initialization -----------------------
numDirections = 8;       % Number of possible movement directions.
binSize = 20;            % Binning resolution (ms).
gaussianScale = 50;      % Scale for the Gaussian kernel.
targetAngles = [30 70 110 150 190 230 310 350];

% Preprocess the test spike data.
processedTrial = binAndSqrtSpikes(testTrialData, binSize, true);
smoothedTrial = computeFiringRates(processedTrial, binSize, gaussianScale);

trialDuration = size(testTrialData.spikes, 2);
numNeurons = size(smoothedTrial(1,1).rates, 1);

% ------------------ Determine Reaching Direction ------------------
if trialDuration <= 560
    timeWindowIndex = (trialDuration/binSize) - (320/binSize) + 1;
    
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);
    numNeurons = numNeurons - length(removedNeuronIndices);
    
    trainingProjWeights = updatedModelParameters.classify(timeWindowIndex).wLDA_kNN;
    pcaDimension        = updatedModelParameters.classify(timeWindowIndex).dPCA_kNN;
    ldaDimension        = updatedModelParameters.classify(timeWindowIndex).dLDA_kNN;
    optimalProjTrain    = updatedModelParameters.classify(timeWindowIndex).wOpt_kNN;
    trainingMeanFiring  = updatedModelParameters.classify(timeWindowIndex).mFire_kNN;
    
    testProjection = optimalProjTrain' * (processedFiringVector - trainingMeanFiring);
    
    % --- Confidence-Based Classification ---
    % Call a modified kNN that returns both a predicted label and a confidence measure.
    [predictedLabel, confidence] = getKNNs_confidence(testProjection, trainingProjWeights, ldaDimension, 20);
    threshold = 0.5;  % Example confidence threshold.
    if confidence < threshold
        % If confidence is low, retain the previously determined direction.
        predictedLabel = updatedModelParameters.actualLabel;
    else
        updatedModelParameters.actualLabel = predictedLabel;
    end
else
    predictedLabel = updatedModelParameters.actualLabel;
    timeWindowIndex = 1;
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);
    numNeurons = numNeurons - length(removedNeuronIndices);
end

% ---------------- PCR: Predicting Hand Position ----------------
polyDegree = updatedModelParameters.polyd;
if trialDuration <= 560
    numNeurons = size(smoothedTrial(1,1).rates, 1) - length(removedNeuronIndices);
    timeWindowIndex = (trialDuration / binSize) - (320 / binSize) + 1;
    
    averagePosX = updatedModelParameters.averages(timeWindowIndex).avX(:, predictedLabel);
    averagePosY = updatedModelParameters.averages(timeWindowIndex).avY(:, predictedLabel);
    meanFiringPCR = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).fMean;
    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).by;
    
    % Ensure firing vector has exactly 70 elements (same as regressionCoeffX)
    firingVector = processedFiringVector(1:length(regressionCoeffX));  
    
    % Expand features with polynomial terms but maintain 70 features
    polyFiringVector = zeros(size(firingVector)); % Initialize same size as firingVector
    for d = 1:polyDegree
        polyFiringVector = polyFiringVector + (firingVector - mean(meanFiringPCR)).^d;
    end

    % Predict position using polynomial regression
    predictedX = polyFiringVector' * regressionCoeffX + averagePosX;
    predictedY = polyFiringVector' * regressionCoeffY + averagePosY;
    
    try
        predictedX = predictedX(trialDuration, 1);
        predictedY = predictedY(trialDuration, 1);
    catch
        predictedX = predictedX(end, 1);
        predictedY = predictedY(end, 1);
    end
else

    numNeurons = size(smoothedTrial(1,1).rates, 1) - length(removedNeuronIndices);
    averagePosX = updatedModelParameters.averages(13).avX(:, predictedLabel);
    averagePosY = updatedModelParameters.averages(13).avY(:, predictedLabel);
    meanFiringPCR = updatedModelParameters.pcr(predictedLabel, 13).fMean;
    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, 13).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, 13).by;
    
    % Ensure firing vector has exactly 70 elements (same as regressionCoeffX)
    firingVector = processedFiringVector(1:length(regressionCoeffX));  

  % Expand features with polynomial terms but maintain 70 features
    polyFiringVector = zeros(size(firingVector)); % Initialize same size as firingVector
    for d = 1:polyDegree
        polyFiringVector = polyFiringVector + (firingVector - mean(meanFiringPCR)).^d;
    end
    
    % Predict position using polynomial regression
    predictedX = polyFiringVector' * regressionCoeffX + averagePosX;
    predictedY = polyFiringVector' * regressionCoeffY + averagePosY;
    
    try
        predictedX = predictedX(trialDuration, 1);
        predictedY = predictedY(trialDuration, 1);
    catch
        predictedX = predictedX(end, 1);
        predictedY = predictedY(end, 1);
    end
end


% --- Modified kNN function returning confidence ---
function [predictedLabel, confidence] = getKNNs_confidence(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
    trainingMatrix = trainingProjection';
    testingMatrix = testProjection;
    trainingSquared = sum(trainingMatrix .* trainingMatrix, 2);
    testingSquared  = sum(testingMatrix .* testingMatrix, 1);
    
    % Compute squared Euclidean distances.
    distanceMatrix = trainingSquared(:, ones(1, length(testingMatrix))) + ...
                     testingSquared(ones(1, length(trainingMatrix)), :) - ...
                     2 * trainingMatrix * testingMatrix;
    distanceMatrix = distanceMatrix';
    
    k = 20;  % Fixed number of neighbors.
    [~, sortedIndices] = sort(distanceMatrix, 2);
    nearestNeighbors = sortedIndices(:, 1:k);
    
    numTrialsPerDirection = size(trainingProjection, 2) / 8;
    directionLabels = [ones(1, numTrialsPerDirection), 2*ones(1, numTrialsPerDirection), ...
                       3*ones(1, numTrialsPerDirection), 4*ones(1, numTrialsPerDirection), ...
                       5*ones(1, numTrialsPerDirection), 6*ones(1, numTrialsPerDirection), ...
                       7*ones(1, numTrialsPerDirection), 8*ones(1, numTrialsPerDirection)]';
    nearestLabels = reshape(directionLabels(nearestNeighbors), [], k);
    
    % Determine the predicted label as the mode.
    predictedLabel = mode(mode(nearestLabels, 2));
    
    % Calculate confidence as the average fraction of neighbors voting for the predicted label.
    votes = sum(nearestLabels == predictedLabel, 2);
    confidence = mean(votes) / k;
end

% Note: The helper functions (binAndSqrtSpikes, computeFiringRates, getKNNs, etc.) remain unchanged.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Helper Functions

function processedTrials = binAndSqrtSpikes(rawTrialData, binSize, applySqrt)
% BINANDSQRTSPIKES Re-bins the spike data and applies a square-root transform.
%   This function converts high-resolution spike data into coarser bins and,
%   if requested, applies a square-root transformation to reduce the influence
%   of high-firing neurons.
%
%   Inputs:
%       rawTrialData - Structure containing the original spike data.
%       binSize      - New binning resolution (in ms).
%       applySqrt    - Boolean flag to apply square-root transform.
%
%   Output:
%       processedTrials - Structure with binned (and possibly transformed) spikes.

    processedTrials = struct;
    for col = 1:size(rawTrialData,2)
        for row = 1:size(rawTrialData,1)
            spikeMatrix = rawTrialData(row,col).spikes;  % Rows: neurons; Columns: time points
            [numNeurons, numTimePoints] = size(spikeMatrix);
            newBinEdges = 1:binSize:(numTimePoints+1);
            binnedSpikes = zeros(numNeurons, numel(newBinEdges)-1);
            for binIdx = 1:(numel(newBinEdges)-1)
                binnedSpikes(:,binIdx) = sum(spikeMatrix(:, newBinEdges(binIdx):newBinEdges(binIdx+1)-1), 2);
            end
            if applySqrt
                binnedSpikes = sqrt(binnedSpikes);
            end
            processedTrials(row,col).spikes = binnedSpikes;
        end
    end
end

function trialsWithRates = computeFiringRates(binnedTrials, binSize, gaussianScale)
% COMPUTEFIRINGRATES Smooths the binned spike data using a Gaussian kernel.
%   This function convolves each neuron's binned spike train with a Gaussian
%   kernel to obtain smoothed firing rates.
%
%   Inputs:
%       binnedTrials - Structure with binned spike data.
%       binSize      - Binning resolution (ms).
%       gaussianScale- Scale factor for the Gaussian kernel.
%
%   Output:
%       trialsWithRates - Structure with computed firing rates.

    trialsWithRates = struct;
    kernelWindowSize = 10 * (gaussianScale / binSize);
    normalizedStd = gaussianScale / binSize;
    alphaParam = (kernelWindowSize - 1) / (2 * normalizedStd);
    timeVector = -(kernelWindowSize-1)/2 : (kernelWindowSize-1)/2;
    gaussianTemp = exp((-1/2) * (alphaParam * timeVector/((kernelWindowSize-1)/2)).^2)';
    gaussianKernel = gaussianTemp / sum(gaussianTemp);
    
    for col = 1:size(binnedTrials,2)
        for row = 1:size(binnedTrials,1)
            [numNeurons, numBins] = size(binnedTrials(row,col).spikes);
            smoothedRates = zeros(numNeurons, numBins);
            for neuronIdx = 1:numNeurons
                smoothedRates(neuronIdx,:) = conv(binnedTrials(row,col).spikes(neuronIdx,:), gaussianKernel, 'same') / (binSize/1000);
            end
            trialsWithRates(row,col).rates = smoothedRates;
        end
    end
end

function predictedLabel = getKNNs(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
% GETKNNs Determines the reaching direction using a k-Nearest Neighbors approach.
%   This function computes the Euclidean distances between the test projection
%   and training projection data, then uses the kNN algorithm to assign a label.
%
%   Inputs:
%       testProjection     - Projection of test data (after PCA-LDA).
%       trainingProjection - Projection of training data (after PCA-LDA).
%       ldaDimension       - Number of LDA dimensions used.
%       neighborhoodFactor - Factor to adjust the number of nearest neighbors.
%
%   Output:
%       predictedLabel     - Predicted reaching direction label.

    trainingMatrix = trainingProjection';
    testingMatrix = testProjection;
    trainingSquared = sum(trainingMatrix .* trainingMatrix, 2);
    testingSquared = sum(testingMatrix .* testingMatrix, 1);
    
    % Compute the squared Euclidean distance between each test and training point.
    distanceMatrix = trainingSquared(:, ones(1, length(testingMatrix))) + ...
                     testingSquared(ones(1, length(trainingMatrix)), :) - ...
                     2 * trainingMatrix * testingMatrix;
    distanceMatrix = distanceMatrix';
    
    % Determine the k nearest neighbors.
    k = 25;  % Fixed number of neighbors.
    [~, sortedIndices] = sort(distanceMatrix, 2);
    nearestNeighbors = sortedIndices(:, 1:k);
    
    % Map training trials to their corresponding direction labels.
    numTrialsPerDirection = size(trainingProjection, 2) / 8;
    directionLabels = [ones(1, numTrialsPerDirection), 2*ones(1, numTrialsPerDirection), ...
                       3*ones(1, numTrialsPerDirection), 4*ones(1, numTrialsPerDirection), ...
                       5*ones(1, numTrialsPerDirection), 6*ones(1, numTrialsPerDirection), ...
                       7*ones(1, numTrialsPerDirection), 8*ones(1, numTrialsPerDirection)]';
    nearestLabels = reshape(directionLabels(nearestNeighbors), [], k);
    predictedLabel = mode(mode(nearestLabels, 2));
end
end


