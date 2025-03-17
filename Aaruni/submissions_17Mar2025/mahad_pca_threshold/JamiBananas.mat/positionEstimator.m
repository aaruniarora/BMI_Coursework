function [predictedX, predictedY, updatedModelParameters] = positionEstimator(testTrialData, trainedModelParameters)
% POSITIONESTIMATOR Decodes hand position from spike data using PCR,
% with an (optional) confidence-based update on the classifier output.
%
% Inputs:
%   testTrialData         - Structure containing spike data for the trial.
%   trainedModelParameters- Structure with model parameters from training.
%
% Outputs:
%   predictedX            - Decoded X position
%   predictedY            - Decoded Y position
%   updatedModelParameters- Possibly updated model parameters

% Copy the trained model parameters
updatedModelParameters = trainedModelParameters;

% ----------------------- Initialization -----------------------
numDirections  = 8;          % Number of possible movement directions
binSize        = 20;         % Binning resolution (ms)
gaussianScale  = 50;         % Scale for the smoothing kernel
targetAngles   = [30 70 110 150 190 230 310 350]; %#ok<NASGU> (not explicitly used here)

% Preprocess the test spike data
processedTrial = binAndTransformSpikes(testTrialData, binSize, true);
smoothedTrial  = smoothFiringRatesEMA(processedTrial, binSize, gaussianScale);

trialDuration  = size(testTrialData.spikes, 2);
numNeurons     = size(smoothedTrial(1,1).rates, 1);
lastbinsize    = updatedModelParameters.endBin;

% ------------------ Determine Reaching Direction ------------------
if trialDuration <= lastbinsize
    timeWindowIndex = (trialDuration/binSize) - (updatedModelParameters.startBin/binSize) + 1;
    
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);
    
    trainingProjWeights = updatedModelParameters.classify(timeWindowIndex).wLDA_kNN;
    pcaDimension        = updatedModelParameters.classify(timeWindowIndex).dPCA_kNN; %#ok<NASGU>
    ldaDimension        = updatedModelParameters.classify(timeWindowIndex).dLDA_kNN;
    optimalProjTrain    = updatedModelParameters.classify(timeWindowIndex).wOpt_kNN;
    trainingMeanFiring  = updatedModelParameters.classify(timeWindowIndex).mFire_kNN;
    
    testProjection = optimalProjTrain' * (processedFiringVector - trainingMeanFiring);
    
    % --- Confidence-Based Classification via KNN ---
    [predictedLabel, confidence] = knnPredictWithConfidence(testProjection, ...
        trainingProjWeights, ldaDimension, 8);
    
    threshold = 0;  % Example confidence threshold
    if confidence < threshold
        % If confidence is too low, revert to a previously determined direction
        predictedLabel = updatedModelParameters.actualLabel;
    else
        updatedModelParameters.actualLabel = predictedLabel;
    end
    
else
    predictedLabel   = updatedModelParameters.actualLabel;
    timeWindowIndex  = 1;  % fallback
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);
end

% ---------------- PCR: Predicting Hand Position ----------------
if trialDuration <= lastbinsize
    timeWindowIndex = (trialDuration/binSize) - (updatedModelParameters.startBin/binSize) + 1;
    
    averagePosX   = updatedModelParameters.averages(timeWindowIndex).avX(:, predictedLabel);
    averagePosY   = updatedModelParameters.averages(timeWindowIndex).avY(:, predictedLabel);
    meanFiringPCR = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).fMean;
    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).by;
    
    predictedX = (processedFiringVector - mean(meanFiringPCR))' * regressionCoeffX + averagePosX;
    predictedY = (processedFiringVector - mean(meanFiringPCR))' * regressionCoeffY + averagePosY;
    
    % Attempt to slice by trial duration if dimensioned that way
    try
        predictedX = predictedX(trialDuration, 1);
        predictedY = predictedY(trialDuration, 1);
    catch
        predictedX = predictedX(end, 1);
        predictedY = predictedY(end, 1);
    end
    
else
    averagePosX   = updatedModelParameters.averages(13).avX(:, predictedLabel);
    averagePosY   = updatedModelParameters.averages(13).avY(:, predictedLabel);
    meanFiringPCR = updatedModelParameters.pcr(predictedLabel, 13).fMean;
    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, 13).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, 13).by;
    
    predictedX = (processedFiringVector(1:length(regressionCoeffX)) - ...
                  mean(processedFiringVector(1:length(regressionCoeffX))))' * regressionCoeffX + averagePosX;
    predictedY = (processedFiringVector(1:length(regressionCoeffY)) - ...
                  mean(processedFiringVector(1:length(regressionCoeffY))))' * regressionCoeffY + averagePosY;
    
    try
        predictedX = predictedX(trialDuration, 1);
        predictedY = predictedY(trialDuration, 1);
    catch
        predictedX = predictedX(end, 1);
        predictedY = predictedY(end, 1);
    end
end

end


% --- Modified kNN function returning confidence ---
function [predictedLabel, confidence] = knnPredictWithConfidence(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
% KNN-based classification with confidence measure
%
% Inputs:
%   testProjection     - [D x 1] test data in the LDA subspace
%   trainingProjection - [D x N] training data in the LDA subspace
%   ldaDimension       - number of LDA dimensions (unused here except for clarity)
%   neighborhoodFactor - factor to choose k or other adjustments (unused directly)
%
% Outputs:
%   predictedLabel - integer label (1..8)
%   confidence     - fraction of neighbors that matched the predicted label

    trainingMatrix  = trainingProjection';  % N x D
    testingMatrix   = testProjection;       % D x 1
    trainingSquared = sum(trainingMatrix .* trainingMatrix, 2); % N x 1
    testingSquared  = sum(testingMatrix .* testingMatrix, 1);   % scalar
    
    distanceMatrix = trainingSquared + testingSquared - 2 * (trainingMatrix * testingMatrix);
    distanceMatrix = distanceMatrix';  % now 1 x N
    
    k = 20; % number of neighbors
    [~, sortedIndices] = sort(distanceMatrix, 2);
    nearestNeighbors   = sortedIndices(:, 1:k);
    
    numTrialsPerDirection = size(trainingProjection, 2) / 8;
    directionLabels = [ones(1, numTrialsPerDirection), 2*ones(1, numTrialsPerDirection), ...
                       3*ones(1, numTrialsPerDirection), 4*ones(1, numTrialsPerDirection), ...
                       5*ones(1, numTrialsPerDirection), 6*ones(1, numTrialsPerDirection), ...
                       7*ones(1, numTrialsPerDirection), 8*ones(1, numTrialsPerDirection)]';
    nearestLabels = directionLabels(nearestNeighbors);  % k labels
    predictedLabel = mode(nearestLabels);
    
    votes = sum(nearestLabels == predictedLabel);
    confidence = votes / k;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local Helper Functions (Renamed consistently)

function processedTrials = binAndTransformSpikes(rawTrials, binInterval, applySqrt)
% Re-bin spikes and optionally apply sqrt transform
    processedTrials = struct;

    for colIdx = 1:size(rawTrials,2)
        for rowIdx = 1:size(rawTrials,1)
            spikeMatrix = rawTrials(rowIdx, colIdx).spikes;  
            [numNeurons, numTimePoints] = size(spikeMatrix);

            newBinEdges = 1:binInterval:(numTimePoints+1);
            binnedSpikes = zeros(numNeurons, numel(newBinEdges)-1);

            for binIdx = 1:numel(newBinEdges)-1
                binnedSpikes(:, binIdx) = ...
                    sum(spikeMatrix(:, newBinEdges(binIdx):newBinEdges(binIdx+1)-1), 2);
            end
            if applySqrt
                binnedSpikes = sqrt(binnedSpikes);
            end

            processedTrials(rowIdx, colIdx).spikes   = binnedSpikes;
            if isfield(rawTrials(rowIdx, colIdx), 'handPos')
                processedTrials(rowIdx, colIdx).handPos = rawTrials(rowIdx, colIdx).handPos(1:2,:);
            else
                processedTrials(rowIdx, colIdx).handPos = [];
            end
            processedTrials(rowIdx, colIdx).bin_size = binInterval;
        end
    end
end

function trialsWithRates = smoothFiringRatesGaussian(binnedTrials, binSize, gaussianScale)
% Gaussian smoothing
    trialsWithRates       = struct;
    kernelWindowSize      = 10 * (gaussianScale / binSize);
    normalizedStd         = gaussianScale / binSize;
    alphaParam            = (kernelWindowSize - 1)/(2 * normalizedStd);
    timeVector            = -(kernelWindowSize-1)/2 : (kernelWindowSize-1)/2;
    gaussianTemp          = exp((-1/2) * (alphaParam * timeVector/((kernelWindowSize-1)/2)).^2)';
    gaussianKernel        = gaussianTemp / sum(gaussianTemp);

    for col = 1:size(binnedTrials,2)
        for row = 1:size(binnedTrials,1)
            [numNeurons, numBins] = size(binnedTrials(row,col).spikes);
            smoothedRates = zeros(numNeurons, numBins);
            for neuronIdx = 1:numNeurons
                smoothedRates(neuronIdx,:) = ...
                    conv(binnedTrials(row,col).spikes(neuronIdx,:), gaussianKernel, 'same') ...
                    / (binSize/1000);
            end
            trialsWithRates(row,col).rates = smoothedRates;
        end
    end
end

function [predictedLabel] = knnPredictSimple(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
% Basic kNN without confidence measure
    trainingMatrix = trainingProjection';
    testingMatrix  = testProjection;
    trainingSquared = sum(trainingMatrix .* trainingMatrix, 2);
    testingSquared  = sum(testingMatrix .* testingMatrix, 1);
    
    distanceMatrix = trainingSquared(:, ones(1, length(testingMatrix))) + ...
                     testingSquared(ones(1, length(trainingMatrix)), :) - ...
                     2 * trainingMatrix * testingMatrix;
    distanceMatrix = distanceMatrix';

    k = 25;
    [~, sortedIndices] = sort(distanceMatrix, 2);
    nearestNeighbors   = sortedIndices(:, 1:k);

    numTrialsPerDirection = size(trainingProjection, 2) / 8;
    directionLabels = [ones(1, numTrialsPerDirection), 2*ones(1, numTrialsPerDirection), ...
                       3*ones(1, numTrialsPerDirection), 4*ones(1, numTrialsPerDirection), ...
                       5*ones(1, numTrialsPerDirection), 6*ones(1, numTrialsPerDirection), ...
                       7*ones(1, numTrialsPerDirection), 8*ones(1, numTrialsPerDirection)]';
    nearestLabels  = reshape(directionLabels(nearestNeighbors), [], k);
    predictedLabel = mode(mode(nearestLabels, 2));
end

function [predictedLabel, confidence] = knnPredictConfidenceAlt1(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
% Alternate approach: nearest-centroid classification returning confidence
    numDirections = 8;
    numTrialsPerDirection = size(trainingProjection, 2) / numDirections;
    directionLabels = [ ...
        ones(1,numTrialsPerDirection), ...
        2*ones(1,numTrialsPerDirection), ...
        3*ones(1,numTrialsPerDirection), ...
        4*ones(1,numTrialsPerDirection), ...
        5*ones(1,numTrialsPerDirection), ...
        6*ones(1,numTrialsPerDirection), ...
        7*ones(1,numTrialsPerDirection), ...
        8*ones(1,numTrialsPerDirection) ...
    ];
    
    centroids = zeros(size(trainingProjection,1), numDirections);
    for dirIdx = 1:numDirections
        colsForThisDir = (directionLabels == dirIdx);
        centroids(:, dirIdx) = mean(trainingProjection(:, colsForThisDir), 2);
    end

    Ntest = size(testProjection, 2);
    dists = zeros(Ntest, numDirections);
    for iTest = 1:Ntest
        diffToCentroids = centroids - testProjection(:, iTest);
        dists(iTest,:)  = sum(diffToCentroids.^2, 1); 
    end
    
    [~, perSampleLabels] = min(dists, [], 2);
    predictedLabel = mode(perSampleLabels);
    votesForLabel  = sum(perSampleLabels == predictedLabel);
    confidence     = votesForLabel / Ntest;
end

function [predictedLabel, confidence] = knnPredictConfidenceAlt2(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
% Alternate approach: Gaussian classifier
    numDirections = 8;
    numTrialsPerDirection = size(trainingProjection,2) / numDirections;
    directionLabels = [ones(1, numTrialsPerDirection), 2*ones(1, numTrialsPerDirection), ...
                       3*ones(1, numTrialsPerDirection), 4*ones(1, numTrialsPerDirection), ...
                       5*ones(1, numTrialsPerDirection), 6*ones(1, numTrialsPerDirection), ...
                       7*ones(1, numTrialsPerDirection), 8*ones(1, numTrialsPerDirection)]';

    D = size(trainingProjection, 1);
    means = zeros(D, numDirections);
    covariances = zeros(D, D, numDirections);
    regVal = 1e-4;

    for dirIdx = 1:numDirections
        inds = find(directionLabels == dirIdx);
        classData = trainingProjection(:, inds);
        means(:, dirIdx) = mean(classData, 2);
        covMat = cov(classData');
        covMat = covMat + regVal*eye(D);
        covariances(:,:,dirIdx) = covMat;
    end
    
    Ntest = size(testProjection, 2);
    likelihoods = zeros(Ntest, numDirections);
    for i = 1:Ntest
        x = testProjection(:, i);
        for dirIdx = 1:numDirections
            mu = means(:, dirIdx);
            diff = x - mu;
            covMat = covariances(:,:,dirIdx);
            exponent = -0.5 * (diff' / covMat) * diff;
            normFactor = 1 / (((2*pi)^(D/2)) * sqrt(det(covMat)));
            likelihoods(i, dirIdx) = normFactor * exp(exponent);
        end
    end
    posterior = likelihoods ./ sum(likelihoods, 2);
    [~, sampleLabels] = max(posterior, [], 2);
    
    predictedLabel = mode(sampleLabels);
    confidence     = sum(sampleLabels == predictedLabel) / Ntest;
end

function finalTrials = smoothFiringRatesEMA(binnedTrials, binInterval, gaussianScale)
% Exponential moving average smoothing
    windowSize = 26; 
    if windowSize < 1, windowSize = 1; end
    alpha = 25 / (windowSize + 1);

    finalTrials = struct;
    for colIdx = 1:size(binnedTrials,2)
        for rowIdx = 1:size(binnedTrials,1)
            spikeMatrix = binnedTrials(rowIdx, colIdx).spikes;
            [numNeurons, numBins] = size(spikeMatrix);

            smoothedRates = zeros(numNeurons, numBins);
            for neuronIdx = 1:numNeurons
                rateTrain = spikeMatrix(neuronIdx, :) / (binInterval / 1000);
                smoothedRates(neuronIdx, 1) = rateTrain(1);
                for t = 2:numBins
                    smoothedRates(neuronIdx, t) = alpha * rateTrain(t) + ...
                        (1 - alpha) * smoothedRates(neuronIdx, t-1);
                end
            end

            finalTrials(rowIdx, colIdx).rates    = smoothedRates;
            finalTrials(rowIdx, colIdx).handPos  = binnedTrials(rowIdx, colIdx).handPos;
            finalTrials(rowIdx, colIdx).bin_size = binnedTrials(rowIdx, colIdx).bin_size;
        end
    end
end
