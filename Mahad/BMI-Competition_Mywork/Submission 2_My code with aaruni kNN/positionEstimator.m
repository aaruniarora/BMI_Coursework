function [predictedX, predictedY, updatedModelParameters] = positionEstimator(testTrialData, trainedModelParameters)
% POSITIONESTIMATOR Decodes hand position from spike data using PCR.
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
targetAngles   = [30 70 110 150 190 230 310 350]; 
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
    optimalProjTrain    = updatedModelParameters.classify(timeWindowIndex).wOpt_kNN;
    trainingMeanFiring  = updatedModelParameters.classify(timeWindowIndex).mFire_kNN;
    
    testProjection = optimalProjTrain' * (processedFiringVector - trainingMeanFiring);
    
    % --- New kNN classification without confidence ---
    % Using NN_num=8, pow=2, alp=1, method 'soft', and type 'dist'
    predictedLabel = KNN_classifier(testProjection, trainingProjWeights, 8, 2, 1, 'soft', 'dist');
    updatedModelParameters.actualLabel = predictedLabel;
    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New kNN function (Replaces previous confidence-based kNN)
function [output_lbl] = KNN_classifier(test_weight, train_weight, NN_num, pow, alp, method, type)
% KNN_classifier: kNN classification without confidence calculation.
%
% Inputs:
%  test_weight: [projDim x #TestSamples] - LDA-projected test sample(s)
%  train_weight: [projDim x #TrainSamples] - LDA-projected training samples
%  NN_num: Determines k as k = trainlen / NN_num (trainlen per angle)
%  pow: Exponent for distance-based weighting (if type is 'dist')
%  alp: Parameter for exponential weighting (if type is 'exp')
%  method: 'hard' or 'soft'. Here, 'soft' is used for distance-weighted kNN.
%  type: 'dist' for inverse-distance weighting or 'exp' for exponential weighting.
%
% Output:
%  output_lbl: Predicted direction label for each test sample

    if strcmp(method, 'hard')
        trainlen = size(train_weight, 2) / 8;
        k = max(1, round(trainlen / NN_num));
    
        output_lbl = zeros(1, size(test_weight, 2));
    
        for i = 1:size(test_weight, 2)
            distances = sum((train_weight - test_weight(:, i)).^2, 1);
            [~, indices] = sort(distances, 'ascend');
            nearestIndices = indices(1:k);
        
            trainLabels = ceil(nearestIndices / trainlen);
            modeLabel = mode(trainLabels);
            output_lbl(i) = modeLabel;
        end
    end
    if strcmp(method, 'soft')
        % Distance-weighted kNN
        nAngles = 8;  % 8 reaching angles
        trainlen = size(train_weight, 2) / nAngles;
        k = max(1, round(trainlen / NN_num));
    
        output_lbl = zeros(1, size(test_weight, 2));
    
        for i = 1:size(test_weight, 2)
            distances = sum((train_weight - test_weight(:, i)).^2, 1);
    
            % Sort and get top-k nearest neighbors
            [sortedDist, sortedIdx] = sort(distances, 'ascend');
            nearestIdx  = sortedIdx(1:k);
            nearestDist = sortedDist(1:k);
    
            % Convert index -> direction label
            trainLabels = ceil(nearestIdx / trainlen);  % labels from 1..8
    
            % Compute distance-based weights
            if strcmp(type, 'dist')
                weights = 1 ./ (nearestDist.^pow + eps);
            end
            if strcmp(type, 'exp')
                weights = exp(-alp .* nearestDist);
            end
    
            % Sum up weights for each angle
            angleWeights = zeros(1, nAngles);
            for nn = 1:k
                angle = trainLabels(nn);
                angleWeights(angle) = angleWeights(angle) + weights(nn);
            end
            
            % Final predicted label is the angle with the highest sum of weights
            [~, bestAngle] = max(angleWeights);
            output_lbl(i) = bestAngle;
        end
    end
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
