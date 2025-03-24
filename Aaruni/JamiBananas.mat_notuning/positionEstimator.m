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
binSize = 20;            % Binning resolution (ms).
targetAngles = [30 70 110 150 190 230 310 350];

% Preprocess the test spike data.
processedTrial = binAndSqrtSpikes(testTrialData, binSize, true);
smoothedTrial = computeFiringRates1(processedTrial, binSize);

trialDuration = size(testTrialData.spikes, 2);
numNeurons = size(smoothedTrial(1,1).rates, 1);
lastbinsize = trainedModelParameters.endBin;

% ------------------ Determine Reaching Direction ------------------
if trialDuration <= lastbinsize
    timeWindowIndex = (trialDuration/binSize) - (320/binSize)+1;
    
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
    [predictedLabel, confidence] = getKNNs_confidence(testProjection, trainingProjWeights);
    threshold = 0;  % Example confidence threshold.
    if confidence < threshold
        % If confidence is low, retain the previously determined direction.
        predictedLabel = updatedModelParameters.actualLabel;
    else
        updatedModelParameters.actualLabel = predictedLabel;
    end
else
    predictedLabel = updatedModelParameters.actualLabel;
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);

end

% ---------------- PCR: Predicting Hand Position ----------------
if trialDuration <= lastbinsize
    timeWindowIndex = (trialDuration/binSize) - (320/binSize) + 1;

    averagePosX = updatedModelParameters.averages(timeWindowIndex).avX(:, predictedLabel);
    averagePosY = updatedModelParameters.averages(timeWindowIndex).avY(:, predictedLabel);
    meanFiringPCR = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).fMean;
    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).by;

    predictedX = (processedFiringVector - mean(meanFiringPCR))' * regressionCoeffX + averagePosX;
    predictedY = (processedFiringVector - mean(meanFiringPCR))' * regressionCoeffY + averagePosY;
    
    try
        predictedX = predictedX(trialDuration, 1);
        predictedY = predictedY(trialDuration, 1);
    catch
        predictedX = predictedX(end, 1);
        predictedY = predictedY(end, 1);
    end
else

    averagePosX = updatedModelParameters.averages(13).avX(:, predictedLabel);
    averagePosY = updatedModelParameters.averages(13).avY(:, predictedLabel);

    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, 13).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, 13).by;
    
    predictedX = (processedFiringVector(1:length(regressionCoeffX)) - mean(processedFiringVector(1:length(regressionCoeffX))))' * regressionCoeffX + averagePosX;
    predictedY = (processedFiringVector(1:length(regressionCoeffY)) - mean(processedFiringVector(1:length(regressionCoeffY))))' * regressionCoeffY + averagePosY;
    
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
function [predictedLabel, confidence] = getKNNs_confidence(testProjection, trainingProjection)
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

function processedTrials = binAndSqrtSpikes(rawTrials, binInterval, applySqrt)
    processedTrials = struct;

    for colIdx = 1:size(rawTrials,2)
        for rowIdx = 1:size(rawTrials,1)
            spikeMatrix = rawTrials(rowIdx, colIdx).spikes;  % [98 x Time]
            [numNeurons, numTimePoints] = size(spikeMatrix);

            newBinEdges = 1:binInterval:numTimePoints+1;
            binnedSpikes = zeros(numNeurons, numel(newBinEdges)-1);

            for binIdx = 1:numel(newBinEdges)-1
                % Sum the spikes within each bin
                binnedSpikes(:, binIdx) = ...
                    sum(spikeMatrix(:, newBinEdges(binIdx):newBinEdges(binIdx+1)-1), 2);
            end

            % Optional sqrt transform
            if applySqrt
                binnedSpikes = sqrt(binnedSpikes);
            end

            % Store back into the structure
            processedTrials(rowIdx, colIdx).spikes   = binnedSpikes;

            % IMPORTANT: also store handPos, so that computeFiringRates1 doesn't fail
            if isfield(rawTrials(rowIdx, colIdx), 'handPos')
                % If you only need 2D, slice the first two rows
                processedTrials(rowIdx, colIdx).handPos = rawTrials(rowIdx, colIdx).handPos(1:2,:);
            else
                processedTrials(rowIdx, colIdx).handPos = [];
            end

            % Store bin_size if your code references it downstream
            processedTrials(rowIdx, colIdx).bin_size = binInterval;
        end
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: computeFiringRates
% This function computes smoothed firing rates using an EMA kernel instead
% of a Gaussian kernel.
%
% Inputs:
%   binnedTrials - structure with binned spike data
%   binInterval  - bin size (in ms)
%   gaussianScale- scale factor (used for backward-compatibility; here,
%                  we reinterpret or repurpose it for EMA smoothing)
%
% Output:
%   finalTrials  - structure with the same fields as in the original
%                  computeFiringRates function, but with EMA-smoothed rates.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function finalTrials = computeFiringRates1(binnedTrials, binInterval)

    % One way to link 'gaussianScale' to the EMA alpha is by reusing the same
    % window size notion from the old code. For example:
    %   windowSize = 10 * (gaussianScale / binInterval);
    % Then we pick a standard formula for alpha, e.g., alpha = 2 / (windowSize + 1)
    %
    % The exact relationship between gaussianScale and alpha is up to you. 
    % Below is one simple interpretation that keeps bigger 'gaussianScale' 
    % implying heavier (slower) smoothing.

    windowSize = 26;   % same "10 * scale/bin" logic
    if windowSize < 1, windowSize = 1; end             % guard from degenerate cases
    alpha = 25 / (windowSize + 1);                      % a standard EMA formula

    finalTrials = struct;
    for colIdx = 1:size(binnedTrials,2)
        for rowIdx = 1:size(binnedTrials,1)

            % Get the binned spike matrix for this trial:
            %   rows: neurons
            %   cols: time bins (already aggregated by binSize ms)
            spikeMatrix = binnedTrials(rowIdx, colIdx).spikes;
            [numNeurons, numBins] = size(spikeMatrix);

            % Pre-allocate space for the smoothed rates
            smoothedRates = zeros(numNeurons, numBins);

            % For each neuron, run an EMA over time bins
            % and convert spike counts to firing rate (spikes/sec)
            % The factor (binInterval/1000) is used to convert from "counts per bin"
            % to "spikes per second" (i.e., Hz), as in your original Gaussian code.
            for neuronIdx = 1:numNeurons
                % Convert to instantaneous rate in each bin
                rateTrain = spikeMatrix(neuronIdx, :) / (binInterval / 1000);

                % Initialize the first time bin
                smoothedRates(neuronIdx, 1) = rateTrain(1);

                % Run the forward-pass exponential smoothing
                for t = 2:numBins
                    smoothedRates(neuronIdx, t) = ...
                        alpha * rateTrain(t) + (1 - alpha) * smoothedRates(neuronIdx, t-1);
                end
            end

            % Store results in the same output format as before
            finalTrials(rowIdx, colIdx).rates    = smoothedRates;
            finalTrials(rowIdx, colIdx).handPos  = binnedTrials(rowIdx, colIdx).handPos;
            finalTrials(rowIdx, colIdx).bin_size = binnedTrials(rowIdx, colIdx).bin_size;
        end
    end
end

