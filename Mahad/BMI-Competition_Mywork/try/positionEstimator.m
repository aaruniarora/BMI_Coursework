
function [predictedX, predictedY, updatedModelParameters] = positionEstimator(testTrialData, trainedModelParameters)
% POSITIONESTIMATOR Decodes hand position from spike data using PCR,
% with a confidence-based classification and direction-locking.
%
% Inputs:
%   testTrialData         - Structure containing spike data for the trial.
%                           (Should include .trialId if you want trial-based reset.)
%   trainedModelParameters- Structure with model parameters from training.
%
% Outputs:
%   predictedX            - Decoded X position.
%   predictedY            - Decoded Y position.
%   updatedModelParameters- (Possibly updated) model parameters that keep
%                           track of direction stability and trial context.
%
% NOTE: This version includes direction-locking logic: once a direction is
% identified, it tries to stay with that direction unless repeated mismatches
% or other confidence criteria force a reset.
%
% ---------------------------------------------------------------
% Copy the trained model parameters so we can update them safely:
updatedModelParameters = trainedModelParameters;

% ----------------------- Initialization -----------------------
numDirections      = 8;       % Number of possible movement directions.
binSize            = 20;      % Binning resolution (ms).
gaussianScale      = 50;      % Scale for the Gaussian (or used to define EMA alpha).
targetAngles       = [30 70 110 150 190 230 310 350]; %#ok<NASGU> (Unused here)
lastbinsize        = updatedModelParameters.endBin;   % e.g., 560 ms or similar

% A small helper to ensure these fields exist in updatedModelParameters
if ~isfield(updatedModelParameters, 'actualLabel') 
    updatedModelParameters.actualLabel = []; 
end
if ~isfield(updatedModelParameters, 'trial_id')
    updatedModelParameters.trial_id = 0; 
end

% ------------------ Check Trial Context (Optional) ------------------
% If your testTrialData includes a .trialId, we can track resets across trials
if isfield(testTrialData, 'trialId')
    if updatedModelParameters.trial_id == 0
        % First time calling for this new model
        updatedModelParameters.trial_id = testTrialData.trialId;
        updatedModelParameters.actualLabel = [];
    elseif updatedModelParameters.trial_id ~= testTrialData.trialId
        % A new trial started => reset direction history
        updatedModelParameters.trial_id    = testTrialData.trialId;
        updatedModelParameters.actualLabel = [];
    end
end

% ------------------ Preprocess the Test Spike Data ------------------
processedTrial = binAndSqrtSpikes(testTrialData, binSize, true);
smoothedTrial  = computeFiringRates1(processedTrial, binSize, gaussianScale);

trialDuration  = size(testTrialData.spikes, 2);  % # of ms in the test sample
numNeurons     = size(smoothedTrial(1,1).rates, 1);

% We often reference the time-window index by:
%   timeWindowIndex = (trialDuration / binSize) - (320 / binSize) + 1;
% (Assuming your training started at 320 ms, for example.)

% Make sure we clamp or protect the index so we don't go out of array bounds
timeWindowIndex = round((trialDuration / binSize) - (320 / binSize) + 1);
timeWindowIndex = max(1, min(timeWindowIndex, length(updatedModelParameters.classify)));

% (In your code, it's only valid if trialDuration <= lastbinsize, so adapt as needed.)
if trialDuration <= lastbinsize
    % 1) Remove low-firing neurons as done during training
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);
    
    % 2) Retrieve the classification parameters from training
    trainingProjWeights = updatedModelParameters.classify(timeWindowIndex).wLDA_kNN;
    optimalProjTrain    = updatedModelParameters.classify(timeWindowIndex).wOpt_kNN;
    trainingMeanFiring  = updatedModelParameters.classify(timeWindowIndex).mFire_kNN;
    
    % 3) Project the test vector into the LDA/kNN space
    testProjection = optimalProjTrain' * (processedFiringVector - trainingMeanFiring);
    
    % 4) Classify to get predicted direction label + confidence
    [predictedLabel, confidence] = getKNNs_confidence(testProjection, ...
        trainingProjWeights, updatedModelParameters.classify(timeWindowIndex).dLDA_kNN, 8);
    
    % 5) Confidence threshold check (optional)
    confidenceThreshold = 0.5;  % For example
    if confidence < confidenceThreshold && ~isempty(updatedModelParameters.actualLabel)
        % If the confidence is below threshold, keep the last stable label
        predictedLabel = updatedModelParameters.actualLabel(end);
    end
    
else
    % Once we exceed the last bin size, just stick to the most recent direction 
    % (or your fallback logic) without re-classifying:
    predictedLabel       = mode(updatedModelParameters.actualLabel); 
    removedNeuronIndices = updatedModelParameters.lowFirers{1};
    smoothedTrial.rates(removedNeuronIndices, :) = [];
    processedFiringVector = reshape(smoothedTrial.rates, [], 1);
end

% ------------------ Direction-Locking / Blocking Logic ------------------
% Incorporate logic to prevent abrupt changes or repeated mismatches.

if ~isempty(updatedModelParameters.actualLabel)
    % 1) If the new predictedLabel differs from our last stable label
    lastStableLabel = updatedModelParameters.actualLabel(end);
    if lastStableLabel ~= predictedLabel
        % We have a mismatch => Check how often we've seen mismatches recently
        historyLen = length(updatedModelParameters.actualLabel);
        if historyLen > 10
            % Count how many times the last 5 bins differ from the new label
            recentWindowSize = 5;
            startIdx = max(1, historyLen - (recentWindowSize - 1));
            mismatchCount = sum(updatedModelParameters.actualLabel(startIdx:end) ~= predictedLabel);
            
            if mismatchCount >= 3
                % If at least 3 out of 5 are mismatches, we reset 
                % (or you can degrade to keep the old label, etc.)
                updatedModelParameters.actualLabel = [];
            end
        end
    end
end

% Now, either we have an empty actualLabel, or one with potential new label appended
if ~isempty(updatedModelParameters.actualLabel)
    % Possibly wait for a certain length of classification before "locking"
    if length(updatedModelParameters.actualLabel) > 6
        % Force the label to be the majority vote if enough samples exist
        predictedLabel = mode(updatedModelParameters.actualLabel);
    end
    
    % Append the new label
    updatedModelParameters.actualLabel(end+1) = predictedLabel;
    
    % Optionally force them all to match => "lock" the direction
    updatedModelParameters.actualLabel(:) = predictedLabel;
else
    % If there's no existing label history, just set the current predictedLabel
    updatedModelParameters.actualLabel(end+1) = predictedLabel;
end

% ---------------------- PCR: Predicting Hand Position -------------------
% (Same logic as before, using the predictedLabel you just locked in.)

if trialDuration <= lastbinsize
    % (Recompute timeWindowIndex in case it changed)
    timeWindowIndex = round((trialDuration/binSize) - (320/binSize) + 1);
    timeWindowIndex = max(1, min(timeWindowIndex, length(updatedModelParameters.classify)));
    
    averagePosX      = updatedModelParameters.averages(timeWindowIndex).avX(:, predictedLabel);
    averagePosY      = updatedModelParameters.averages(timeWindowIndex).avY(:, predictedLabel);
    meanFiringPCR    = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).fMean;
    regressionCoeffX = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).bx;
    regressionCoeffY = updatedModelParameters.pcr(predictedLabel, timeWindowIndex).by;

    predictedX = (processedFiringVector - meanFiringPCR)' * regressionCoeffX + averagePosX;
    predictedY = (processedFiringVector - meanFiringPCR)' * regressionCoeffY + averagePosY;

    try
        predictedX = predictedX(trialDuration, 1);
        predictedY = predictedY(trialDuration, 1);
    catch
        % If that specific index doesn't exist, fallback to the last one
        predictedX = predictedX(end, 1);
        predictedY = predictedY(end, 1);
    end
    
else
    % Beyond lastbinsize, use the final bin's PCR parameters
    finalBinIndex     = 13;  % or however many bins you had total (like 560/20=28)
    averagePosX       = updatedModelParameters.averages(finalBinIndex).avX(:, predictedLabel);
    averagePosY       = updatedModelParameters.averages(finalBinIndex).avY(:, predictedLabel);
    meanFiringPCR     = updatedModelParameters.pcr(predictedLabel, finalBinIndex).fMean;
    regressionCoeffX  = updatedModelParameters.pcr(predictedLabel, finalBinIndex).bx;
    regressionCoeffY  = updatedModelParameters.pcr(predictedLabel, finalBinIndex).by;
    
    fvLen = length(regressionCoeffX);  % e.g. how many neurons * bins in the regression
    
    predictedX = (processedFiringVector(1:fvLen) - mean(processedFiringVector(1:fvLen)))' ...
        * regressionCoeffX + averagePosX;
    predictedY = (processedFiringVector(1:fvLen) - mean(processedFiringVector(1:fvLen)))' ...
        * regressionCoeffY + averagePosY;
    
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


function [predictedLabel, confidence] = getKNNs_confidence1(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
    % Replace kNN with a Nearest-Centroid approach.
    % Same inputs/outputs so it can directly replace your existing kNN code.
    %
    % Inputs:
    %   testProjection     - [D x Ntest] matrix: columns are test samples in LDA space.
    %   trainingProjection - [D x Ntrain] matrix: columns are training samples in LDA space.
    %   ldaDimension       - Not used here, but preserved for signature consistency.
    %   neighborhoodFactor - Not used here, but preserved for signature consistency.
    %
    % Outputs:
    %   predictedLabel  - Single integer label (1..8).
    %   confidence      - Single scalar in [0,1].
    %
    % -----------------------------------------------------------------------

    % Number of directions
    numDirections = 8;
    
    % Count how many total training samples there are for each direction:
    numTrialsPerDirection = size(trainingProjection, 2) / numDirections;
    
    % Build direction labels for each training sample [1..8].
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
    
    % Compute the centroid for each direction (mean over columns that belong to that direction).
    % trainingProjection is D x Ntrain. We'll gather columns belonging to each direction
    % and compute mean across them.
    centroids = zeros(size(trainingProjection,1), numDirections);  % (D x 8)
    for dirIdx = 1:numDirections
        colsForThisDir = (directionLabels == dirIdx);
        centroids(:, dirIdx) = mean(trainingProjection(:, colsForThisDir), 2);
    end

    % testProjection can have multiple columns (multiple test samples).
    % We'll classify each column (test sample) to the nearest centroid.
    % Then aggregate a single label + confidence in the same scalar form 
    % as the existing kNN code (which ends up returning one label/confidence).
    %
    % If you truly only ever call this with one test sample at a time, 
    % this loop effectively does a single pass anyway.
    %
    % Distances to centroids: (Ntest x 8)
    Ntest = size(testProjection, 2);
    dists = zeros(Ntest, numDirections);
    for iTest = 1:Ntest
        diffToCentroids = centroids - testProjection(:, iTest);
        dists(iTest,:) = sum(diffToCentroids.^2, 1);  % Euclidean^2 distance
    end
    
    % For each test sample, pick the class (direction) with min distance:
    [~, perSampleLabels] = min(dists, [], 2);  % Ntest x 1 integer labels
    
    % Just like your kNN code does 'mode(mode(...))', we reduce multiple test samples
    % to a single final label: 
    predictedLabel = mode(perSampleLabels);
    
    % A simple "confidence" measure: fraction of test samples that voted for predictedLabel.
    % This mimics how the kNN version lumps multiple test samples into a single label/confidence.
    votesForLabel = sum(perSampleLabels == predictedLabel);
    confidence = votesForLabel / Ntest;
end

function [predictedLabel, confidence] = getKNNs_confidence2(testProjection, trainingProjection, ldaDimension, neighborhoodFactor)
    % Gaussian Classifier Replacement for kNN
    % This function uses a class-conditional multivariate Gaussian model to
    % assign a label to the test projection data. It has the same input/output
    % interface as the original kNN function.
    %
    % Inputs:
    %   testProjection     - [D x Ntest] matrix: test samples in LDA space.
    %   trainingProjection - [D x Ntrain] matrix: training samples in LDA space.
    %   ldaDimension       - Not directly used, but kept for compatibility.
    %   neighborhoodFactor - Not used here, but kept for compatibility.
    %
    % Outputs:
    %   predictedLabel - Scalar integer label (1 to 8).
    %   confidence     - Scalar in [0,1] representing the fraction of test
    %                    samples voting for predictedLabel.
    
    % Define number of classes (directions) and build labels for training data.
    numDirections = 8;
    numTrialsPerDirection = size(trainingProjection,2) / numDirections;
    directionLabels = [ones(1, numTrialsPerDirection), 2*ones(1, numTrialsPerDirection), ...
                       3*ones(1, numTrialsPerDirection), 4*ones(1, numTrialsPerDirection), ...
                       5*ones(1, numTrialsPerDirection), 6*ones(1, numTrialsPerDirection), ...
                       7*ones(1, numTrialsPerDirection), 8*ones(1, numTrialsPerDirection)]';
    
    % Precompute class means and covariance matrices.
    D = size(trainingProjection, 1);  % Dimensionality (should equal ldaDimension)
    means = zeros(D, numDirections);
    covariances = zeros(D, D, numDirections);
    regularization = 1e-4;  % Small value to regularize covariance matrices
    
    for dirIdx = 1:numDirections
        indices = find(directionLabels == dirIdx);
        classData = trainingProjection(:, indices);  % D x numTrialsPerDirection
        means(:, dirIdx) = mean(classData, 2);
        % Compute covariance with observations as rows (so transpose classData)
        covMat = cov(classData');  % D x D
        % Regularize to avoid singularity issues.
        covMat = covMat + regularization * eye(D);
        covariances(:,:,dirIdx) = covMat;
    end
    
    % For each test sample, compute the likelihood under each Gaussian model.
    Ntest = size(testProjection, 2);
    likelihoods = zeros(Ntest, numDirections);
    for i = 1:Ntest
        x = testProjection(:, i);
        for dirIdx = 1:numDirections
            mu = means(:, dirIdx);
            covMat = covariances(:,:,dirIdx);
            % Compute Mahalanobis distance and corresponding likelihood.
            diff = x - mu;
            exponent = -0.5 * (diff' / covMat) * diff;
            normFactor = 1 / (((2*pi)^(D/2)) * sqrt(det(covMat)));
            likelihoods(i, dirIdx) = normFactor * exp(exponent);
        end
    end
    
    % Compute posterior probabilities assuming equal class priors.
    posterior = likelihoods ./ sum(likelihoods, 2);
    
    % Assign each test sample to the class with the highest posterior probability.
    [~, sampleLabels] = max(posterior, [], 2);
    
    % Use mode over test samples to decide the final label.
    predictedLabel = mode(sampleLabels);
    
    % Confidence is the fraction of test samples that voted for the predicted label.
    confidence = sum(sampleLabels == predictedLabel) / Ntest;
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
function finalTrials = computeFiringRates1(binnedTrials, binInterval, gaussianScale)

    % One way to link 'gaussianScale' to the EMA alpha is by reusing the same
    % window size notion from the old code. For example:
    %   windowSize = 10 * (gaussianScale / binInterval);
    % Then we pick a standard formula for alpha, e.g., alpha = 2 / (windowSize + 1)
    %
    % The exact relationship between gaussianScale and alpha is up to you. 
    % Below is one simple interpretation that keeps bigger 'gaussianScale' 
    % implying heavier (slower) smoothing.

    windowSize = 26;   
    if windowSize < 1, windowSize = 1; end             % guard from degenerate cases
    alpha = 25/(windowSize+1) ;                  % a standard EMA formula

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


