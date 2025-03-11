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
    [predictedLabel, confidence] = getKNNs_confidence(testProjection, trainingProjWeights, ldaDimension, 8);
    threshold = 0;  % Example confidence threshold.
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
if trialDuration <= lastbinsize
    numNeurons = size(smoothedTrial(1,1).rates, 1) - length(removedNeuronIndices);
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
    numNeurons = size(smoothedTrial(1,1).rates, 1) - length(removedNeuronIndices);
    averagePosX = updatedModelParameters.averages(13).avX(:, predictedLabel);
    averagePosY = updatedModelParameters.averages(13).avY(:, predictedLabel);
    meanFiringPCR = updatedModelParameters.pcr(predictedLabel, 13).fMean;
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

