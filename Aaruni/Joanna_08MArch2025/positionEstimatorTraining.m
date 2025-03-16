function [modelParameters, aggregatedFiringRates] = positionEstimatorTraining(trainingData)
% This function trains a position estimator model from neural spike data.
% It preprocesses the training data, extracts firing rates, performs PCA and LDA,
% and computes regression coefficients for position estimation.
%
% Input:
%   trainingData - structure containing spike and hand position information
%
% Outputs:
%   modelParameters      - structure with classifier and regression
%   parameters
%   aggregatedFiringRates - aggregated firing rates used in model training

% ------------------------- Initialization -------------------------
numDirections = 8;                    % number of movement directions
timeBin = 20;                         % binning interval (in ms)
gaussianScale = 50;                   % scaling factor for Gaussian smoothing
numTrainingTrials = length(trainingData); % total number of training trials

% Preprocess spike data: binning and square-root transformation
binnedTrials = binAndSqrtSpikes(trainingData, timeBin, true);
% Smooth binned data with a Gaussian kernel to obtain firing rates
smoothedTrials = computeFiringRates(binnedTrials, timeBin, gaussianScale);
targetAnglesDeg = [30 70 110 150 190 230 310 350];  % target directions (degrees)

% Initialize parameters for classifier training
modelParameters = struct;
startBin = 320;
% ------------------------- Initialization -------------------------
numDirections     = 8;       % number of movement directions
timeBin           = 20;      % binning interval (in ms)
gaussianScale     = 50;      % scaling factor for Gaussian smoothing
numTrainingTrials = length(trainingData); % total number of training trials

% Determine the minimum spike length across all trials to ensure we don't exceed array bounds.
minTime = inf;
for tr = 1:numTrainingTrials
    for direc = 1:size(trainingData, 2)
        currentLen = size(trainingData(tr, direc).spikes, 2);
        if currentLen < minTime
            minTime = currentLen;
        end
    end
end

% Compute dynamic endBin based on the minimum spike length.
startBin = 320;  
endBin = floor((minTime - startBin) / timeBin) * timeBin + startBin;
modelParameters.startBin = startBin;
modelParameters.endBin   = endBin;


% Now, proceed with binning and smoothing using the computed endBin.
binnedTrials = binAndSqrtSpikes(trainingData, timeBin, true);
smoothedTrials = computeFiringRates(binnedTrials, timeBin, gaussianScale);

modelIndex = 1;
timeBinLimits = ([startBin:timeBin:endBin] / timeBin);
removedNeurons = {};

% ------------------ Aggregating Firing Rates ------------------
% Aggregate firing rate matrices from each trial and direction.
numNeurons = size(smoothedTrials(1,1).rates, 1);
for directionIdx = 1:numDirections
    for trialIdx = 1:numTrainingTrials
        for timeIdx = 1:(endBin/timeBin)
            rowStart = numNeurons*(timeIdx-1)+1;
            rowEnd = numNeurons*timeIdx;
            colIndex = numTrainingTrials*(directionIdx-1)+trialIdx;
            aggregatedFiringRates(rowStart:rowEnd, colIndex) = smoothedTrials(trialIdx, directionIdx).rates(:, timeIdx);
        end
    end
end

% Remove neurons with very low average firing rate for numerical stability.
lowFiringNeurons = [];
for neuronIdx = 1:numNeurons
    avgFiringRate = mean(mean(aggregatedFiringRates(neuronIdx:numNeurons:end, :)));
    if avgFiringRate < 0.1
        lowFiringNeurons = [lowFiringNeurons, neuronIdx];
    end
end
clear aggregatedFiringRates
removedNeurons{end+1} = lowFiringNeurons;
modelParameters.lowFirers = removedNeurons;

% --------- Build Classifier Parameters for Varying Time Windows ---------
for currentTimeBinLimit = timeBinLimits
    numNeurons = size(smoothedTrials(1,1).rates, 1);
    % Re-aggregate firing rates for the current time window limit.
    for directionIdx = 1:numDirections
        for trialIdx = 1:numTrainingTrials
            for timeIdx = 1:currentTimeBinLimit
                rowStart = numNeurons*(timeIdx-1)+1;
                rowEnd = numNeurons*timeIdx;
                colIndex = numTrainingTrials*(directionIdx-1)+trialIdx;
                aggregatedFiringRates(rowStart:rowEnd, colIndex) = smoothedTrials(trialIdx, directionIdx).rates(:, timeIdx);
            end
        end
    end
    
    % Remove data from neurons with low firing rates.
    removalIndices = [];
    for neuronIdx = lowFiringNeurons
        removalIndices = [removalIndices, neuronIdx:numNeurons:length(aggregatedFiringRates)];
    end
    aggregatedFiringRates(removalIndices, :) = [];
    numNeurons = length(aggregatedFiringRates) / (endBin/timeBin);
    
    % Create labels for each direction (for supervised LDA)
    directionLabels = [ones(1, numTrainingTrials), 2*ones(1, numTrainingTrials), ...
                       3*ones(1, numTrainingTrials), 4*ones(1, numTrainingTrials), ...
                       5*ones(1, numTrainingTrials), 6*ones(1, numTrainingTrials), ...
                       7*ones(1, numTrainingTrials), 8*ones(1, numTrainingTrials)];
    
    % --------------------- Principal Component Analysis ---------------------
    [principalComponents, eigenValues] = svdPCA(aggregatedFiringRates);
 
    % Compute class means for each direction.
    classMeanMatrix = zeros(size(aggregatedFiringRates,1), numDirections);
    for directionIdx = 1:numDirections
        classMeanMatrix(:,directionIdx) = mean(aggregatedFiringRates(:, numTrainingTrials*(directionIdx-1)+1:numTrainingTrials*directionIdx), 2);
    end
    overallMean = mean(aggregatedFiringRates, 2);
    betweenClassScatter = (classMeanMatrix - overallMean) * (classMeanMatrix - overallMean)';
    totalScatter = (aggregatedFiringRates - overallMean) * (aggregatedFiringRates - overallMean)';
    withinClassScatter = totalScatter - betweenClassScatter;
    
    % Set reduction dimensions (arbitrary values for now)
    numPCADimensions = 35;
    numLDADimensions = 4;
    
    % ----------------- Linear Discriminant Analysis (LDA) -----------------
    ldaMatrix = (principalComponents(:,1:numPCADimensions)' * withinClassScatter * principalComponents(:,1:numPCADimensions)) \ ...
                (principalComponents(:,1:numPCADimensions)' * betweenClassScatter * principalComponents(:,1:numPCADimensions));
    [ldaEigenVectors, ldaEigenValues] = eig(ldaMatrix);
    [~, ldaSortIndices] = sort(diag(ldaEigenValues), 'descend');
    optimalProjection = principalComponents(:,1:numPCADimensions) * ldaEigenVectors(:, ldaSortIndices(1:numLDADimensions));
    projectedDataWeights = optimalProjection' * (aggregatedFiringRates - overallMean);
    
    %figure;
    %plot(projectedDataWeights)

    % Store classifier parameters for this time window.
    modelParameters.classify(modelIndex).wLDA_kNN = projectedDataWeights;
    modelParameters.classify(modelIndex).dPCA_kNN = numPCADimensions;
    modelParameters.classify(modelIndex).dLDA_kNN = numLDADimensions;
    modelParameters.classify(modelIndex).wOpt_kNN = optimalProjection;
    modelParameters.classify(modelIndex).mFire_kNN = overallMean;
    modelIndex = modelIndex + 1;
end

% --------------------- Principal Components Regression (PCR) ---------------------
[meanPosX, meanPosY, resampledPosX, resampledPosY] = getPaddedAndResampledPositions(trainingData, numDirections, numTrainingTrials, timeBin);
xTestIntervals = resampledPosX(:, [startBin:timeBin:endBin]/timeBin, :);
yTestIntervals = resampledPosY(:, [startBin:timeBin:endBin]/timeBin, :);

% Create a time vector corresponding to the firing data
timeBinsForFiring = repelem(timeBin:timeBin:endBin, numNeurons);
testingTimeBins = startBin:timeBin:endBin;

% Compute PCR regression coefficients for each direction and time window.
for directionIdx = 1:numDirections
    posXDirection = squeeze(xTestIntervals(:,:,directionIdx));
    posYDirection = squeeze(yTestIntervals(:,:,directionIdx));
    
    numTimeWindows = ((endBin - startBin) / timeBin) + 1;
    for timeWindowIdx = 1:numTimeWindows
        % Demean the position data at the current time window.
        demeanedX = posXDirection(:, timeWindowIdx) - mean(posXDirection(:, timeWindowIdx));
        demeanedY = posYDirection(:, timeWindowIdx) - mean(posYDirection(:, timeWindowIdx));
        
        % Extract firing rates corresponding to the current time window and direction.
        windowedFiringRates = aggregatedFiringRates(timeBinsForFiring <= testingTimeBins(timeWindowIdx), directionLabels == directionIdx);
        %[pcaEigenVectors, ~] = performPCA(windowedFiringRates);
        [pcaEigenVectors, ~] = svdPCA(windowedFiringRates);
        %[pcaEigenVectors, ~] = covPCA(windowedFiringRates);
        
        % Project the windowed firing data onto principal components.
        Z = pcaEigenVectors(:, 1:numPCADimensions)' * (windowedFiringRates - mean(windowedFiringRates, 1));
        
        % Compute regression coefficients for the X and Y positions.
        Bx = (pcaEigenVectors(:,1:numPCADimensions) * inv(Z*Z') * Z) * demeanedX;
        By = (pcaEigenVectors(:,1:numPCADimensions) * inv(Z*Z') * Z) * demeanedY;
        
        % Store PCR regression coefficients and mean firing rates.
        modelParameters.pcr(directionIdx, timeWindowIdx).bx = Bx;
        modelParameters.pcr(directionIdx, timeWindowIdx).by = By;
        modelParameters.pcr(directionIdx, timeWindowIdx).fMean = mean(windowedFiringRates, 1);
        modelParameters.averages(timeWindowIdx).avX = squeeze(mean(meanPosX, 1));
        modelParameters.averages(timeWindowIdx).avY = squeeze(mean(meanPosY, 1));
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: binAndSqrtSpikes
% This function re-bins the spike data and optionally applies a square-root transformation.
function processedTrials = binAndSqrtSpikes(rawTrials, binInterval, applySqrt)
    processedTrials = struct;
    for colIdx = 1:size(rawTrials,2)
        for rowIdx = 1:size(rawTrials,1)
            spikeMatrix = rawTrials(rowIdx, colIdx).spikes;  % rows: neurons, cols: time points
            [numNeurons, numTimePoints] = size(spikeMatrix);
            newBinEdges = 1:binInterval:numTimePoints+1;
            binnedSpikes = zeros(numNeurons, numel(newBinEdges)-1);
            for binIdx = 1:numel(newBinEdges)-1
                binnedSpikes(:,binIdx) = sum(spikeMatrix(:, newBinEdges(binIdx):newBinEdges(binIdx+1)-1), 2);
            end
            if applySqrt
                binnedSpikes = sqrt(binnedSpikes);
            end
            processedTrials(rowIdx, colIdx).spikes = binnedSpikes;
            processedTrials(rowIdx, colIdx).handPos = rawTrials(rowIdx, colIdx).handPos(1:2,:);
            processedTrials(rowIdx, colIdx).bin_size = binInterval;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: computeFiringRates
% This function computes smoothed firing rates using a Gaussian kernel.
function finalTrials = computeFiringRates(binnedTrials, binInterval, gaussianScale)
    finalTrials = struct;
    gaussianWindowSize = 10*(gaussianScale/binInterval);
    normStd = gaussianScale/binInterval;
    alphaParam = (gaussianWindowSize-1)/(2*normStd);
    timeWindow = -(gaussianWindowSize-1)/2:(gaussianWindowSize-1)/2;
    gaussianTemp = exp((-1/2) * (alphaParam * timeWindow/((gaussianWindowSize-1)/2)).^2)';
    gaussianKernel = gaussianTemp/sum(gaussianTemp);
    
    for colIdx = 1:size(binnedTrials,2)
        for rowIdx = 1:size(binnedTrials,1)
            [numNeurons, numBins] = size(binnedTrials(rowIdx, colIdx).spikes);
            smoothedRates = zeros(numNeurons, numBins);
            for neuronIdx = 1:numNeurons
                smoothedRates(neuronIdx,:) = conv(binnedTrials(rowIdx, colIdx).spikes(neuronIdx,:), gaussianKernel, 'same')/(binInterval/1000);
            end
            finalTrials(rowIdx, colIdx).rates = smoothedRates;
            finalTrials(rowIdx, colIdx).handPos = binnedTrials(rowIdx, colIdx).handPos;
            finalTrials(rowIdx, colIdx).bin_size = binnedTrials(rowIdx, colIdx).bin_size;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: getPaddedAndResampledPositions
% This function pads each trial's hand position data to a common length
% and downsamples (resamples) it based on the binning resolution.
function [meanPosX, meanPosY, resampledPosX, resampledPosY] = getPaddedAndResampledPositions(positionData, numDirections, numTrainingTrials, binInterval)
    % Determine the maximum trajectory length among all trials.
    dataCells = struct2cell(positionData);
    trajectoryLengths = [];
    for idx = 2:3:(numTrainingTrials*numDirections*3)
        trajectoryLengths = [trajectoryLengths, size(dataCells{idx},2)];
    end
    maxTrajectoryLength = max(trajectoryLengths); 
    clear dataCells
    
    % Preallocate matrices for hand position data.
    meanPosX = zeros(numTrainingTrials, maxTrajectoryLength, numDirections);
    meanPosY = zeros(numTrainingTrials, maxTrajectoryLength, numDirections);
    
    % Pad trajectories with the last recorded value if they are shorter than maximum length.
    for directionIdx = 1:numDirections
        for trialIdx = 1:numTrainingTrials
            currentLength = trajectoryLengths(numTrainingTrials*(directionIdx-1) + trialIdx);
            meanPosX(trialIdx,:,directionIdx) = [positionData(trialIdx,directionIdx).handPos(1,:), ...
                positionData(trialIdx,directionIdx).handPos(1,end)*ones(1, maxTrajectoryLength-currentLength)];
            meanPosY(trialIdx,:,directionIdx) = [positionData(trialIdx,directionIdx).handPos(2,:), ...
                positionData(trialIdx,directionIdx).handPos(2,end)*ones(1, maxTrajectoryLength-currentLength)];
            % Downsample the padded trajectory based on binInterval.
            tempPosX = meanPosX(trialIdx,:,directionIdx);
     
            %tempPosY = meanPosX(trialIdx,:,directionIdx);
            tempPosY = meanPosY(trialIdx,:,directionIdx);

            resampledPosX(trialIdx,:,directionIdx) = tempPosX(1:binInterval:end);
            resampledPosY(trialIdx,:,directionIdx) = tempPosY(1:binInterval:end);
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [principalComponents, eigenValues, sortIndices, eigenVectors] = svdPCA(dataMatrix)
    % Remove mean across trials (each row)
    centeredData = dataMatrix - mean(dataMatrix, 2);
    
    % Compute SVD of the centered data matrix
    [U, S, V] = svd(centeredData, 'econ');
    
    % The right singular vectors (V) are the eigenvectors of the covariance matrix
    eigenVectors = V;
    
    % Convert singular values to eigenvalues:
    % For covariance matrix (centeredData' * centeredData)/n, eigenvalues = (singular values)^2/n
    s = diag(S);                     % Singular values (already in descending order)
    eigenVals = (s.^2) / size(dataMatrix,2);
    eigenValues = diag(eigenVals);   % Return as a diagonal matrix
    
    % The sort indices are simply the natural order since SVD sorts singular values descending
    sortIndices = 1:length(s);
    
    % Project data onto the new eigenbasis
    principalComponents = centeredData * eigenVectors;
    
    % Normalize each principal component (each column) to have unit norm
    principalComponents = principalComponents ./ sqrt(sum(principalComponents.^2, 1));

end

