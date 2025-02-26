function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % POSITION ESTIMATOR (DECODER)
    %
    % This function decodes the (x,y) hand position from test data using the
    % modelParameters produced during training. It performs the same
    % preprocessing steps (trimming, padding, Gaussian filtering, and binning)
    % and then uses PCA, LDA, kNN, and a simple RNN update to predict the hand position.
    %
    % newModelParameters is returned unmodified (the RNN here is stateless).
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Check if the test data contains the handPos field.
    if isfield(test_data, 'handPos')
        if size(test_data.spikes,2) > 300
             spikes = test_data.spikes(:, 301:end);
             handPos = test_data.handPos(:, 301:end);
        else
             spikes = test_data.spikes;
             handPos = test_data.handPos;
        end
    else
        % If handPos is not provided, create a dummy handPos using startHandPos.
        spikes = test_data.spikes;
        handPos = repmat(test_data.startHandPos, 1, size(test_data.spikes,2));
    end
    
    % Pad the trial if its length is less than the training padLength
    padLength = modelParameters.padLength;
    currentLength = size(spikes,2);
    if currentLength < padLength
         spikes = [spikes, zeros(size(spikes,1), padLength - currentLength)];
         lastPos = handPos(:,end);
         handPos = [handPos, repmat(lastPos, 1, padLength - currentLength)];
    end
    
    % Apply Gaussian filtering to the spikes
    gKernel = modelParameters.gKernel;
    spikesFiltered = zeros(size(spikes));
    for i = 1:size(spikes,1)
         spikesFiltered(i,:) = conv(spikes(i,:), gKernel, 'same');
    end
    
    % Bin the data using the same binSize as in training
    binSize = modelParameters.binSize;
    [binnedSpikes, binnedHandPos] = binSingleTrial(spikesFiltered, handPos, binSize);
    % binnedHandPos is a 3 x numBins matrix; we use only the first two rows (x,y)
    
    % Use the latest bin's spike counts as features
    currentBin = size(binnedSpikes,2);
    currentSpikeBin = binnedSpikes(:, currentBin)';  % row vector
    
    % Apply the same PCA transformation (center and project)
    currentSpikeBinCentered = currentSpikeBin - modelParameters.PCA.mu;
    featuresPCA = currentSpikeBinCentered * modelParameters.PCA.coeff;  % 1 x nPC
    
    % Apply LDA transformation
    featuresLDA = featuresPCA * modelParameters.LDA.W;  % 1 x nLDA
    
    % --- kNN prediction ---
    k = modelParameters.kNN.k;
    diffs = modelParameters.kNN.features - repmat(featuresLDA, size(modelParameters.kNN.features,1), 1);
    distances = sqrt(sum(diffs.^2, 2));
    [~, idx] = sort(distances);
    idx = idx(1:k);
    knn_estimate = mean(modelParameters.kNN.handPos(idx,:), 1);  % 1 x 2
    
    % --- RNN update ---
    % Use the previous decoded hand position or the starting position if none
    if isempty(test_data.decodedHandPos)
         prevHand = test_data.startHandPos(:)';
    else
         prevHand = test_data.decodedHandPos(:, end)';
    end
    % Form the RNN input: [current LDA features, previous hand position, bias]
    rnnInput = [featuresLDA, prevHand, 1];
    delta = rnnInput * modelParameters.RNN.weights;  % 1 x 2
    rnn_estimate = prevHand + delta;
    
    % Combine the kNN and RNN estimates (here, by averaging)
    finalEstimate = (knn_estimate + rnn_estimate) / 2;
    x = finalEstimate(1);
    y = finalEstimate(2);
    
    newModelParameters = modelParameters;  % no update in this implementation
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction for binning a single trial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [binnedSpikes, binnedHandPos] = binSingleTrial(spikes, handPos, binSize)
    % Bin spikes and hand positions for one trial.
    T = size(spikes,2);
    numBins = floor(T/binSize);
    binnedSpikes = zeros(size(spikes,1), numBins);
    binnedHandPos = zeros(size(handPos,1), numBins);
    for b = 1:numBins
        idxStart = (b-1)*binSize + 1;
        idxEnd = b*binSize;
        binnedSpikes(:,b) = sum(spikes(:, idxStart:idxEnd), 2);
        binnedHandPos(:,b) = handPos(:, idxEnd);
    end
end
