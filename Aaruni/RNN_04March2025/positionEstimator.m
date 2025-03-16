function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATED POSITION ESTIMATOR (MAHAD-STYLE MERGE)
    %
    % Key updates:
    %  - Use same binning + smoothing + remove low-firing neurons from training.
    %  - Classify each trial’s direction with “soft kNN” or confidence-based approach.
    %  - Retrieve the correct PCR coefficients for the bin index.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 1) Basic parameters
    binSize  = 20;
    alphaEMA = 0.35;
    sigmaG   = 50;

    start_idx = modelParameters.start_idx;
    stop_idx  = modelParameters.stop_idx;
    numBins   = length(start_idx:binSize:stop_idx);

    % 2) Preprocess the test trial (bin, sqrt, smooth)
    preData = preprocessing(test_data, binSize, 'EMA', alphaEMA, sigmaG);

    % 3) Determine current bin from the trial length
    T = size(test_data.spikes,2);
    currentBin = floor((T - start_idx)/binSize)+1;
    currentBin = max(currentBin,1);
    currentBin = min(currentBin, numBins);

    % 4) Remove low-firing neurons (the same set from training!)
    lowFirers = modelParameters.lowFirers; % from training
    trialRates = preData.rate;  % shape [nNeurons x #bins]
    if ~isempty(lowFirers)
        trialRates(lowFirers,:) = [];
    end

    % 5) Flatten the first currentBin bins to match how we built training data
    trialRates = trialRates(:, 1:currentBin);
    testCol    = trialRates(:);  % #neurons*currentBin x 1

    % 6) Classification
    %    If T <= stop_idx, do a fresh classification. Otherwise, keep old label.
    if T <= stop_idx
        cStruct    = modelParameters.classify(currentBin);
        meanFiring = cStruct.mean_firing;   % a vector
        wTest      = cStruct.wTest';        % [ldaDim x neurons*bin]
        trainW     = cStruct.wTrain;        % [ldaDim x totalSamples]

        testWeight = wTest*(testCol - meanFiring);
        outLabel   = KNN_classifier_soft(testWeight, trainW, 8);  % or any #neighbors logic
    else
        outLabel = modelParameters.actualLabel;  % skip reclassification
    end
    modelParameters.actualLabel = outLabel;

    % 7) PCR-based position
    %    Retrieve the correct bin’s average, fMean, and bx/by from pcr struct
    avX = modelParameters.averages(currentBin).avX(:, outLabel);
    avY = modelParameters.averages(currentBin).avY(:, outLabel);

    pcrStruct = modelParameters.pcr(outLabel, currentBin);
    bx        = pcrStruct.bx;
    by        = pcrStruct.by;
    fMean     = pcrStruct.fMean;

    x = computePCRposition(testCol, fMean, bx, avX, currentBin);
    y = computePCRposition(testCol, fMean, by, avY, currentBin);
end

%% ========================================================================
% HELPER: same as training’s preprocessing
function preprocessed_data = preprocessing(training_data, bin_group, filter_type, alpha, sigma)
    % identical to your training version
    [rows, cols] = size(training_data);
    preprocessed_data(rows,cols).rate = [];

    for c = 1:cols
        for r = 1:rows
            spk   = training_data(r,c).spikes;
            [nNeurons,T] = size(spk);
            nBins = floor(T/bin_group);

            binnedSpikes = zeros(nNeurons, nBins);
            for b = 1:nBins
                s_ = (b-1)*bin_group+1; e_ = b*bin_group;
                if b==nBins
                    binnedSpikes(:,b) = sum(spk(:,s_:end),2);
                else
                    binnedSpikes(:,b) = sum(spk(:,s_:e_),2);
                end
            end

            % sqrt
            binnedSpikes = sqrt(binnedSpikes);

            % smoothing
            if strcmp(filter_type,'EMA')
                out = zeros(size(binnedSpikes));
                for nn=1:nNeurons
                    for t=2:nBins
                        out(nn,t)= alpha*binnedSpikes(nn,t)+(1-alpha)*out(nn,t-1);
                    end
                end
                binnedSpikes = out/(bin_group/1000);
            elseif strcmp(filter_type,'Gaussian')
                % ...
            end
            preprocessed_data(r,c).rate = binnedSpikes;
        end
    end
end

%% ========================================================================
% HELPER: soft kNN
function outLabel = KNN_classifier_soft(testVec, trainMat, NN_num)
    nAngles  = 8;
    trainLen = size(trainMat,2)/nAngles;
    k        = max(1, round(trainLen/NN_num));

    distances = sum((trainMat - testVec).^2,1);
    [sortedDist, sortedIdx] = sort(distances,'ascend');
    nearestIdx  = sortedIdx(1:k);
    nearestDist = sortedDist(1:k);

    trainLabels = ceil(nearestIdx / trainLen);

    w = 1./(nearestDist+1e-6).^2;
    angleWeights = zeros(1,nAngles);
    for i=1:k
        angle = trainLabels(i);
        angleWeights(angle)= angleWeights(angle)+w(i);
    end
    [~,bestAngle] = max(angleWeights);
    outLabel = bestAngle;
end

%% ========================================================================
% HELPER: computePCRposition
function pos = computePCRposition(testCol, meanFiring, bCoef, avPos, currentBin)
    % replicate your “calculatePosition” logic
    offsetTerm = (testCol(1:length(bCoef)) - mean(meanFiring))'* bCoef + avPos;
    % Then pick the entry for currentBin or last
    try
        pos = offsetTerm(currentBin,1);
    catch
        pos = offsetTerm(end,1);
    end
end
