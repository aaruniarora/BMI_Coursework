function modelParameters = positionEstimatorTraining(training_data)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATED POSITION ESTIMATOR TRAINING (MAHAD-STYLE MERGE)
    % 
    % Key updates:
    %  - Identify min_time_length across trials to define stop_idx.
    %  - Remove low-firing neurons (once) and store them in modelParameters.lowFirers.
    %  - Train separate PCA+LDA classifiers for each time bin.
    %  - Train separate PCR regressions for each time bin / direction.
    %  - (Optional) keep partial code for RNN if you want to add temporal decoding.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 1) Basic setup
    [nTrials, nDirections] = size(training_data);
    binSize   = 20;        % binning (ms)
    alphaEMA  = 0.35;      % smoothing
    sigmaGauss= 50;        % if you want Gaussian
    start_idx = 320;

    % Find minimum length across all trials so we don’t index out-of-bounds
    min_time_length = inf;
    for t = 1:nTrials
        for d = 1:nDirections
            Tlen = size(training_data(t,d).spikes, 2);
            if Tlen < min_time_length
                min_time_length = Tlen;
            end
        end
    end
    stop_idx = floor((min_time_length - start_idx)/binSize)*binSize + start_idx;
    time_bins = start_idx:binSize:stop_idx;  % e.g. 320:20:560
    numBins   = length(time_bins);

    % Store in modelParameters
    modelParameters = struct;
    modelParameters.start_idx = start_idx;
    modelParameters.stop_idx  = stop_idx;

    % 2) Preprocess all data at once
    %    (bin, sqrt, EMA or Gaussian)
    preprocessed_data = preprocessing(training_data, binSize, 'EMA', alphaEMA, sigmaGauss);

    % 3) Build a “big” spike matrix for all times (up to stop_idx) to find low-firing neurons
    allSpikes = [];
    for d = 1:nDirections
        for t = 1:nTrials
            fullRates = preprocessed_data(t,d).rate; 
            % Only keep bins up to numBins
            fullRates = fullRates(:, 1:numBins);
            % Flatten
            col = fullRates(:);
            allSpikes = [allSpikes, col];
        end
    end

    % Remove low-firing neurons once
    nNeurons  = size(preprocessed_data(1,1).rate,1); % e.g. 98
    lowFirers = findLowFiringNeurons(allSpikes, nNeurons, 0.5); 
    % store them
    modelParameters.lowFirers = lowFirers;

    % ============== CLASSIFIER TRAINING (PCA+LDA) ==============
    % For each bin from 1..numBins, build the spike matrix minus lowFirers,
    % do PCA+LDA, store in modelParameters.classify(b).

    for b = 1:numBins
        % Build spike matrix for bin b across all trials/directions
        [spikeMat_b, labels_b] = buildSpikeMatrix(preprocessed_data, b, nTrials, nDirections, lowFirers);

        % PCA
        pcaDim = 40;  % choose dimension
        [coeff, score] = doPCA(spikeMat_b, pcaDim);

        % LDA
        ldaDim = 6;
        [ldaOutputs, ldaWeights] = doLDA(spikeMat_b, score, labels_b, ldaDim, nTrials);

        % Store
        modelParameters.classify(b).wTrain      = ldaWeights;
        modelParameters.classify(b).wTest       = ldaOutputs;
        modelParameters.classify(b).mean_firing = mean(spikeMat_b,2);
        modelParameters.classify(b).pcaDim      = pcaDim;
        modelParameters.classify(b).ldaDim      = ldaDim;
    end

    % ============== HAND POSITION PREPROCESSING ==============
    % For each trial, center/pad as needed, then pick out same time bins
    [xPos, yPos, fmtX, fmtY] = handPos_processing(training_data, binSize, time_bins);

    % ============== PCR REGRESSION ==============
    % For each direction, for each time bin, compute regression coefs
    % We'll gather spikes again minus lowFirers, do PCA, and regress => b_x, b_y
    modelParameters.pcr = [];
    modelParameters.averages = struct;

    time_div = kron(binSize:binSize:stop_idx, ones(1, nNeurons - length(lowFirers)));
    % or we can build a separate function to filter the data for each bin/timeWindow

    for dirIdx = 1:nDirections
        currXpos = fmtX(:,:,dirIdx); % #trials x #bins
        currYpos = fmtY(:,:,dirIdx);

        for w = 1:numBins
            [bX, bY, windowFiring] = calcRegressionCoefficients( ...
                w, time_div, [1:nDirections], dirIdx, allSpikes, pcaDim, time_bins, currXpos, currYpos);

            % Store
            modelParameters.pcr(dirIdx,w).bx    = bX;
            modelParameters.pcr(dirIdx,w).by    = bY;
            modelParameters.pcr(dirIdx,w).fMean = mean(windowFiring, 1);
        end
    end

    % Also store average position for each bin
    for w = 1:numBins
        modelParameters.averages(w).avX = squeeze(mean(xPos,1));  % #bins x #directions
        modelParameters.averages(w).avY = squeeze(mean(yPos,1));
    end
end

%% ========================================================================
% HELPER: Preprocessing (like Mahad’s approach but merges Aaruni’s code).
function preprocessed_data = preprocessing(training_data, bin_group, filter_type, alpha, sigma)
    [nTrials, nAngles] = size(training_data);
    preprocessed_data(nTrials,nAngles).rate = [];

    for ang = 1:nAngles
        for tr = 1:nTrials
            spk   = training_data(tr,ang).spikes;
            [nNeurons, T] = size(spk);
            nBins = floor(T/bin_group);

            binned_spikes = zeros(nNeurons, nBins);
            for b = 1:nBins
                s_idx = (b-1)*bin_group + 1;
                e_idx = b*bin_group;
                if b == nBins
                    binned_spikes(:,b) = sum(spk(:, s_idx:end),2);
                else
                    binned_spikes(:,b) = sum(spk(:, s_idx:e_idx),2);
                end
            end

            % sqrt
            binned_spikes = sqrt(binned_spikes);

            % smoothing
            if strcmp(filter_type,'EMA')
                ema_ = zeros(size(binned_spikes));
                for nn=1:nNeurons
                    for t=2:nBins
                        ema_(nn,t) = alpha*binned_spikes(nn,t)+(1-alpha)*ema_(nn,t-1);
                    end
                end
                binned_spikes = ema_ / (bin_group/1000);
            elseif strcmp(filter_type,'Gaussian')
                % etc.
            end

            preprocessed_data(tr,ang).rate = binned_spikes;
        end
    end
end

%% ========================================================================
% HELPER: findLowFiringNeurons => returns a list of neuron indices
function lowFirers = findLowFiringNeurons(bigMatrix, nNeurons, thresholdFR)
    % bigMatrix is shape (nNeurons * #bins, #allTrials) typically, or a flattened form
    % We can reconstruct each neuron's portion or do a step to measure average FR.
    lowFirers = [];

    % Suppose we have total columns = #allTrials
    avgFRall = mean(bigMatrix,2); % average across columns
    % We expect that each block of size nNeurons in the row dimension is 1 bin
    % OR we have nNeurons in each chunk, so we can measure each neuron's row?

    % Easiest might be to do something like:
    nRows = size(bigMatrix,1);
    % # bins total = nRows / nNeurons (assuming perfect multiple)
    nb = nRows / nNeurons;

    for n = 1:nNeurons
        % gather all the row blocks: n, nNeurons + n, 2*nNeurons + n, ...
        idxs = n:nNeurons:nRows;
        FR = mean(avgFRall(idxs)); % average across bins
        if FR < thresholdFR
            lowFirers(end+1) = n; %#ok<AGROW>
        end
    end
end

%% ========================================================================
% HELPER: buildSpikeMatrix for classification at bin b
function [spikeMat, labels] = buildSpikeMatrix(preprocessed_data, binIndex, nTrials, nDirections, lowFirers)
    % Build a #Neurons*(binIndex) x (#Trials*nDirections) matrix
    % removing lowFirers.

    spikeMat = [];
    labels   = [];
    for d = 1:nDirections
        for t = 1:nTrials
            r = preprocessed_data(t,d).rate(:,1:binIndex);
            % remove lowFirers first
            keep_ = 1:size(r,1);
            keep_(lowFirers) = [];
            r = r(keep_,:);

            col = r(:);  % flatten
            spikeMat = [spikeMat, col];
            labels   = [labels, d];
        end
    end
end

%% ========================================================================
% HELPER: doPCA => reduce dimension
function [coeff, score] = doPCA(spikeMat, pcaDim)
    % center
    m0 = mean(spikeMat,2);
    Xc = spikeMat - m0;
    C  = Xc'*Xc;
    [V,D] = eig(C);
    [~, idx] = sort(diag(D),'descend');
    V = V(:, idx(1:pcaDim));
    % project
    score = Xc * V * diag(1./sqrt(diag(D(idx(1:pcaDim)))));
    coeff = V;  % if you want to store or reapply
end

%% ========================================================================
% HELPER: doLDA
function [ldaOutputs, ldaWeights] = doLDA(spikeMat, pcaScore, labels, ldaDim, nTrials)
    % Adapted from your code
    nClasses = length(unique(labels));
    overall_mean = mean(spikeMat,2);

    sw = zeros(size(spikeMat,1));
    sb = zeros(size(spikeMat,1));
    for c = 1:nClasses
        idx = (labels==c);
        cData = spikeMat(:, idx);
        mC = mean(cData,2);
        devW = cData - mC;
        sw   = sw + devW*devW';
        devB = (mC - overall_mean);
        swC  = sum(idx); 
        sb   = sb + swC*(devB*devB');
    end

    pWithin  = pcaScore'* sw * pcaScore;
    pBetween = pcaScore'* sb * pcaScore;

    [V_lda, D_lda] = eig(pinv(pWithin)*pBetween);
    [~, iD] = sort(diag(D_lda),'descend');
    V_lda = V_lda(:, iD(1:ldaDim));

    ldaOutputs = pcaScore * V_lda;
    ldaWeights = ldaOutputs' * (spikeMat - overall_mean);
end

%% ========================================================================
% HELPER: handPos_processing
function [xPos, yPos, fmtX, fmtY] = handPos_processing(training_data, binSize, time_bins)
    [nTrials,nAngles] = size(training_data);

    % Find max trajectory length
    maxLen = 0;
    for d=1:nAngles
        for t=1:nTrials
            L = size(training_data(t,d).handPos,2);
            if L>maxLen, maxLen=L; end
        end
    end

    xPos = zeros(nTrials,maxLen,nAngles);
    yPos = zeros(nTrials,maxLen,nAngles);

    for d=1:nAngles
        for t=1:nTrials
            hp = training_data(t,d).handPos;
            lenHP = size(hp,2);
            padSz = maxLen - lenHP;
            if padSz>0
                xPos(t,:,d)=[hp(1,:), repmat(hp(1,end),1,padSz)];
                yPos(t,:,d)=[hp(2,:), repmat(hp(2,end),1,padSz)];
            else
                xPos(t,:,d)=hp(1,:);
                yPos(t,:,d)=hp(2,:);
            end
        end
    end

    % pick out time_bins
    fmtX = xPos(:, time_bins/binSize, :);  % e.g. if time_bins are 320:20:560 => 16..28
    fmtY = yPos(:, time_bins/binSize, :);
end

%% ========================================================================
% HELPER: calcRegressionCoefficients
function [Bx, By, FilteredFiring] = calcRegressionCoefficients( ...
    timeWindowIndex, time_division, labels, directionIndex, bigSpikes, pcaDim, time_bins, currX, currY)

    % This is a placeholder that you might adapt to do the correct filtering
    % for your timeWindowIndex. For example, gather the subset of bigSpikes that
    % belongs to timeWindowIndex, do PCA, solve for bX,bY.
    % In practice, you want to replicate the approach used in your training code.

    % For demonstration, we’ll do a minimal version:
    centeredX = currX(:,timeWindowIndex)-mean(currX(:,timeWindowIndex));
    centeredY = currY(:,timeWindowIndex)-mean(currY(:,timeWindowIndex));

    % Instead of timeDivision logic, we might just re-slice bigSpikes for directionIndex
    % That gets complicated if bigSpikes is for all directions. We'll keep it simple:
    FilteredFiring = bigSpikes;  % In reality, you'd only keep the columns that belong to directionIndex

    % Center
    FF_centered = FilteredFiring - mean(FilteredFiring,1);

    % PCA
    [U, ~] = doPCA(FF_centered, pcaDim);
    P = FF_centered'*U; % #samples x pcaDim ?

    % Ridge or pseudo-inverse
    M = (P'*P);
    lambda = 1; 
    R = (M+lambda*eye(size(M)))\(P');

    Bx = U * R * centeredX;
    By = U * R * centeredY;
end
