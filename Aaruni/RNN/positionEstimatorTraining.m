function modelParameters = positionEstimatorTraining(training_data)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % POSITION ESTIMATOR TRAINING
    %
    % This function implements a full decoding pipeline that:
    % 1. Preprocesses the data by removing the first 300 ms and the last 100 ms,
    %    then pads each trial to the same length.
    % 2. Applies Gaussian filtering to smooth the spike trains.
    % 3. Bins the filtered data in non-overlapping 20 ms windows.
    % 4. Extracts features (spike counts) and corresponding hand positions.
    % 5. Reduces dimensionality via PCA and then finds discriminative features
    %    with LDA using the known reaching angle labels.
    % 6. Stores the training samples for kNN regression.
    % 7. Trains a simple linear recurrent (RNN) model to predict hand position changes.
    %
    % The learned parameters (including padding length, Gaussian kernel,
    % PCA and LDA parameters, kNN samples, and RNN weights) are saved in
    % the modelParameters structure.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Parameters
    noDirections = 8;
    reaching_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23,18, 31/18, 35/18] .* pi;

    %% 1. Preprocessing: Trim and pad trials
    [preprocTrials, padLength] = preprocessTrials(training_data);
    modelParameters.padLength = padLength;
    
    %% 2. Gaussian Filtering
    sigma = 20;  % standard deviation in ms (adjust as needed)
    kernelRadius = 3*sigma;  % covers most of the kernel mass
    gKernel = createGaussianKernel(kernelRadius, sigma);
    modelParameters.gKernel = gKernel;
    
    preprocTrials = applyGaussianFilter(preprocTrials, gKernel);
    
    %% 3. Binning at 20 ms
    binSize = 20;
    modelParameters.binSize = binSize;
    binnedTrials = binTrials(preprocTrials, binSize);
    
    %% 4. Extract features for PCA/LDA/kNN
    % For each bin in every trial, we create a feature vector (the spike counts
    % from all neurons) and record the corresponding (x,y) hand position.
    [featuresMatrix, handPosMatrix, labels] = extractFeaturesAndLabels(binnedTrials);
    
    %% 5. PCA on the features
    nPC = 10;
    [coeff, score, mu] = cov_PCA(featuresMatrix, nPC);
    % [coeff, score, mu] = getPCA(featuresMatrix', nPC);
    % variance_threshold = 0.95; % Choose number of principal components to retain 95% variance
    % [coeff, score, mu, nPC] = svd_PCA(featuresMatrix, variance_threshold);
    modelParameters.PCA.mu = mu;
    modelParameters.PCA.coeff = coeff;
    modelParameters.PCA.nPC = nPC;
    featuresPCA = score;  % (samples x nPC)
    
    %% 6. LDA on the PCA features
    % Labels (from 1 to 8) indicate the reaching angle (the column index).
    [Wlda] = performLDA(featuresPCA, labels);
    modelParameters.LDA.W = Wlda;
    featuresLDA = featuresPCA * Wlda;  % (samples x nLDA)
    
    %% 7. kNN training: store samples in LDA space with corresponding hand positions
    kNN.k = 5;
    kNN.features = featuresLDA;
    kNN.handPos = handPosMatrix;  % corresponding hand positions (x,y)
    kNN.labels = labels;
    modelParameters.kNN = kNN;
    
    %% 8. Train a simple (linear) RNN model
    % The RNN predicts the difference between consecutive hand positions using:
    %   delta = weights * [current LDA features, previous hand position, bias]
    RNN.weights = trainSimpleRNN(binnedTrials, mu, coeff, Wlda, binSize);
    modelParameters.RNN = RNN;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunctions for positionEstimatorTraining
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [preprocTrials, padLength] = preprocessTrials(trials)
    % Remove the first 300 ms and last 100 ms from each trial,
    % then pad with zeros (for spikes) and repeat the last hand position.
    [numRows, numCols] = size(trials);
    padLength = 0;
    preprocTrials = trials;
    startIdx = 301;
    for i = 1:numRows
        for j = 1:numCols
            trial = trials(i,j);
            T = size(trial.spikes,2);
            endIdx = T - 100;
            if endIdx < startIdx
                error('Trial too short after removal.');
            end
            trial.spikes = trial.spikes(:, startIdx:endIdx);
            trial.handPos = trial.handPos(:, startIdx:endIdx);
            currentLen = size(trial.spikes,2);
            if currentLen > padLength
                padLength = currentLen; % to get the largest trial size
            end
            preprocTrials(i,j) = trial;
        end
    end
    % Pad each trial to padLength
    for i = 1:numRows
        for j = 1:numCols
            trial = preprocTrials(i,j);
            currentLen = size(trial.spikes,2);
            if currentLen < padLength
                padSize = padLength - currentLen;
                trial.spikes = [trial.spikes, zeros(size(trial.spikes,1), padSize)];
                lastPos = trial.handPos(:, end);
                trial.handPos = [trial.handPos, repmat(lastPos, 1, padSize)];
            end
            preprocTrials(i,j) = trial;
        end
    end
end

function gKernel = createGaussianKernel(radius, sigma)
    % Create a 1D Gaussian kernel centered at zero.
    x = -radius:radius;
    gKernel = exp(-(x.^2)/(2*sigma^2));
    gKernel = gKernel / sum(gKernel);
end

function trialsOut = applyGaussianFilter(trials, gKernel)
    % Convolve each neuron's spike train with the Gaussian kernel.
    [numRows, numCols] = size(trials);
    trialsOut = trials;
    for i = 1:numRows
        for j = 1:numCols
            trial = trials(i,j);
            [numNeurons, ~] = size(trial.spikes);
            filteredSpikes = zeros(size(trial.spikes));
            for n = 1:numNeurons
                filteredSpikes(n,:) = conv(trial.spikes(n,:), gKernel, 'same');
            end
            trial.spikes = filteredSpikes;
            trialsOut(i,j) = trial;
        end
    end
end

function binnedTrials = binTrials(trials, binSize)
    % Bin the spikes by summing counts over non-overlapping windows.
    % For handPos, take the last sample in each bin.
    [numRows, numCols] = size(trials);
    binnedTrials = trials;
    for i = 1:numRows
        for j = 1:numCols
            trial = trials(i,j);
            T = size(trial.spikes,2);
            numBins = floor(T / binSize);
            binnedSpikes = zeros(size(trial.spikes,1), numBins);
            binnedHandPos = zeros(size(trial.handPos,1), numBins);
            for b = 1:numBins
                idxStart = (b-1)*binSize + 1;
                idxEnd = b*binSize;
                binnedSpikes(:,b) = sum(trial.spikes(:, idxStart:idxEnd), 2);
                binnedHandPos(:,b) = trial.handPos(:, idxEnd);  % last sample in the bin
            end
            trial.spikes = binnedSpikes;
            trial.handPos = binnedHandPos;
            binnedTrials(i,j) = trial;
        end
    end
end

function [featuresMatrix, handPosMatrix, labels] = extractFeaturesAndLabels(trials)
    % Convert each bin of every trial into a training sample.
    % Each row in featuresMatrix contains spike counts from all neurons.
    % handPosMatrix holds the corresponding (x,y) positions.
    % labels indicate the reaching angle (trial column index).
    [numRows, numCols] = size(trials);
    featuresList = [];
    handPosList = [];
    labelList = [];
    for i = 1:numRows
        for j = 1:numCols
            trial = trials(i,j);
            numBins = size(trial.spikes,2);
            for b = 1:numBins
                feat = trial.spikes(:, b)';  % row vector (1 x numNeurons)
                featuresList = [featuresList; feat];
                handPosList = [handPosList; trial.handPos(1:2, b)'];
                labelList = [labelList; j];  % j is the reaching angle index
            end
        end
    end
    featuresMatrix = featuresList;
    handPosMatrix = handPosList;
    labels = labelList;
end

function [ev, prinComp, mu] = getPCA(data, nPC)
    % Perform Principal Component Analysis (PCA) on the data.
    % Inputs:
    %   data - matrix of firing rates (neurons x time/trials)
    %
    % Outputs:
    %   prinComp - projection of data onto principal components
    %   evals    - eigenvalues (sorted in descending order)
    %   sortIdx  - indices used for sorting eigenvalues
    %   ev       - eigenvectors corresponding to eigenvalues
    %
    % Subtract the cross-trial mean
    mu = mean(data, 2);
    dataCT = data - mu;
    % Calculate covariance matrix
    covMat = dataCT' * dataCT / size(data, 2);
    % Get eigenvalues and eigenvectors
    [evects, evalsMat] = eig(covMat);
    % Sort eigenvalues and eigenvectors in descending order
    [~, sortIdx] = sort(diag(evalsMat), 'descend');
    evects = evects(:, sortIdx);
    % Return eigenvectors as well
    ev = evects(:, 1:nPC);
    % Project firing rate data onto the new basis
    prinComp = dataCT * ev;
    disp('Before'); size(prinComp)
    % Normalize
    prinComp = prinComp ./ sqrt(sum(prinComp.^2));
    disp('After'); size(prinComp)
    % Extract sorted eigenvalues
    evalsDiag = diag(evalsMat);
    evals = diag(evalsDiag(sortIdx));
    mu = mu';
end

function [coeff, score, mu] = cov_PCA(X, nPC)
    % Compute PCA: center the data, get covariance, then the top nPC eigenvectors.
    mu = mean(X,1);
    Xc = X - mu;
    C = cov(Xc);
    [V, D] = eig(C);
    [d, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    coeff = V(:, 1:nPC);
    score = Xc * coeff;

    % Normalize each principal component (each column) to have unit norm
    % normFactors = sqrt(sum(score.^2, 1));  % 1 x nPC vector, norm of each column
    % score = score ./ repmat(normFactors, size(score, 1), 1);
end

function [coeff, score, mu, nPC] = svd_PCA(X, variance_threshold)
    % Compute PCA using Singular Value Decomposition (SVD)
    % Center the data by subtracting the mean from each sample
    mu = mean(X, 1);
    Xc = X - mu;

    % Perform SVD on the centered data (using economy size decomposition)
    [U, S, V] = svd(Xc, 'econ');

    % Compute variance explained
    singular_values = diag(S);
    explained_variance = (singular_values.^2) / sum(singular_values.^2);
    cum_variance = cumsum(explained_variance);

    nPC = find(cum_variance >= variance_threshold, 1);

    % The principal component directions are given by the columns of V
    coeff = V(:, 1:nPC);

    % Reduce data dimensionality: Compute the projection (scores) of the data onto the principal components
    score = Xc * coeff;
end

function Wlda = performLDA(X, labels)
    % Compute the LDA projection matrix.
    classes = unique(labels);
    numClasses = length(classes);
    nFeatures = size(X,2);
    mu_overall = mean(X,1);
    Sw = zeros(nFeatures);
    Sb = zeros(nFeatures);
    for i = 1:numClasses
        Xi = X(labels == classes(i), :);
        mu_i = mean(Xi,1);
        Sw = Sw + cov(Xi) * (size(Xi,1)-1);
        diff = mu_i - mu_overall;
        Sb = Sb + size(Xi,1) * (diff' * diff);
    end
    % Solve the generalized eigenvalue problem (with regularization)
    [V, D] = eig(Sb, Sw + eye(nFeatures)*1e-6);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    nLDA = numClasses - 1;
    Wlda = V(:, 1:nLDA);
end

function weights = trainSimpleRNN(binnedTrials, mu, coeff, Wlda, binSize)
    % Train a simple linear RNN model using least squares.
    % For each trial, for time steps t = 2:end, create input:
    %   [LDA_features (current bin), previous hand position, 1]
    % and target:
    %   delta = current hand position - previous hand position.
    nLDA = size(Wlda,2);
    X_total = [];
    Y_total = [];
    
    [numRows, numCols] = size(binnedTrials);
    for i = 1:numRows
        for j = 1:numCols
            trial = binnedTrials(i,j);
            numBins = size(trial.spikes,2);
            if numBins < 2, continue; end
            % Compute LDA features for each bin using PCA and LDA parameters
            features_lda = zeros(numBins, nLDA);
            for b = 1:numBins
                spike_bin = trial.spikes(:, b)'; % 1 x numNeurons
                spike_bin_centered = spike_bin - mu;
                pca_feat = spike_bin_centered * coeff;  % 1 x nPC
                features_lda(b,:) = pca_feat * Wlda;      % 1 x nLDA
            end
            % For time steps t=2...numBins, get input and target delta
            for b = 2:numBins
                prevHand = trial.handPos(1:2, b-1)';  % previous (x,y)
                currentHand = trial.handPos(1:2, b)';   % current (x,y)
                delta = currentHand - prevHand;          % change in hand pos
                current_feat = features_lda(b, :);        % current LDA features
                X_sample = [current_feat, prevHand, 1];     % add bias term
                X_total = [X_total; X_sample];
                Y_total = [Y_total; delta];
            end
        end
    end
    % Solve for weights: (nLDA + 3) x 2 matrix
    weights = pinv(X_total) * Y_total;
end
