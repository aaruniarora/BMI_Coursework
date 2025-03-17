function modelParameters = positionEstimatorTraining(trainingData)
    % Parameters
    classifierBinSize = 20;        % 20ms bins for classifier
    classifierWindowSize = 300;    % 300ms analysis window
    regressorBinSize = 20;         % Matching regressor params
    regressorWindowSize = 300;
    
    [numTrials, numAngles] = size(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);

    %% 1. Classifier Training with Feature Tracking
    [classifierFeatures, validFeatures, featureMask] = preprocessClassifierFeatures(trainingData, classifierBinSize, classifierWindowSize);
    classifierLabels = repelem(1:numAngles, numTrials)';
    
    % Train regularized LDA
    classifier = fitcdiscr(classifierFeatures, classifierLabels, ...
        'DiscrimType', 'diagLinear', ...
        'Gamma', 0.3, ...
        'Prior', 'uniform');

    %% 2. Regressor Training
    regressors = cell(1, numAngles);
    for dir = 1:numAngles
        [allFeat, allPos] = preprocessForRegression(trainingData(:, dir), regressorBinSize, regressorWindowSize);
        
        [coeff, score, ~, ~, explained, mu] = pca(allFeat);
        numComponents = find(cumsum(explained) >= 95, 1, 'first');
        Xb = [score(:,1:numComponents), ones(size(score,1),1)];
        Beta = Xb \ allPos;
        
        regressors{dir} = struct(...
            'projMatrix', coeff(:,1:numComponents), ...
            'Beta', Beta, ...
            'mu', mu, ...
            'binSize', regressorBinSize, ...
            'windowSize', regressorWindowSize);
    end

    %% 3. Store Feature Metadata
    modelParameters = struct(...
        'classifier', classifier, ...
        'regressors', {regressors}, ...
        'featureMask', featureMask, ...
        'expectedFeatures', size(classifierFeatures, 2), ...
        'classifierParams', struct('binSize', classifierBinSize, 'windowSize', classifierWindowSize));
end

function [features, validFeatures, featureMask] = preprocessClassifierFeatures(data, binSize, windowSize)
    % Feature extraction and selection
    numNeurons = size(data(1,1).spikes, 1);
    features = [];
    
    % 1. Extract raw features
    for angle = 1:size(data,2)
        for trial = 1:size(data,1)
            spikes = data(trial,angle).spikes(:,1:windowSize);
            [fr, ~] = preprocessSpikes(spikes, binSize);
            features = [features; fr(:)']; 
        end
    end
    
    % 2. Feature selection
    featVars = var(features);
    validFeatures = featVars > 1e-6; % Threshold for minimum variance
    features = features(:, validFeatures);
    
    % 3. Create full feature mask
    featureMask = false(1, numNeurons*(windowSize/binSize));
    featureMask(1:length(validFeatures)) = validFeatures;
end

function [fr, bins] = preprocessSpikes(spikes, binSize)
    T = size(spikes, 2);
    bins = floor(T / binSize);
    spikesBinned = sum(reshape(spikes(:,1:bins*binSize), size(spikes,1), binSize, []), 2);
    fr = (1000 / binSize) * permute(spikesBinned, [1,3,2]);
end

function [allFeat, allPos] = preprocessForRegression(trials, binSize, windowSize)
    allFeat = []; allPos = [];
    windowBins = windowSize / binSize;
    
    for tr = 1:numel(trials)
        [fr, binCount] = preprocessSpikes(trials(tr).spikes, binSize);
        handPos = trials(tr).handPos(1:2, :);
        
        for t = windowBins:binCount-1
            featVec = fr(:, t-windowBins+1:t);
            allFeat = [allFeat; featVec(:)'];
            allPos = [allPos; handPos(:, t*binSize)'];
        end
    end
end