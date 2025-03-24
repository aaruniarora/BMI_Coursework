function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    %% Initial Parameters
    % clc;close all;clear;
    % load('monkeydata_training.mat')
    % test_data = trial;
    % rng(2013);

    %% Parameters
    % [test_length, directions] = size(test_data); 

    bin_group = 20;
    alpha = 0.3; % ema decay
    start_idx = modelParameters.start_idx;
    stop_idx = modelParameters.stop_idx;

    % Soft kNN parameters
    k = 20; % 8 for hard kNN and 20 for soft
    pow = 1; 
    alp = 1e-6; 

   %% Preprocess the trial data
   preprocessed_test = preprocessing(test_data, bin_group, alpha);
   neurons = size(preprocessed_test(1,1).rate, 1);

   %% Use indexing based on data given
   curr_bin = size(test_data.spikes, 2);
   idx = min(max(floor((curr_bin - start_idx) / bin_group) + 1, 1), length(modelParameters.classify));

   %%  Remove low firing neurons for PCA and PCR
   spikes_test = extract_features(preprocessed_test, neurons, curr_bin/bin_group, 'nodebug');
   removed_neurons = modelParameters.removeneurons;
   spikes_test(removed_neurons, :) = [];

   %% Reshape dataset
   spikes_test = reshape(spikes_test, [], 1);

   %% KNN calssification
   if curr_bin <= stop_idx 
      train_weight = modelParameters.classify(idx).wTrain;
      test_weight =  modelParameters.classify(idx).wTest;
      meanFiringTrain = modelParameters.classify(idx).mean_firing;
       
      test_weight = test_weight' * (spikes_test(:) - meanFiringTrain(:));
      % Play around with hard and soft kNN. Soft kNN also has dist and exp types!
      outLabel = KNN_classifier(test_weight, train_weight, k, pow, alp, 'soft', 'dist');

    else 
       outLabel = modelParameters.actualLabel;
    end
    modelParameters.actualLabel = outLabel; 
    
    %% Estimate position using PCR results for both within and beyond maxTime
    avX = modelParameters.averages(idx).avX(:,outLabel);
    avY =  modelParameters.averages(idx).avY(:,outLabel);
    meanFiring = modelParameters.pcr(outLabel, idx).f_mean;
    bx = modelParameters.pcr(outLabel,idx).bx;
    by = modelParameters.pcr(outLabel,idx).by;
    
    x = calculatePosition(spikes_test, meanFiring, bx, avX, curr_bin);
    y = calculatePosition(spikes_test, meanFiring, by, avY, curr_bin);
end

%% HELPER FUNCTIONS FOR PREPROCESSING OF SPIKES

function preprocessed_data = preprocessing(training_data, bin_group, alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing trials in the following manner:
    % 1. Pad each trial’s spikes out to max_time_length
    % 2. Bin data to get the firing rate
    % 3. Apply square root transformation
    % 4. Smooth using a recursive filter, exponential moving average (EMA)
% Inputs:
    % training_data: input training data containing the spikes and hand positions
    % bin_group: binning resolution in milliseconds
    % filter_type: choose between 'EMA' and 'Gaussian' filtering
    % alpha: Smoothing factor (0 < alpha <= 1). A higher alpha gives more weight to the current data point.
    % sigma: gaussian filtering window
    % debug: plots if debug=='debug'
% Output:
    % preprocessed_data: preprocessed dataset with spikes and hand positions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialise
    [rows, cols] = size(training_data); 
    preprocessed_data = struct;

    spike_cells = {training_data.spikes};
    max_time_length = max(cellfun(@(sc) size(sc, 2), spike_cells));
    clear spike_cells;

    % Pad each trial’s spikes out to max_time_length
    for tl = 1:rows
        for dir = 1:cols
            curr_spikes = training_data(tl, dir).spikes; 
            curr_spikes = fill_nan(curr_spikes, 'spikes');
            [num, T] = size(curr_spikes);
            if T < max_time_length
                padNeeded = max_time_length - T;
                training_data(tl, dir).spikes = [curr_spikes, zeros(num, padNeeded)]; % repmat(curr_spikes(:, end), 1, padNeeded)
            end
        end
    end

    % Bin the spikes by summing counts over non-overlapping windows to get the firing rate
    for c = 1:cols
        for r = 1:rows
            train = training_data(r,c);
            [neurons, timepoints] = size(train.spikes);
            num_bins = floor(timepoints / bin_group); % 28

            binned_spikes = zeros(neurons, num_bins);

            for b = 1:num_bins
                start_time = (b-1)*bin_group + 1; % 1, 21, 41, ..., 541
                end_time = b*bin_group; % 20, 40, 60, ..., 560
                if b == num_bins % gets all the leftover points for the last bin
                    binned_spikes(:,b) = sum(train.spikes(:, start_time:end), 2);
                else
                    binned_spikes(:,b) = sum(train.spikes(:, start_time:end_time), 2);
                end
            end
            
            % Apply sqrt transformation 
            sqrt_spikes = sqrt(binned_spikes);

            % Apply EMA smoothing
            ema_spikes = ema_filter(sqrt_spikes, alpha, neurons);
            preprocessed_data(r,c).rate = ema_spikes / (bin_group/1000); % spikes per second
        end
    end
end


function ema_spikes = ema_filter(sqrt_spikes, alpha, num_neurons)
    % Runs exponential moving average filter on the given data
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:size(sqrt_spikes, 2)
            ema_spikes(n, t) = alpha * sqrt_spikes(n, t) + (1 - alpha) * ema_spikes(n, t - 1);
        end
    end
end


function data = fill_nan(data, data_type)

    if strcmp(data_type, 'spikes')
        data(isnan(data)) = 0;
    end
    
    if strcmp(data_type, 'handpos')
        % Forward fill
        for r = 2:length(data)
            if isnan(data(r))
                data(r) = data(r-1);
            end
        end
        % Backward fill for any remaining NaNs
        for r = length(data)-1:-1:1
            if isnan(data(r))
                data(r) = data(r+1);
            end
        end
    end
end


%% Extract features
function spikes_matrix = extract_features(preprocessed_data, neurons, curr_bin, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Arranging data as:
% rows: 2744 time points --> 98 neurons x 28 bins
% cols: 800 --> 8 angles and 100 trials so angle 1, trial 1; angle 1, trial 2; ...; angle 8, Trial 100
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [rows, cols] = size(preprocessed_data); % 98 x 16
    
    for r = 1:rows % 98
        for c = 1:cols % 16
            for k = 1:curr_bin
                c_idx = rows * (c - 1) + r; % 100 (1 - 1) + 1 = 1; 1; 1...x13; 101; 
                r_start = neurons * (k - 1) + 1; % 98 (1 - 1) + 1 = 1; 99; 197;...
                r_end = neurons * k; % 98; 196;...
                spikes_matrix(r_start:r_end,c_idx) = preprocessed_data(r,c).rate(:,k);  
            end
        end
    end

    if strcmp(debug, 'debug')
        figure; title(['Firing Rate for Bin ' num2str(curr_bin)]);
        plot(spikes_matrix); 
    end
end


%% Helper function for position calculation
function pos = calculatePosition(neuraldata, meanFiring, b, av, curr_bin)
    pos = (neuraldata(1:length(b)) - mean(meanFiring))' * b + av;
    % pos = ((neuraldata(1:length(b)) - meanFiring))' * b + av;
    try
        pos = pos(curr_bin, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end
end


%% kNN
function output_lbl = KNN_classifier(test_weight, train_weight, NN_num, pow, alp, method, type)

    if strcmp(method, 'hard')
    % Input:
    %  test_weight: Testing dataset after projection 
    %  train_weight: Training dataset after projection
    %  NN_num: Used to determine the number of nearest neighbors
     
        trainlen = size(train_weight, 2) / 8; 
        k = max(1, round(trainlen / NN_num)); 
    
        output_lbl = zeros(1, size(test_weight, 2));
    
        for i = 1:size(test_weight, 2)
            % distances = sum(bsxfun(@minus, train_weight, test_weight(:, i)).^2, 1);
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
        %
        % Inputs:
        %  test_weight: [projDim x #TestSamples]  (the LDA-projected test sample(s))
        %  train_weight: [projDim x #TrainSamples]
        %  NN_num: sets how we pick k, i.e. k = trainlen/NN_num or similar
        %
        % Output:
        %  output_lbl: predicted direction label for each test sample

        nAngles = 8;  % you have 8 reaching angles
        trainlen = size(train_weight, 2) / nAngles; 
        k = max(1, round(trainlen / NN_num)); 
    
        output_lbl = zeros(1, size(test_weight, 2));
    
        for i = 1:size(test_weight, 2)
            % For the i-th test sample:
            distances = sum((train_weight - test_weight(:, i)).^2, 1);
    
            % Sort and get top-k nearest neighbors
            [sortedDist, sortedIdx] = sort(distances, 'ascend');
            nearestIdx    = sortedIdx(1:k);
            nearestDist   = sortedDist(1:k);
    
            % Convert index -> direction label
            % If train_weight is grouped angle-by-angle, we do this:
            trainLabels = ceil(nearestIdx / trainlen);  % each from 1..8
    
            % Compute distance-based weights, e.g. 1/d^2
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
            
            % % Final predicted label is the angle with the highest sum of weights
            % [~, bestAngle] = max(angleWeights);
            % output_lbl(i) = bestAngle;
    
            % Or we can use probability distribution
            p = angleWeights / sum(angleWeights);
            [~, bestAngle] = max(p);
            output_lbl(i) = bestAngle;
        end
    end
end

%% kNN
% function [predictedLabel, confidence] = getKNNs_confidence1(testProjection, trainingProjection)
%     % Replace kNN with a Nearest-Centroid approach.
%     % Same inputs/outputs so it can directly replace your existing kNN code.
%     %
%     % Inputs:
%     %   testProjection     - [D x Ntest] matrix: columns are test samples in LDA space.
%     %   trainingProjection - [D x Ntrain] matrix: columns are training samples in LDA space.
%     %   ldaDimension       - Not used here, but preserved for signature consistency.
%     %   neighborhoodFactor - Not used here, but preserved for signature consistency.
%     %
%     % Outputs:
%     %   predictedLabel  - Single integer label (1..8).
%     %   confidence      - Single scalar in [0,1].
%     %
%     % -----------------------------------------------------------------------
% 
%     % Number of directions
%     numDirections = 8;
% 
%     % Count how many total training samples there are for each direction:
%     numTrialsPerDirection = size(trainingProjection, 2) / numDirections;
% 
%     % Build direction labels for each training sample [1..8].
%     directionLabels = [ ...
%         ones(1,numTrialsPerDirection), ...
%         2*ones(1,numTrialsPerDirection), ...
%         3*ones(1,numTrialsPerDirection), ...
%         4*ones(1,numTrialsPerDirection), ...
%         5*ones(1,numTrialsPerDirection), ...
%         6*ones(1,numTrialsPerDirection), ...
%         7*ones(1,numTrialsPerDirection), ...
%         8*ones(1,numTrialsPerDirection) ...
%     ];
% 
%     % Compute the centroid for each direction (mean over columns that belong to that direction).
%     % trainingProjection is D x Ntrain. We'll gather columns belonging to each direction
%     % and compute mean across them.
%     centroids = zeros(size(trainingProjection,1), numDirections);  % (D x 8)
%     for dirIdx = 1:numDirections
%         colsForThisDir = (directionLabels == dirIdx);
%         centroids(:, dirIdx) = mean(trainingProjection(:, colsForThisDir), 2);
%     end
% 
%     % testProjection can have multiple columns (multiple test samples).
%     % We'll classify each column (test sample) to the nearest centroid.
%     % Then aggregate a single label + confidence in the same scalar form 
%     % as the existing kNN code (which ends up returning one label/confidence).
%     %
%     % If you truly only ever call this with one test sample at a time, 
%     % this loop effectively does a single pass anyway.
%     %
%     % Distances to centroids: (Ntest x 8)
%     Ntest = size(testProjection, 2);
%     dists = zeros(Ntest, numDirections);
%     for iTest = 1:Ntest
%         diffToCentroids = centroids - testProjection(:, iTest);
%         dists(iTest,:) = sum(diffToCentroids.^2, 1);  % Euclidean^2 distance
%     end
% 
%     % For each test sample, pick the class (direction) with min distance:
%     [~, perSampleLabels] = min(dists, [], 2);  % Ntest x 1 integer labels
% 
%     % Just like your kNN code does 'mode(mode(...))', we reduce multiple test samples
%     % to a single final label: 
%     predictedLabel = mode(perSampleLabels);
% 
%     % A simple "confidence" measure: fraction of test samples that voted for predictedLabel.
%     % This mimics how the kNN version lumps multiple test samples into a single label/confidence.
%     votesForLabel = sum(perSampleLabels == predictedLabel);
%     confidence = votesForLabel / Ntest;
% end

