function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
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

    % % Initial Parameters
    % clc;close all;clear;
    % load('monkeydata_training.mat')
    % test_data = trial;
    % rng(2013);

    %% Parameters
    % [test_length, directions] = size(test_data); 

    bin_group = 20;
    alpha = 0.3; % ema decay
    sigma = 50;  % standard deviation in ms for gaussian
    start_idx = modelParameters.start_idx;
    stop_idx = modelParameters.stop_idx;
    dir_stop = 560;

    % Soft kNN parameters
    k = 8; % 8 for hard kNN and 20 for soft
    pow = 1; 
    alp = 1e-6;
    confidence_threshold = 0.5;

    if ~isfield(modelParameters, 'actualLabel')
        modelParameters.actualLabel = 1; % Default label
    end


   %% Preprocess the trial data
   preprocessed_test = preprocessing(test_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
   neurons = size(preprocessed_test(1,1).rate, 1);

   %% Use indexing based on data given
   curr_bin = size(test_data.spikes, 2);
   idx = min(max(floor((curr_bin - start_idx) / bin_group) + 1, 1), length(modelParameters.classify));

   %%  Remove low firing neurons for PCA and PCR
   spikes_test = extract_features(preprocessed_test, neurons, curr_bin/bin_group, 'nodebug');
   removed_neurons = modelParameters.removeneurons;
   spikes_test(removed_neurons, :) = [];
   % neurons = orig_neurons - length(removed_neurons);

   %% Reshape dataset
   spikes_test = reshape(spikes_test, [], 1);

   %% KNN calssification
   if curr_bin <= dir_stop
      % Classification is applicable within the time range
      train_weight = modelParameters.classify(idx).wTrain;
      test_weight =  modelParameters.classify(idx).wTest;
      meanFiringTrain = modelParameters.classify(idx).mean_firing;
       
      test_weight = test_weight' * (spikes_test(:) - meanFiringTrain(:));
      % Play around with hard and soft kNN. Soft kNN also has dist and exp types!
      last_known_label = modelParameters.actualLabel; % Last known direction label
      [outLabel, confidence] = getKNNs_confidence1(test_weight, train_weight,0,0);

      if confidence_threshold > confidence
          outLabel = round(0.7 * modelParameters.actualLabel + 0.3* outLabel);
      end

    else 
       % Beyond maxTime, use the last known label without re-classification
       outLabel = modelParameters.actualLabel;
    end
    modelParameters.actualLabel = outLabel; % Update the actual label in model parameters
    
    %% Estimate position using PCR results for both within and beyond maxTime
    % Estimate the hand position using PCR, applicable for both within the specified 
    % time range and beyond. This step uses regression coefficients and average firing 
    % rates to calculate the X and Y coordinates of the hand position.
    
    avX = modelParameters.averages(idx).avX(:,outLabel);
    avY =  modelParameters.averages(idx).avY(:,outLabel);
    meanFiring = modelParameters.pcr(outLabel, idx).fMean;
    bx = modelParameters.pcr(outLabel,idx).bx;
    by = modelParameters.pcr(outLabel,idx).by;
    
    x = calculatePosition(spikes_test, meanFiring, bx, avX, curr_bin);
    y = calculatePosition(spikes_test, meanFiring, by, avY, curr_bin);
    %predicted = outLabel; %The predicted direction of movement

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function preprocessed_data = preprocessing(training_data, bin_group, filter_type, alpha, sigma, method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing trials in the following manner:
    % 1. Bin data to get the firing rate
    % 2. Apply square root transformation
    % 3. Smooth using a recursive filter, exponential moving average (EMA),
    % or gaussian filter
% Inputs:
    % training_data: input training data containing the spikes and hand positions
    % bin_group: binning resolution in milliseconds
    % alpha: Smoothing factor (0 < alpha <= 1). A higher alpha gives more weight to the current data point.
% Output:
    % preprocessed_data: preprocessed dataset with spikes and hand positions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialise
    [rows, cols] = size(training_data); 
    preprocessed_data = struct;
   
    % Bin the spikes by summing counts over non-overlapping windows to get the firing rate
    for c = 1:cols
        for r = 1:rows
            train = training_data(r,c);
            [neurons, timepoints] = size(train.spikes);
            num_bins = floor(timepoints / bin_group); % 28
            % t_new = 1:bin_group:timepoints + 1;
            % num_bins = numel(t_new) - 1;

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

            % Apply gaussian smoothing
            if strcmp(filter_type, 'Gaussian')
                gKernel = gaussian_filter(bin_group, sigma);
                % Convolve each neuron's spike train with the Gaussian kernel.
                gaussian_spikes = zeros(size(sqrt_spikes));
                for n = 1:neurons
                    gaussian_spikes(n,:) = conv(sqrt_spikes(n,:), gKernel, 'same')/(bin_group/1000);
                end
                preprocessed_data(r,c).rate = gaussian_spikes; % spikes per millisecond
            end

            % Apply EMA smoothing
            if strcmp(filter_type, 'EMA')
                ema_spikes = ema_filter(sqrt_spikes, alpha, neurons);
                preprocessed_data(r,c).rate = ema_spikes / (bin_group/1000); % spikes per second
            end
            
            % preprocessed_data(r,c).handPos = training_data(r,c).handPos;
        end
    end
    
    if strcmp(method, 'debug')
        plot_r = 1; plot_c = 1; plot_n =1;
        figure; sgtitle('After preprocessing');
        subplot(1,2,1); hold on;
        % plot(training_data(plot_r,plot_c).spikes(plot_n,:), DisplayName='Original', LineWidth=1.5); 
        plot(preprocessed_data(plot_r,plot_c).rate(plot_n,:), DisplayName='Preprocessed', LineWidth=1.5);
        xlabel('Bins'); ylabel('Firing Rate (spikes/s)');
        title('Spikes'); legend show; hold off;
    
        subplot(1,2,2); hold on;
        plot(preprocessed_data(plot_r,plot_c).handPos(1,:), preprocessed_data(plot_r,plot_c).handPos(2,:), DisplayName='Original', LineWidth=1.5); 
        xlabel('x pos'); ylabel('y pos');
        title('Hand Positions'); legend show; hold off;
    end
end


function ema_spikes = ema_filter(sqrt_spikes, alpha, num_neurons)
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:size(sqrt_spikes, 2)
            ema_spikes(n, t) = alpha * sqrt_spikes(n, t) + (1 - alpha) * ema_spikes(n, t - 1);
        end
    end
end


function gKernel = gaussian_filter(bin_group, sigma)
    % Create a 1D Gaussian kernel centered at zero.
    gaussian_window = 10*(sigma/bin_group);
    e_std = sigma/bin_group;
    alpha = (gaussian_window-1)/(2*e_std);

    time_window = -(gaussian_window-1)/2:(gaussian_window-1)/2;
    gKernel = exp((-1/2) * (alpha * time_window/((gaussian_window-1)/2)).^2)';
    gKernel = gKernel / sum(gKernel);
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
    % pos = adjustFinalPosition(pos, curr_bin);

    try
        pos = pos(curr_bin, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end

end


%% kNN
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
