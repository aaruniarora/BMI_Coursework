function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

    %% Parameters

    bin_group = 20;
    alpha = 0.3; % ema decay
    start_idx = modelParameters.start_idx;
    stop_idx = modelParameters.stop_idx;
    % dir_stop = 460;

    % Soft kNN parameters
    k = 20; % 8 for hard kNN and 20 for soft
    pow = 1; 
    % alp = 1e-6;
    % confidence_threshold = 0.5;

    if ~isfield(modelParameters, 'actualLabel')
        modelParameters.actualLabel = []; % Default label
    end


   %% Preprocess the trial data
   preprocessed_test = preprocessing(test_data, bin_group,alpha);
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

   %%

    function [outLabel] = classify_angle(train_weight,test_weight,meanFiringTrain)
      
      test_weight = test_weight' * (spikes_test(:) - meanFiringTrain(:));
      % Play around with hard and soft kNN. Soft kNN also has dist and exp types!
      % [outLabel, confidence] = getKNNs_confidence3(test_weight, train_weight,'euclidean');
      % [outLabel, confidence] = getKNNs_confidence1(test_weight, train_weight);
      [outLabel] = KNN_classifier(test_weight, train_weight, k, pow, 'soft', 'dist');
  
    end
   
   %% KNN calssification
   if curr_bin <= stop_idx
      % Classification is applicable within the time range
      train_weight = modelParameters.classify(idx).wTrain;
      test_weight =  modelParameters.classify(idx).wTest;
      meanFiringTrain = modelParameters.classify(idx).mean_firing;
       
      [outLabel] = classify_angle(train_weight,test_weight,meanFiringTrain);  
          % if confidence_threshold > confidence
      %     outLabel = round(0.7 * modelParameters.actualLabel + 0.3* outLabel);
      % end

    else 
       % Beyond maxTime, use the last known label without re-classification
       outLabel = mode(modelParameters.actualLabel);
   end

    if modelParameters.trial_id == 0
    modelParameters.trial_id = test_data.trialId;
    else 
    if modelParameters.trial_id ~= test_data.trialId
        modelParameters.iterations = 0;
        modelParameters.trial_id = test_data.trialId;
        modelParameters.actualLabel = [];
    end
    end
    modelParameters.iterations = modelParameters.iterations + 1;
    
    % disp(modelParameters.actualLabel)

    %% Reset `actualLabel` if there are repeated inconsistencies
    if ~isempty(modelParameters.actualLabel)
        if modelParameters.actualLabel(end) ~= outLabel
            if length(modelParameters.actualLabel) > 10 && sum(modelParameters.actualLabel(end-4:end) ~= outLabel) >= 5
                % If the last 5 classifications contain at least 3 mismatches, reset
                modelParameters.actualLabel = [];
            end
        end
    end

    len_b_mode = 6;
    % Update the actual label in model parameters
    if ~isempty(modelParameters.actualLabel)
        
    % Accumulate stable labels before following the mode
    if length(modelParameters.actualLabel) > len_b_mode  % Wait until there are at least 5 labels
        outLabel = mode(modelParameters.actualLabel);
        
    end
    modelParameters.actualLabel(end+1) = outLabel;
    modelParameters.actualLabel(:) = outLabel; 
     % Ensure all entries are consistent
    else
        % For the very first classification, just set the label
        modelParameters.actualLabel(end+1) = outLabel;
    end 
    
    
    %% Estimate position using PCR results for both within and beyond maxTime
    % Estimate the hand position using PCR, applicable for both within the specified 
    % time range and beyond. This step uses regression coefficients and average firing 
    % rates to calculate the X and Y coordinates of the hand position.
    
    avX = modelParameters.averages(idx).avX(:,outLabel);
    avY =  modelParameters.averages(idx).avY(:,outLabel);
    meanFiring = modelParameters.pcr(outLabel, idx).fMean;
    % stdev = modelParameters.pcr(outLabel, idx).fstd;
    bx = modelParameters.pcr(outLabel,idx).bx;
    by = modelParameters.pcr(outLabel,idx).by;
    x = calculatePosition(spikes_test, meanFiring, bx, avX, curr_bin);
    y = calculatePosition(spikes_test, meanFiring, by, avY, curr_bin);
    

end


%% HELPER FUNCTIONS FOR PREPROCESSING OF SPIKES
function preprocessed_data = preprocessing(training_data, bin_group, alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing trials in the following manner:
    % 1. Pad each trial's spikes out to max_time_length
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
    % Pad each trial's spikes out to max_time_length
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
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:size(sqrt_spikes, 2)
            ema_spikes(n, t) = alpha * sqrt_spikes(n, t) + (1 - alpha) * ema_spikes(n, t - 1);
        end
    end
end

% 
% function gKernel = gaussian_filter(bin_group, sigma)
%     % Create a 1D Gaussian kernel centered at zero.
%     gaussian_window = 10*(sigma/bin_group);
%     e_std = sigma/bin_group;
%     alpha = (gaussian_window-1)/(2*e_std);
% 
%     time_window = -(gaussian_window-1)/2:(gaussian_window-1)/2;
%     gKernel = exp((-1/2) * (alpha * time_window/((gaussian_window-1)/2)).^2)';
%     gKernel = gKernel / sum(gKernel);
% end


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
% w1 = 1;
% w2 = 1/w1;
    % pos = ((neuraldata(1:length(b)) - mean(meanFiring))./stdev)'*b*w1 + av;
    pos = ((neuraldata(1:length(b)) - meanFiring))'*b + av;
    % pos = adjustFinalPosition(pos, curr_bin);

    try
        pos = pos(curr_bin, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end

end




function [output_lbl] = KNN_classifier(test_weight, train_weight, NN_num, pow, method, type)

    % if strcmp(method, 'hard')
    % % Input:
    % %  test_weight: Testing dataset after projection 
    % %  train_weight: Training dataset after projection
    % %  NN_num: Used to determine the number of nearest neighbors
    % 
    %     trainlen = size(train_weight, 2) / 8; 
    %     k = max(1, round(trainlen / NN_num)); 
    % 
    %     output_lbl = zeros(1, size(test_weight, 2));
    % 
    %     for i = 1:size(test_weight, 2)
    %         % distances = sum(bsxfun(@minus, train_weight, test_weight(:, i)).^2, 1);
    %         distances = sum((train_weight - test_weight(:, i)).^2, 1);
    % 
    %         [~, indices] = sort(distances, 'ascend');
    %         nearestIndices = indices(1:k);
    % 
    % 
    %         trainLabels = ceil(nearestIndices / trainlen); 
    %         modeLabel = mode(trainLabels);
    %         output_lbl(i) = modeLabel;
    % 
    %         % % Compute confidence (fraction of nearest neighbors voting for modeLabel)
    %         % confidence(i) = sum(trainLabels == modeLabel) / k;
    %         % % If confidence is below threshold, keep last known label
    %         % if confidence(i) < confidence_threshold
    %         %     output_lbl(i) = last_label;
    %         % else
    %         %     output_lbl(i) = modeLabel;
    %         % end
    %     end
    % end

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
        % confidence = zeros(1, size(test_weight, 2)); % Store confidence
    
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

            % % Confidence: probability of predicted label
            % confidence(i) = p(bestAngle);
            % 
            % % If confidence is below threshold, keep last known label
            % if confidence(i) < confidence_threshold
            %     output_lbl(i) = last_label;
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