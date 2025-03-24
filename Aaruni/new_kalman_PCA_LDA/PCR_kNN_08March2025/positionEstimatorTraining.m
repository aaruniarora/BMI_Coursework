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
    % clc;close all;clear;
    % load('monkeydata_training.mat')
    % training_data = trial;
    % rng(2013);

    %% Parameters
    [training_length, directions] = size(training_data); 
    reaching_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23/18, 31/18, 35/18] .* pi;
    bin_group = 20;
    alpha = 0.35; % arbitrary value decided through multiple trials
    sigma = 50;  % standard deviation in ms
    start_idx = 320; 

    % Determine the minimum spike length across all trials to ensure we don't exceed array bounds.
    min_time_length = inf;
    for tl = 1:training_length
        for dir = 1:directions
            curr_len = size(training_data(tl, dir).spikes, 2);
            if curr_len < min_time_length
                min_time_length = curr_len;
            end
        end
    end
    
    stop_idx = floor((min_time_length - start_idx) / bin_group) * bin_group + start_idx;
    time_bins = start_idx:bin_group:stop_idx;  % e.g. 320:20:560
    num_bins = time_bins / bin_group;

   % Store in modelParameters
    modelParameters = struct;
    modelParameters.start_idx = start_idx;
    modelParameters.stop_idx  = stop_idx;

    %% Spikes Preprocessing: Binning (20ms), Sqrt Transformation, EMA Smotthing
    preprocessed_data = preprocessing(training_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
    assignin('base', 'preprocessed_data', preprocessed_data); 
    orig_neurons = size(preprocessed_data(1,1).rate, 1);

    %% Remove data from neurons with low firing rates.
    [spikes_mat, ~] = extract_features(preprocessed_data, orig_neurons, stop_idx/bin_group, 'nodebug');
    removed_neurons = remove_neurons(spikes_mat, orig_neurons, 'nodebug');
    neurons = orig_neurons - length(removed_neurons);
    modelParameters.removeneurons = removed_neurons;
    clear spikes_mat

    %% Dimensionality parameters
    pca_threshold = 0.44; % =40 for cov and =0.44 for svd
    lda_dim = 6;
 
    for curr_bin = 1: length(num_bins)
        %% Extract features/restructure data for further analysis
        [spikes_matrix, labels] = extract_features(preprocessed_data, orig_neurons, num_bins(curr_bin), 'nodebug');
        
        %% Remove data from neurons with low firing rates.
        spikes_matrix(removed_neurons, : ) = [];

        %% PCA for dimensionality reduction of the neural data
        [coeff, score, nPC] = perform_PCA(spikes_matrix, pca_threshold, 'cov', 'nodebug');
        % score = (score - mean(score, 2))/std(score, 2);

        %% LDA to maximise class separability across different directions
        [outputs, weights] = perform_LDA(spikes_matrix, score, labels, lda_dim, training_length, 'nodebug');

        %% kNN training: store samples in LDA space with corresponding hand positions
        modelParameters.classify(curr_bin).dPCA_kNN = nPC;
        modelParameters.classify(curr_bin).dLDA_kNN = lda_dim;

        modelParameters.classify(curr_bin).wTrain = weights;
        modelParameters.classify(curr_bin).wTest= outputs;
    
        modelParameters.classify(curr_bin).mean_firing = mean(spikes_matrix, 2);
        modelParameters.classify(curr_bin).labels_kNN = labels(:)';

        % disp(['At bin=',num2str(curr_bin), ...
        %       ', spikes_matrix is ', num2str(size(spikes_matrix,1)), ' x ', num2str(size(spikes_matrix,2))]);
        % disp(['Mean firing is ', num2str(size(mean(spikes_matrix, 2))), ' x ', num2str(size(mean(spikes_matrix, 2)))]);
    end

    %% Hand Positions Preprocessing: Binning (20ms), Centering, Padding
    [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, bin_group, num_bins*bin_group);

    %% PCR
    time_division = kron(bin_group:bin_group:stop_idx, ones(1, neurons)); 
    Interval = start_idx:bin_group:stop_idx;

    % Loop through each direction to model hand positions separately for each.
    
    for directionIndex = 1:length(reaching_angles)
    
        % Extract the current direction's hand position data for all trials.
        currentXPositions = formatted_xPos(:,:,directionIndex);
        currentYPositions = formatted_yPos(:,:,directionIndex);
    
        % Loop through each time window to calculate regression coefficients.
        % These coefficients are used to predict hand positions from neural data.
    
        for timeWindowIndex = 1:((stop_idx-start_idx)/bin_group)+1
    
            % Calculate regression coefficients and the windowed firing rates for the current time window and direction.
    
            [regressionCoefficientsX, regressionCoefficientsY, windowedFiring] = calcRegressionCoefficients(timeWindowIndex, time_division, labels, directionIndex, spikes_matrix, pca_threshold, Interval, currentXPositions, currentYPositions);
            % figure; plot(regressionCoefficientsX, regressionCoefficientsY); title('PCR');
            
            % Store the calculated regression coefficients and the mean windowed firing rates in the model parameters structure.
            modelParameters.pcr(directionIndex,timeWindowIndex).bx = regressionCoefficientsX;
            modelParameters.pcr(directionIndex,timeWindowIndex).by = regressionCoefficientsY;
            modelParameters.pcr(directionIndex,timeWindowIndex).fMean = mean(windowedFiring,1);
    
            % Store the average hand positions across all trials for each time window.
            % These averages can be useful for evaluating the model's performance.
    
            modelParameters.averages(timeWindowIndex).avX = squeeze(mean(xPos,1));
            modelParameters.averages(timeWindowIndex).avY = squeeze(mean(yPos,1));
            
        end
    end    
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

    spike_cells = {training_data.spikes};
    max_time_length = max(cellfun(@(sc) size(sc, 2), spike_cells));
    clear spike_cells;

    % Pad each trial’s spikes out to max_time_length
    for tl = 1:rows
        for dir = 1:cols
            curr_spikes = training_data(tl, dir).spikes; 
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
            % t_new = 1:bin_group:timepoints + 1;
            % num_bins = numel(t_new) - 1;

            binned_spikes = zeros(neurons, num_bins);
            % binned_handPos = zeros(size(train.handPos,1), num_bins);

            for b = 1:num_bins
                start_time = (b-1)*bin_group + 1; % 1, 21, 41, ..., 541
                end_time = b*bin_group; % 20, 40, 60, ..., 560
                if b == num_bins % gets all the leftover points for the last bin
                    binned_spikes(:,b) = sum(train.spikes(:, start_time:end), 2);
                    % binned_handPos(:,b) = mean(train.handPos(:, start_time:end), 2);
                else
                    binned_spikes(:,b) = sum(train.spikes(:, start_time:end_time), 2);
                    % binned_handPos(:,b) = mean(train.handPos(:, start_time:end_time), 2);
                end
            end
            % binned_handPos_centred = bsxfun(@minus, binned_handPos, mean(binned_handPos, 2));
            % preprocessed_data(r,c).handPos = binned_handPos_centred;
            
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


function [spikes_matrix, labels] = extract_features(preprocessed_data, neurons, curr_bin, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Arranging data as:
% rows: 2744 time points --> 98 neurons x 28 bins
% cols: 800 --> 8 angles and 100 trials so angle 1, trial 1; angle 1, trial 2; ...; angle 8, Trial 100
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [rows, cols] = size(preprocessed_data);
    labels = zeros(rows * cols, 1);
    
    for r = 1:rows
        for c = 1:cols
            for k = 1:curr_bin
                c_idx = rows * (c - 1) + r; % 100 (1 - 1) + 1 = 1; 1; 1...x13; 101; 
                r_start = neurons * (k - 1) + 1; % 98 (1 - 1) + 1 = 1; 99; 197;...
                r_end = neurons * k; % 98; 196;...
                spikes_matrix(r_start:r_end,c_idx) = preprocessed_data(r,c).rate(:,k);  
                labels(c_idx) = c; 
            end
        end
    end

    if strcmp(debug, 'debug')
        figure; title(['Firing Rate for Bin ' num2str(curr_bin)]);
        plot(spikes_matrix); 
    end
end


function removed_neurons = remove_neurons(spike_matrix, neurons, debug)
% Remove neurons with very low average firing rate for numerical stability.
    removed_neurons = []; %{}
    % low_fr = [];
    for neuronIdx = 1:neurons
        avgFiringRate = mean(mean(spike_matrix(neuronIdx:neurons:end, :)));
        if avgFiringRate < 0.5
            % low_fr = [low_fr, neuronIdx];
            removed_neurons = [removed_neurons, neuronIdx]; 
        end
    end
    % removed_neurons{end+1} = low_fr;

    if strcmp(debug, 'debug')
        disp(removed_neurons);
    end
end


function [coeff, score, nPC] = perform_PCA(data, threshold, method, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PCA for dimensionality reduction
% Inputs:
% Outputs:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(method, 'cov')
        % nPC = threshold;
        data_centred = data - mean(data,2);
        % C = cov(data_centred);
        C = data_centred' * data_centred;
        [V, D] = eig(C);
        [d, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        explained_variance = cumsum(d) / sum(d);
        nPC = find(explained_variance >= threshold, 1); % Find the number of PCs that explain at least 95% variance
        score = data_centred * V * diag(1./sqrt(d));
        score = score(:, 1:nPC);
        coeff = V(:, 1:nPC);
    end

    if strcmp(method, 'svd')
        variance_threshold = threshold;
        Xc = data - mean(data,2);
        % Perform SVD on the centered data (using economy size decomposition)
        [U, S, V] = svd(Xc, 'econ');
        % Compute variance explained
        singular_values = diag(S);
        explained_variance = (singular_values.^2) / sum(singular_values.^2);
        cum_variance = cumsum(explained_variance);
        nPC = find(cum_variance >= variance_threshold, 1);
        % assignin('base', "nPC", nPC);
        % The principal component directions are given by the columns of V
        coeff = V(:, 1:nPC);
        % Reduce data dimensionality: Compute the projection (scores) of the data onto the principal components
        score = Xc * coeff;
        score = score(:, 1:nPC);
    end

    if strcmp(debug, 'debug')
        figure; plot(score);
    end
end


function [outputs, weights] = perform_LDA(data, score, labels, lda_dim, training_length, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LDA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the LDA projection matrix.
    classes = unique(labels); % gets 1 to 8

    overall_mean = mean(data, 2); % zeros(size(data), length(classes));
    scatter_within = zeros(size(data,1)); % How much do the samples of each class vary around their own class mean?
    scatter_between = zeros(size(data,1)); % How different are the means of the classes from the overall mean?
    
    for i = 1:length(classes)
        % Calculate mean vectors for each direction
        indicies = training_length*(i-1)+1 : i*training_length; % 1, 101, 201.. : 100, 200, 300... 
        % overall_mean(:,i) = mean(data(:, indicies), 2);

        % Mean of current direction
        mean_dir = mean(data(:, indicies), 2);

        % Scatter within (current direction)
        deviation_within = data(:, indicies) - mean_dir;
        scatter_within = scatter_within + deviation_within * deviation_within';

        % Scatter between (current direction)
        deviation_between = mean_dir - overall_mean;
        % scatter_between = scatter_between + length(indicies) * (deviation_between * deviation_between');
        scatter_between = scatter_between + training_length * (deviation_between * deviation_between');
    end
    
    % This reduces the size of the matrices, which can improve numerical stability.
    project_within = score' * scatter_within * score;  
    project_between = score' * scatter_between * score;
   
    % For numerical stability, we use the pseudoinverse of proj_within.
    [V_lda, D_lda] = eig(pinv(project_within) * project_between);

    % Sort eigenvalues and corresponding eigenvectors in descending order.
    [~, sortIdx] = sort(diag(D_lda), 'descend');
    
    % Select the top lda_dimension eigenvectors and form the final projection.
    V_lda = V_lda(:, sortIdx(1:lda_dim));
    
    % The final projection from the original feature space into the LDA space:
    outputs = score * V_lda;  % [features x lda_dimension]
    
    % Maps the mean-centered neural data into the discriminative space.
    weights = outputs' * (data - overall_mean);  % [lda_dimension x samples]

    if strcmp(debug, 'debug')
        figure; plot(outputs); title('Output');
        figure; plot(weights); title('Weight');
    end
end


function [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, bin_group, bins)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hand Position Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    handPos_cells = {training_data.handPos};          % Extract handPos fields into a cell array
    max_trajectory = max(cellfun(@(hp) size(hp, 2), handPos_cells));
    clear handPos_cells;

    [rows, cols] = size(training_data);
    
    xPos = zeros(rows, max_trajectory, cols);
    yPos = zeros(rows, max_trajectory, cols);

    % Pad each trial to padLength
    for c = 1:cols
        for r = 1:rows
            % Mean Centre
            curr_x = training_data(r,c).handPos(1,:);% - mean(training_data(r,c).handPos(1,:)); %training_data(r,c).handPos(1,301);
            curr_y = training_data(r,c).handPos(2,:);% - mean(training_data(r,c).handPos(2,:)); %training_data(r,c).handPos(2,301);
            
            if size(training_data(r,c).handPos,2) < max_trajectory
                pad_size = max_trajectory - size(training_data(r,c).handPos,2);
                if pad_size > 0
                    % Reformat the data by repeating the last element for padding
                    xPos(r, :, c) = [curr_x, repmat(curr_x(end), 1, pad_size)];
                    yPos(r, :, c) = [curr_y, repmat(curr_y(end), 1, pad_size)];
                else
                    xPos(r, :, c) = curr_x;
                    yPos(r, :, c) = curr_y;
                end
            end
        end
    end 
    formatted_xPos = xPos(:, bins, :);
    formatted_yPos = yPos(:, bins, :);
end

function [regressionCoefficientsX, regressionCoefficientsY, FilteredFiring ] = calcRegressionCoefficients(timeWindowIndex, timeDivision, labels, directionIndex, neuraldata, pca_dimension, Interval, currentXPositions, currentYPositions)

% This function calculates regression coefficients for predicting hand positions
% from neural data using Principal Component Analysis (PCA) and a regression model.

% Input:
%   timeWindowIndex : Index of the current time window for analysis.
%   timeDivision : Array indicating the division of time into bins.
%   labels : Array of direction labels corresponding to each trial.
%   directionIndex : Index indicating the current direction of movement being analyzed.
%   neuraldata : Matrix of neural firing rates, potentially filtered by previous steps.
%   pca_dimension : Number of principal components to retain in the PCA.
%   Interval : Array of time intervals for analysis.
%   currentXPositions : Matrix of x-coordinates of hand positions across trials.
%   currentYPositions : Matrix of y-coordinates of hand positions across trials.


% Output:
%   regressionCoefficientsX : Regression coefficients for predicting x-coordinates of hand positions.
%   regressionCoefficientsY : Regression coefficients for predicting y-coordinates of hand positions.
%   FilteredFiring : Neural data filtered by time and direction, used for regression.

    
    % Center the positions for the current time window

    centeredX = bsxfun(@minus, currentXPositions(:, timeWindowIndex), mean(currentXPositions(:, timeWindowIndex)));
    centeredY = bsxfun(@minus, currentYPositions(:, timeWindowIndex), mean(currentYPositions(:, timeWindowIndex)));
    
    % Filter firing data based on time and direction
    FilteredFiring = filterFiringData(neuraldata, timeDivision, Interval(timeWindowIndex), labels, directionIndex);

    % Center the firing data by subtracting the mean of each neuron's firing rate
    centeredWindowFiring = FilteredFiring  - mean(FilteredFiring ,1);

    % Perform PCA on the centered firing data to reduce dimensionality
    [~, principalVectors, ~] = perform_PCA(centeredWindowFiring, pca_dimension, 'cov', 'nodebug');
    principalComponents = principalVectors' * centeredWindowFiring;

    % Calculate ridge regression coefficients for X and Y using the regression matrix
    % regressionMatrix = (principalComponents * principalComponents') \ principalComponents;
    regressionMatrix = pinv(principalComponents * principalComponents') * principalComponents;

    regressionCoefficientsX = principalVectors * regressionMatrix * centeredX;
    regressionCoefficientsY = principalVectors * regressionMatrix * centeredY;

end

function FilteredFiring = filterFiringData(neuraldata, timeDivision, interval, labels, directionIndex)

% This function filters neural firing data based on specified time and direction criteria. 
% It first selects the firing data up to a given time point (interval) and then further 
% filters the data for a specific movement direction. The function finally centers the 
% filtered data by subtracting the mean firing rate across the selected trials for the specific direction.

% Inputs:
%   neuraldata : Matrix of neural firing rates
%   timeDivision : Array that maps each row in neuraldata to a time interval.
%   interval : Scalar specifying the time point up to which the data should be filtered. 
%   labels : Array of labels indicating the direction of movement associated with each column 
%            in 'neuraldata'. This is used to filter the data based on movement direction.
%   directionIndex : Scalar specifying the direction of movement to filter by. Only data columns
%                    (trials) that correspond to this movement direction will be selected.

% Output:
%   FilteredFiring - The resulting matrix of filtered neural firing rates, where the data has been 
%                    filtered to include only the time points up to 'interval' and trials that 
%                    correspond to the specified 'directionIndex'.


    % Filter the neural data to include only time points up to 'interval'
    timeFilter = timeDivision <= interval;
    % Further filter the data to include only trials corresponding to 'directionIndex'
    directionFilter = labels == directionIndex;
    FilteredFiring  = neuraldata(timeFilter, :);
    % Center the filtered data by subtracting the mean firing rate across the selected trials
    % for the specific direction. 
    FilteredFiring  = FilteredFiring (:, directionFilter) - mean(FilteredFiring(:, directionFilter), 1);
end
