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
    [training_length, directions] = size(training_data); 
    bin_group = 20;
    alpha = 0.35; % arbitrary value decided through multiple trials
    start_idx = 320; 

    spike_cells = {training_data.spikes};  % Extract spike fields into a cell array
    min_time_length = min(cellfun(@(sp) size(sp, 2), spike_cells(:))); % Find min time length
    clear spike_cells;
    
    stop_idx = floor((min_time_length - start_idx) / bin_group) * bin_group + start_idx;
    time_bins = start_idx:bin_group:stop_idx;  % e.g. 320:20:560
    num_bins = time_bins / bin_group;

   % Store in modelParameters
    modelParameters = struct;
    modelParameters.start_idx = start_idx;
    modelParameters.stop_idx  = stop_idx;

    %% Spikes Preprocessing: Binning (20ms), Sqrt Transformation, EMA Smotthing
    preprocessed_data = preprocessing(training_data, bin_group, alpha);
    orig_neurons = size(preprocessed_data(1,1).rate, 1);

    %% Remove data from neurons with low firing rates.
    [spikes_mat, ~] = extract_features(preprocessed_data, orig_neurons, stop_idx/bin_group);
    removed_neurons = remove_neurons(spikes_mat, orig_neurons);
    neurons = orig_neurons - length(removed_neurons);
    modelParameters.removeneurons = removed_neurons;
    clear spikes_mat

    %% Dimensionality parameters
 
    for curr_bin = 1: length(num_bins)
        %% Extract features/restructure data for further analysis
        [spikes_matrix, labels] = extract_features(preprocessed_data, orig_neurons, num_bins(curr_bin));
        
        %% Remove data from neurons with low firing rates.
        spikes_matrix(removed_neurons, : ) = [];

        %% PCA for dimensionality reduction of the neural data
        pca_threshold = 0.42; % =40 for cov and =0.44 for svd
        [~, score, nPC] = perform_PCA(spikes_matrix, pca_threshold);

        %% LDA to maximise class separability across different directions
        lda_dim = 6;
        [outputs, weights] = perform_LDA(spikes_matrix, score, labels, lda_dim, training_length);

        %% kNN training: store samples in LDA space with corresponding hand positions
        modelParameters.classify(curr_bin).dPCA_kNN = nPC;
        modelParameters.classify(curr_bin).dLDA_kNN = lda_dim;

        modelParameters.classify(curr_bin).wTrain = weights;
        modelParameters.classify(curr_bin).wTest= outputs;
    
        modelParameters.classify(curr_bin).mean_firing = mean(spikes_matrix, 2);
        modelParameters.classify(curr_bin).labels_kNN = labels(:)';
    end

    %% Hand Positions Preprocessing: Binning (20ms), Centering, Padding
    [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, num_bins*bin_group);

    %% PCR
    time_division = kron(bin_group:bin_group:stop_idx, ones(1, neurons)); 
    time_interval = start_idx:bin_group:stop_idx;

    % modelling hand positions separately for each direction
    for dir_idx = 1:directions
    
        % Extract the current direction's hand position data for all trials
        curr_X_pos = formatted_xPos(:,:,dir_idx);
        curr_Y_pos = formatted_yPos(:,:,dir_idx);
    
        % Loop through each time window to calculate regression coefficients that predict hand positions from neural data
        for win_idx = 1:((stop_idx-start_idx)/bin_group)+1
    
            % Calculate regression coefficients and the windowed firing rates for the current time window and direction
            [reg_coeff_X, reg_coeff_Y, win_firing] = calc_reg_coeff(win_idx, time_division, labels, dir_idx, ...
                spikes_matrix, pca_threshold, time_interval, curr_X_pos, curr_Y_pos);
            % figure; plot(regressionCoefficientsX, regressionCoefficientsY); title('PCR');
            
            % Store in model parameters
            modelParameters.pcr(dir_idx,win_idx).bx = reg_coeff_X;
            modelParameters.pcr(dir_idx,win_idx).by = reg_coeff_Y;
            modelParameters.pcr(dir_idx,win_idx).f_mean = mean(win_firing,2);
    
            % And store the mean hand positions across all trials for each time window   
            modelParameters.averages(win_idx).avX = squeeze(mean(xPos,1));
            modelParameters.averages(win_idx).avY = squeeze(mean(yPos,1));
            
        end
    end    
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

%% HELPER FUNCTIONS FOR FEATURE EXTRACTION

function [spikes_matrix, labels] = extract_features(preprocessed_data, neurons, curr_bin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rearranging data as:
% rows: 2744 time points --> 98 neurons x 28 bins
% cols: 800 --> 8 angles and 100 trials so angle 1, trial 1; angle 1, trial 2; ...; angle 8, Trial 100
% Inputs:
    % preprocessed_data: input training data containing the spikes and hand positions
    % neurons: number of input neurons
    % curr_bin: current bin resolution in milliseconds
% Outputs:
    % spikes_matrix: rearranged data
    % labels: direction labels for spikes_matrix (column matrix)
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
end


function removed_neurons = remove_neurons(spike_matrix, neurons)
    % Remove neurons with very low average firing rate for numerical stability.
    removed_neurons = []; 
    for neuronIdx = 1:neurons
        avgFiringRate = mean(mean(spike_matrix(neuronIdx:neurons:end, :)));
        if avgFiringRate < 0.5
            removed_neurons = [removed_neurons, neuronIdx]; 
        end
    end
end

%% HELPER FUNCTION FOR DIMENSIONALITY REDUCTION

function [coeff, score, nPC] = perform_PCA(data, threshold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs principal component analysis on the given data based on a given
% threshold value.
% Inputs:
    % data: 
    % threshold: 
    % method:  
% Outputs:
    % coeff:
    % score:
    % nPC:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


function [outputs, weights] = perform_LDA(data, score, labels, lda_dim, training_length)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs least discriminant analysis on the given data after PCA has
% already been performed. This method is called MDF, most discriminant
% feature.
% Input:
% Outputs:
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
    
    % reduce the size of the matrices --> can improve numerical stability
    project_within = score' * scatter_within * score;  
    project_between = score' * scatter_between * score;
    
    % sort eigenvalues and eigenvectors in descending order
    [V_lda, D_lda] = eig(pinv(project_within) * project_between);
    [~, sortIdx] = sort(diag(D_lda), 'descend'); 
    
    % Selects the given lda_dimension eigenvectors to form the final
    % projection (from original feature space to LDA space)
    V_lda = V_lda(:, sortIdx(1:lda_dim));
    outputs = score * V_lda;  % [features x lda_dimension]
    
    % Mapping the mean-centered neural data to the discriminative space
    weights = outputs' * (data - overall_mean);  % [lda_dimension x samples]
end

%% HELPER FUNCTION FOR PREPROCESSING OF HAND POSITIONS

function [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, bins)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hand Position Preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    handPos_cells = {training_data.handPos};
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
            
            % Fill missing values in hand position
            curr_x = fill_nan(curr_x, 'handpos');
            curr_y = fill_nan(curr_y, 'handpos');

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

%% HELPER FUNCTION FOR PCR

function [reg_coeff_X, reg_coeff_Y, filtered_firing ] = ...
calc_reg_coeff(win_idx, time_div, labels, dir_idx, ...
spikes_matrix, pca_dimension, time_interval, curr_X_pos, curr_Y_pos)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform principal component regression to calculate regression coefficients
% that predict hand positions from neural spikes data.
% Input:
%   win_idx : current time window index
%   time_div : array of time divided into bins
%   labels : direction labels corresponding to each trial
%   dir_idx : current movement direction index
%   spikes_matrix : preprocessed matrix of neural firing rates
%   pca_dimension : percentage of principal components to retain in the PCA
%   time_interval : time intervals array
%   curr_X_pos : given matrix of hand positions in the x-coordinate across trials
%   curr_Y_pos : given matrix of hand positions in the y-coordinate across trials
% Output:
%   reg_coeff_X : Regression coefficients for predicting x-coordinates of hand positions.
%   reg_coeff_Y : Regression coefficients for predicting y-coordinates of hand positions.
%   filtered_firing : Neural data filtered by time and direction, used for regression.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Mean centering the hand positions for the current time window
    centered_X = bsxfun(@minus, curr_X_pos(:, win_idx), mean(curr_X_pos(:, win_idx)));
    centered_Y = bsxfun(@minus, curr_Y_pos(:, win_idx), mean(curr_Y_pos(:, win_idx)));
    
    % Filter firing data based on time and direction
    filtered_firing = filter_firing_rate(spikes_matrix, time_div, time_interval(win_idx), labels, dir_idx);

    % Center the firing data by subtracting the mean of each neuron's firing rate
    centered_win_firing = filtered_firing  - mean(filtered_firing ,1);

    % Perform PCA on the centered firing data to reduce dimensionality
    [~, score, ~] = perform_PCA(centered_win_firing, pca_dimension);
    principal_components = score' * centered_win_firing;

    % Calculate ridge regression coefficients for X and Y using the regression matrix
    reg_mat = pinv(principal_components * principal_components') * principal_components;

    reg_coeff_X = score * reg_mat * centered_X;
    reg_coeff_Y = score * reg_mat * centered_Y;

end

function filtered_firing = filter_firing_rate(spikes_matrix, time_div, time_interval, labels, dir_idx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function filters neural firing data based on specified time and direction criteria. 
% It first selects the firing data up to a given time point (interval) and then further 
% filters the data for a specific movement direction. The function finally centers the 
% filtered data by subtracting the mean firing rate across the selected trials for the specific direction.

% Inputs:
%   spikes_matrix : preprocessed matrix of neural firing rates
%   time_div : array mapping spikes_matrix rows into the given time bins
%   time_interval : scalar for time point up to which the data should be filtered
%   labels : column vector of direction labels corresponding to each trial
%   dir_idx : scalar for direction of movement to filter for.
% Output:
%   filtered_firing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timeFilter = time_div <= time_interval;
    directionFilter = labels == dir_idx;
    filtered_firing  = spikes_matrix(timeFilter, :);
    filtered_firing  = filtered_firing (:, directionFilter) - mean(filtered_firing(:, directionFilter), 1);
end
