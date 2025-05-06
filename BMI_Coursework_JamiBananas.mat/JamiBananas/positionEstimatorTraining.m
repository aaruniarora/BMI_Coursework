function modelParameters = positionEstimatorTraining(training_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSITION ESTIMATOR TRAINING FUNCTION
%
% Trains a full decoding model to estimate hand positions from neural spikes.
% Pipeline:
%   1. Preprocesses neural data:
%       - Pads all trials to max length
%       - Bins spikes in 20 ms intervals
%       - Applies smoothing filter (EMA or Gaussian)
%   2. Removes neurons with low firing rate (< 0.5 spk/s)
%   3. Extracts features and assigns direction labels
%   4. Applies PCA for dimensionality reduction
%   5. Applies LDA to find class-discriminative features
%   6. Stores features and labels for later kNN decoding
%   7. Trains a regression model (PCR - ridge and lasso optional) to map spikes to (x,y)
%
% Outputs:
%   modelParameters - struct storing all learned parameters:
%       - preprocessing config
%       - removed neurons
%       - PCA/LDA matrices
%       - regression coefficients
%       - training data in discriminative space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %% Parameters
    [training_length, directions] = size(training_data); 
    bin_group = 20; % Time bin width in ms
    alpha = 0.35; % hypertuned
    sigma = 50;  % standard deviation in ms
    start_idx = 300 + bin_group; 

    % Find min time length
    spike_cells = {training_data.spikes};  % Extract spike fields into a cell array
    min_time_length = min(cellfun(@(sp) size(sp, 2), spike_cells(:))); 
    clear spike_cells;
    
    % Calculate stop_index based on bin_group
    stop_idx = floor((min_time_length - start_idx) / bin_group) * bin_group + start_idx;
    time_bins = start_idx:bin_group:stop_idx;  % e.g. 320:20:560
    num_bins = time_bins / bin_group; % eg. 13

    % Store in modelParameters
    modelParameters = struct;
    modelParameters.start_idx = start_idx;
    modelParameters.stop_idx  = stop_idx;
    modelParameters.bin_group = bin_group;
    modelParameters.directions = directions;
    modelParameters.trial_id = 0;
    modelParameters.iterations = 0;

    %% Spikes Preprocessing: Binning (20ms), Sqrt Transformation, EMA Smoothing
    preprocessed_data = preprocessing(training_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
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
        [~, score, nPC] = perform_PCA(spikes_matrix, pca_threshold, 'nodebug', orig_neurons, removed_neurons);

        %% LDA to maximise class separability across different directions
        [outputs, weights] = perform_LDA(spikes_matrix, score, labels, lda_dim, training_length, 'nodebug');

        %% kNN training: store samples in LDA space with corresponding hand positions
        modelParameters.class(curr_bin).PCA_dim = nPC;
        modelParameters.class(curr_bin).LDA_dim = lda_dim;

        modelParameters.class(curr_bin).lda_weights = weights;
        modelParameters.class(curr_bin).lda_outputs= outputs;
    
        modelParameters.class(curr_bin).mean_firing = mean(spikes_matrix, 2);
        modelParameters.class(curr_bin).labels = labels(:)';
    end

    %% Hand Positions Preprocessing: Binning (20ms), Centering, Padding
    [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, num_bins*bin_group);
    
    %% PCR
    poly_degree = 1;
    modelParameters.polyd = poly_degree;
    reg_meth = 'standard';
    modelParameters.reg_meth = reg_meth;

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
            [reg_coeff_X, reg_coeff_Y, win_firing] = calc_reg_coeff(win_idx, time_division, labels, ...
                dir_idx, spikes_matrix, pca_threshold, time_interval, curr_X_pos, curr_Y_pos,poly_degree, reg_meth);
            % figure; plot(regressionCoefficientsX, regressionCoefficientsY); title('PCR');
            
            % Store in model parameters
            modelParameters.pcr(dir_idx,win_idx).bx = reg_coeff_X;
            modelParameters.pcr(dir_idx,win_idx).by = reg_coeff_Y;
            modelParameters.pcr(dir_idx,win_idx).f_mean = mean(win_firing,2);
    
            % And store the mean hand positions across all trials for each time window   
            modelParameters.averages(win_idx).av_X = squeeze(mean(xPos,1));
            modelParameters.averages(win_idx).av_Y = squeeze(mean(yPos,1));
            
        end
    end    
end

%% HELPER FUNCTIONS FOR PREPROCESSING OF SPIKES

function preprocessed_data = preprocessing(training_data, bin_group, filter_type, alpha, sigma, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing function for spike trains
%
% Input:
%   training_data - original neural dataset with spikes and hand positions
%   bin_group     - number of ms per time bin (e.g., 20 ms)
%   filter_type   - 'EMA' or 'Gaussian'
%   alpha         - EMA smoothing constant (0 < alpha < 1)
%   sigma         - std deviation for Gaussian smoothing (in ms)
%   debug         - if 'debug', plots are shown
%
% Output:
%   preprocessed_data - struct with binned, smoothed firing rates per trial
%
% Steps:
%   1. Pads all trials to max trial time length
%   2. Bins spike counts over `bin_group` intervals
%   3. Applies square root transform
%   4. Applies either a recursive filter, exponential moving average (EMA),
%      or Gaussian smoothing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialise
    [rows, cols] = size(training_data); 
    preprocessed_data = struct;

    spike_cells = {training_data.spikes};
    max_time_length = max(cellfun(@(sc) size(sc, 2), spike_cells));
    clear spike_cells;

    % Fill NaNs with 0's and pad each trial’s spikes out to max_time_length
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

    if strcmp(debug, 'debug')
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


function gKernel = gaussian_filter(bin_group, sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates a normalized 1D Gaussian kernel for convolution
% Inputs:
%   bin_group - bin size in ms
%   sigma     - standard deviation of the Gaussian in ms
% Output:
%   gKernel   - 1D vector of Gaussian filter values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Create a 1D Gaussian kernel centered at zero.
    gaussian_window = 10*(sigma/bin_group);
    e_std = sigma/bin_group;
    alpha = (gaussian_window-1)/(2*e_std);

    time_window = -(gaussian_window-1)/2:(gaussian_window-1)/2;
    gKernel = exp((-1/2) * (alpha * time_window/((gaussian_window-1)/2)).^2)';
    gKernel = gKernel / sum(gKernel);
end


function ema_spikes = ema_filter(sqrt_spikes, alpha, num_neurons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Applies exponential moving average (EMA) smoothing to spike data
% Inputs:
%   sqrt_spikes  - sqrt-transformed spike matrix [neurons x time bins]
%   alpha        - smoothing factor (higher = more recent weight)
%   num_neurons  - number of input neurons
% Output:
%   ema_spikes   - smoothed spike matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:size(sqrt_spikes, 2)
            ema_spikes(n, t) = alpha * sqrt_spikes(n, t) + (1 - alpha) * ema_spikes(n, t - 1);
        end
    end
end


function data = fill_nan(data, data_type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fills NaN values in spike or hand position data
% For spikes the NaN values are replaced with 0's and for hand position
% data we perform a forward then a backward fill.
% Inputs:
%   data       - input vector/matrix
%   data_type  - 'spikes' or 'handpos'
% Output:
%   data       - cleaned data with NaNs filled
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

function [spikes_matrix, labels] = extract_features(preprocessed_data, neurons, curr_bin, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Converts 2D spike data into a 2D matrix of features across bins
% In our case, rearranging data as:
% rows: 2744 time points --> 98 neurons x 28 bins
% cols: 800 --> 8 angles and 100 trials so angle 1, trial 1; angle 1, trial 2; ...; angle 8, Trial 100
%
% Inputs:
%   preprocessed_data - output from preprocessing function
%   neurons           - number of neurons before filtering
%   curr_bin          - number of bins to include (time window)
%   debug             - 'debug' enables plotting
%
% Outputs:
%   spikes_matrix     - matrix [neurons*curr_bin x trials]
%   labels            - direction labels for each column (trial)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identifies and returns neurons with average firing rate < 0.5 Hz
% Remove neurons with very low average firing rate for numerical stability.
%
% Inputs:
%   spike_matrix - matrix of spike data [neurons*bin x trials]
%   neurons      - original number of neurons
% Output:
%   removed_neurons - indices of low-firing neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    removed_neurons = []; 
    for neuronIdx = 1:neurons
        avgFiringRate = mean(mean(spike_matrix(neuronIdx:neurons:end, :)));
        if avgFiringRate < 0.5
            removed_neurons = [removed_neurons, neuronIdx]; 
        end
    end

    if strcmp(debug, 'debug')
        disp(removed_neurons);
    end
end

%% HELPER FUNCTION FOR DIMENSIONALITY REDUCTION

function [coeff, score, nPC] = perform_PCA(data, threshold, debug, orig_neurons, removed_neurons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs Principal Component Analysis (PCA)
%
% Inputs:
%   data        - neural feature matrix [neurons x trials]
%   threshold   - cumulative variance threshold (e.g., 0.44)
%   debug       - if 'debug', plot score
%
% Outputs:
%   coeff       - principal component coefficients (eigenvectors)
%   score       - projected data in PCA space
%   nPC         - number of PCs meeting variance threshold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % nPC = threshold;
    data_centred = data - mean(data,2);
    % C = cov(data_centred);
    C = data_centred' * data_centred;
    [V, D] = eig(C);
    [d, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    explained_variance = cumsum(d) / sum(d);
    nPC = find(explained_variance >= threshold, 1); % Find the number of PCs that explain at least 44% variance
    score = data_centred * V * diag(1./sqrt(d));
    score = score(:, 1:nPC);
    coeff = V(:, 1:nPC);

    if strcmp(debug, 'debug')
        % figure; plot(score);
        
        kept_neurons = setdiff(1:orig_neurons, removed_neurons);
        F = numel(kept_neurons);           % #kept neurons
        for pc = 1:nPC
            [~, sorted] = sort(abs(coeff(:,pc)), 'descend');
            topRows    = sorted(1:5);      % Take more (like 20) to ensure after unique we have enough
            topNeuronIdx = mod(topRows-1,F)+1;  % This maps row to neuron (fixes bin offset)
            topNeuronIDs = kept_neurons(topNeuronIdx);
            topNeuronIDs = unique(topNeuronIDs, 'stable'); % keep only unique neurons
            top5 = topNeuronIDs(1:min(5,length(topNeuronIDs))); % take top 5 if possible
            fprintf('PC%d=%s\n', pc, mat2str(top5));
        end

    end
end


function [outputs, weights] = perform_LDA(data, score, labels, lda_dim, ...
    training_length, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs Linear Discriminant Analysis (LDA) on PCA-transformed data.
% This method is called MDF, most discriminant feature, extraction.
%
% Inputs:
%   data         - original neural data
%   score        - PCA-transformed data
%   labels       - movement direction labels
%   lda_dim      - desired number of LDA components
%   training_len - number of trials per direction
%   debug        - plot if 'debug'
%
% Outputs:
%   outputs      - LDA-transformed features
%   weights      - projection of original data into LDA space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute the LDA projection matrix.
    classes = unique(labels); % gets 1 to 8

    overall_mean = mean(data, 2); % zeros(size(data), length(classes));
    scatter_within = zeros(size(data,1)); % How much do the samples of each class vary around their own class mean?
    scatter_between = zeros(size(data,1)); % How different are the means of the classes from the overall mean?
    
    for i = 1:length(classes)
        % Calculate mean vectors for each direction
        indicies = training_length*(i-1)+1 : i*training_length; % 1, 101, 201.. : 100, 200, 300... 

        % Mean of current direction
        mean_dir = mean(data(:, indicies), 2);

        % Scatter within (current direction)
        deviation_within = data(:, indicies) - mean_dir;
        scatter_within = scatter_within + deviation_within * deviation_within';

        % Scatter between (current direction)
        deviation_between = mean_dir - overall_mean;
        scatter_between = scatter_between + training_length * (deviation_between * deviation_between');
    end
    
    % Reduce the size of the matrices with PCA to improve numerical stability
    project_within = score' * scatter_within * score;  
    project_between = score' * scatter_between * score;
    
    % Sorting eigenvalues and eigenvectors in descending order
    [V_lda, D_lda] = eig(pinv(project_within) * project_between);
    [~, sortIdx] = sort(diag(D_lda), 'descend'); 
    
    % Selects the given lda_dimension eigenvectors to form the final
    % projection (from original feature space to LDA space)
    V_lda = V_lda(:, sortIdx(1:lda_dim));
    outputs = score * V_lda;  % [features x lda_dimension]
    
    % Mapping the mean-centered neural data to the discriminative space
    weights = outputs' * (data - overall_mean);  % [lda_dimension x samples]

    if strcmp(debug, 'debug')
        figure; plot(outputs); title('Output');
        figure; plot(weights); title('Weight');
    end
end

%% HELPER FUNCTION FOR PREPROCESSING OF HAND POSITIONS

function [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, bins)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pads and extracts hand position data aligned to spike bins
%
% Inputs:
%   training_data - input dataset with handPos field
%   bins          - selected bin indices (e.g., 320:20:560)
%
% Outputs:
%   xPos, yPos          - padded x and y positions [trials x time x dirs]
%   formatted_xPos/yPos - same data but indexed by bin times
%
% Steps:
%   1. Forward and then backward fill NaN values 
%   2. Pads all trials with the last value to max trial time length
%   3. Bins hand positions over `bin_group` intervals
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
            curr_x = training_data(r,c).handPos(1,:);
            curr_y = training_data(r,c).handPos(2,:);
            
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
                    % For no padding, just copy the original data
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
    spikes_matrix, pca_thresh, time_interval, curr_X_pos, curr_Y_pos,poly_degree,method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates regression coefficients mapping spikes to hand positions
% using Principal Component Regression (PCR)
%
% Inputs:
%   win_idx       - current time window index
%   time_div      - array of time divided into bins
%   labels        - trial direction labels
%   dir_idx       - current movement direction index
%   spikes_matrix - feature matrix of preprocessed neural firing rates
%   pca_thresh    - PCA variance threshold
%   time_interval - vector of bin centers
%   curr_X_pos    - current matrix of x-coordinates of hand pos for direction
%   curr_Y_pos    - current matrix of y-coordinates of hand pos for direction
%   poly_degree   - polynomial regression order
%   method        - regression type: 'standard', 'ridge', or 'lasso'or
%   'poly'
%
% Outputs:
%   reg_coeff_X/Y - learned regression coefficients for x and y hand
%   positions
%   filtered_firing - centered neural data for this bin and direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Mean centering the hand positions for the current time window
    centered_X = bsxfun(@minus, curr_X_pos(:, win_idx), mean(curr_X_pos(:, win_idx)));
    centered_Y = bsxfun(@minus, curr_Y_pos(:, win_idx), mean(curr_Y_pos(:, win_idx)));
    
    % Filtering firing data based on time and direction and centering it
    filtered_firing = filter_firing_rate(spikes_matrix, time_div, time_interval(win_idx), labels, dir_idx);
    centered_win_firing = filtered_firing  - mean(filtered_firing ,1);

    % Performing PCA for dimensionality reduction and numerical stability
    [~, score, nPC] = perform_PCA(centered_win_firing, pca_thresh, 'nodebug');
    principal_components = score' * centered_win_firing;  % (n_components x n_samples)
    X = principal_components';                            % Transpose: (n_samples x n_components)

    % Project data onto top principal components for polynomial regression
    Z = score(:, 1:nPC)' * centered_win_firing;  % [pca_dim x samples]

    % Select regression method and calculate X and Y regression coefficients 
    if nargin < 10, method = 'standard'; end
    switch lower(method)
        case 'standard'
            reg_mat = pinv(principal_components * principal_components') * principal_components;
            reg_coeff_X = score * reg_mat * centered_X;
            reg_coeff_Y = score * reg_mat * centered_Y;

        case 'poly'
            % Expand X with polynomial features
            X_poly = [];
            for p = 1:poly_degree
                X_poly = [X_poly; Z.^p];  % Use , to concatenate along columns (features)
            end
            % Compute regression coefficients for centered positions
            reg_coeff_X = (X_poly * X_poly') \ (X_poly * centered_X);
            reg_coeff_Y = (X_poly * X_poly') \ (X_poly * centered_Y);
            modelParameters.poly_1 =  1;

        case 'ridge'
            lambda = 1; % regularisation parameter (to be tuned)
            % Ridge Regression: (X'X + λI)^(-1) X'y
            reg_coeff_X = score * ((X' * X + lambda * eye(size(X, 2))) \ (X' * centered_X));
            reg_coeff_Y = score * ((X' * X + lambda * eye(size(X, 2))) \ (X' * centered_Y));

        case 'lasso'
            lambda = 0.1; % regularisation parameter (to be tuned)
            % Lasso must be done column-wise because it solves for one response variable at a time
            % Bx = lasso(X, centered_X, 'Lambda', lambda); % stats toolbox
            % By = lasso(X, centered_Y, 'Lambda', lambda); % stats toolbox
            
            Bx = zeros(size(X, 2), 1);
            By = zeros(size(X, 2), 1);
            for d = 1:size(centered_X, 2)
                Bx(:, d) = custom_lasso(X, centered_X(:, d), lambda);
                By(:, d) = custom_lasso(X, centered_Y(:, d), lambda);
            end

            reg_coeff_X = score * Bx;
            reg_coeff_Y = score * By;

        case 'lms'
            mu = 1e-5;  % Smaller learning rate for stability
            num_epochs = 10;
        
            % Initialize weights
            [~, num_features] = size(X);
            W_x = zeros(num_features, 1);
            W_y = zeros(num_features, 1);
        
            % Clean up any NaNs or Infs in data
            valid_idx = all(isfinite(X), 2) & isfinite(centered_X) & isfinite(centered_Y);
            X_clean = X(valid_idx, :);
            Yx_clean = centered_X(valid_idx);
            Yy_clean = centered_Y(valid_idx);
        
            % Normalize each row to unit norm to avoid large gradients
            X_norms = sqrt(sum(X_clean.^2, 2)) + 1e-8;
            X_clean = bsxfun(@rdivide, X_clean, X_norms);
        
            % LMS training loop
            for epoch = 1:num_epochs
                for i = 1:size(X_clean, 1)
                    xi = X_clean(i, :);
                    err_x = Yx_clean(i) - xi * W_x;
                    err_y = Yy_clean(i) - xi * W_y;
        
                    W_x = W_x + mu * xi' * err_x;
                    W_y = W_y + mu * xi' * err_y;
                end
            end
        
            % Final LMS weights in PCA space
            reg_coeff_X = score * W_x;
            reg_coeff_Y = score * W_y;


        otherwise
            error('Unknown regression method: choose standard, ridge, or lasso');
    end
end


function B = custom_lasso(X, y, lambda, max_iter, tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs Lasso regression using coordinate descent
% Inputs:
%   X        - [samples x features] predictor matrix
%   y        - [samples x 1] response vector
%   lambda   - regularisation parameter
%   max_iter - max iterations (default: 1000)
%   tol      - convergence threshold (default: 1e-4)
% Output:
%   B        - regression coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin < 4, max_iter = 1000; end
    if nargin < 5, tol = 1e-4; end

    [~, p] = size(X);
    B = zeros(p, 1);
    % Xy = X' * y;
    X_sq_sum = sum(X.^2);

    for iter = 1:max_iter
        B_old = B;

        for j = 1:p
            % Partial residual
            r_j = y - X * B + X(:, j) * B(j);

            % Update coordinate j
            rho = X(:, j)' * r_j;

            if rho < -lambda / 2
                B(j) = (rho + lambda / 2) / X_sq_sum(j);
            elseif rho > lambda / 2
                B(j) = (rho - lambda / 2) / X_sq_sum(j);
            else
                B(j) = 0;
            end
        end

        % Check for convergence
        if norm(B - B_old, 2) < tol
            break;
        end
    end
end



function filtered_firing = filter_firing_rate(spikes_matrix, time_div, time_interval, labels, dir_idx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filters spike data by time and direction, centers trials
% Inputs:
%   spikes_matrix  - full feature matrix of preprocessed neural firing 
%                    rates [neurons*time x trials]
%   time_div       - time bin identifier for each row
%   time_interval  - current time bin (upto which the data should be
%                    filtered)
%   labels         - direction labels for each trial (column vector)
%   dir_idx        - target direction
% Output:
%   filtered_firing - centered data for this time and direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Trim the spikes data upto the time point of the time_interval
    trimmed_time = time_div <= time_interval;
    % Then filter the data for a particular direction 
    dir_filter = labels == dir_idx;
    filtered_firing  = spikes_matrix(trimmed_time, :);
    % Mean centre
    filtered_firing  = filtered_firing (:, dir_filter) - mean(filtered_firing(:, dir_filter), 1);
end