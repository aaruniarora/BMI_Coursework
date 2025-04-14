function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSITION ESTIMATOR
%
% Uses trained model parameters to:
%   1. Preprocess incoming spike data (binning, sqrt, smoothing)
%   2. Extract and reshape features from the current trial
%   3. Classify intended movement direction using soft kNN
%   4. Predict x and y hand position using regression coefficients
%
% Inputs:
%   test_data        - A struct representing a single trial
%   modelParameters  - Learned parameters from training
%
% Outputs:
%   x, y             - Estimated hand position (in mm)
%   modelParameters  - Updated with current predicted direction label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Parameters
    bin_group = 20; % Time bin width in ms
    alpha = 0.3; % EMA smoothing factor
    sigma = 50; % Std. deviation for Gaussian filter

    start_idx = modelParameters.start_idx;
    stop_idx = modelParameters.stop_idx;
    directions = modelParameters.directions; % get the number of angles
    polyDegree = modelParameters.polyd;

    %% Soft kNN parameters
    k = 20;    % Number of neighbors for kNN (8 for hard kNN and 20 for soft)
    pow = 1;   % Power factor for distance-based weighting
    alp = 1e-6; % Scaling for exponential weighting

   %% Preprocess the trial data
   preprocessed_test = preprocessing(test_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
   neuron_len = size(preprocessed_test(1,1).rate, 1);

   %% Use indexing based on data given
   curr_bin = size(test_data.spikes, 2);
   idx = min( max( floor( ( curr_bin - start_idx ) / bin_group) + 1, 1), length(modelParameters.class));

   %%  Remove low firing neurons for better accuracy
   spikes_test = extract_features(preprocessed_test, neuron_len, curr_bin/bin_group, 'nodebug');
   removed_neurons = modelParameters.removeneurons;
   spikes_test(removed_neurons, :) = [];

   %% Reshape dataset: Flatten spike data into column vector
   spikes_test = reshape(spikes_test, [], 1);

   %% Predict movement direction using kNN classification
   if curr_bin <= stop_idx 
       % Extract LDA projections and mean firing for the current bin
       train_weight = modelParameters.class(idx).lda_weights;
       test_weight =  modelParameters.class(idx).lda_outputs;
       curr_firing_mean = modelParameters.class(idx).mean_firing;
       
       % Project test spike vector to LDA space
       test_weight = test_weight' * (spikes_test(:) - curr_firing_mean(:));

       % Classify using hard or soft kNN. Soft kNN can be distance (dist) or exponential (exp) based weighting
       output_label = KNN_classifier(directions, test_weight, train_weight, k, pow, alp, 'soft', 'dist');

   else 
       % After max time window, retain previous classification
       output_label = modelParameters.actualLabel;
   end

    modelParameters.actualLabel = output_label; 
    
    %% Estimate hand position (x, y) using PCR model
    av_X = modelParameters.averages(idx).av_X(:, output_label);
    av_Y =  modelParameters.averages(idx).av_Y(:, output_label);
    meanFiring = modelParameters.pcr(output_label, idx).f_mean;
    bx = modelParameters.pcr(output_label, idx).bx;
    by = modelParameters.pcr(output_label, idx).by;
    reg_meth = modelParameters.reg_meth;

    x = position_calc(spikes_test, meanFiring, bx, av_X, curr_bin,reg_meth,polyDegree);
    y = position_calc(spikes_test, meanFiring, by, av_Y, curr_bin,reg_meth,polyDegree);
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


%% HELPER FUNCTION TO PERFORM CLASSIFICATION WITH kNN

function output_label = KNN_classifier(directions, test_weight, train_weight, NN_num, pow, alp, method, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predicts direction label using either hard or soft k-nearest neighbours (kNN).
%
% Inputs:
%   test_weight  - [lda_dim x 1] test sample in LDA space
%   train_weight - [lda_dim x n_samples] training set in LDA space
%   NN_num       - neighbor divisor for k calculation
%   pow          - exponent for inverse-distance weighting (soft)
%   alp          - alpha for exponential soft weighting
%   method       - 'hard' or 'soft'
%   type         - 'dist' (1/distance^pow) or 'exp' (exp(-alpha * d))
%
% Output:
%   output_lbl   - predicted movement direction label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    train_len = size(train_weight, 2) / directions; 
    k = max(1, round(train_len / NN_num)); 

    output_label = zeros(1, size(test_weight, 2));
    
    for i = 1:size(test_weight, 2)
        % For the i-th test sample:
        distances = sum((train_weight - test_weight(:, i)).^2, 1);

        % Sort and get top-k nearest neighbors
        [sorted_dist, sorted_idx] = sort(distances, 'ascend');
        nearest_idx    = sorted_idx(1:k);
        nearest_dist   = sorted_dist(1:k);

        % Convert index -> direction label
        % If train_weight is grouped angle-by-angle, we do this:
        train_labels = ceil(nearest_idx / train_len);  % each from 1..8

        switch method
            case 'soft'

                % Compute distance-based weights, e.g. 1/d^2
                if strcmp(type, 'dist')
                    weights = 1 ./ (nearest_dist.^pow + eps);
                
                elseif strcmp(type, 'exp')
                    weights = exp(-alp .* nearest_dist);
                end
        
                % Sum up weights for each angle
                angle_weights = zeros(1, directions);
                
                for nn = 1:k
                    angle = train_labels(nn); 
                    angle_weights(angle) = angle_weights(angle) + weights(nn);
                end
                
                % % Final predicted label is the angle with the highest sum of weights
                % [~, best_angle] = max(angleWeights);
                % output_label(i) = best_angle;
        
                % Or we can use probability distribution
                p = angle_weights / sum(angle_weights);
                [~, best_angle] = max(p);
                output_label(i) = best_angle;
        
            case 'hard'
                output_label(i) = mode(train_labels);
        
            otherwise
                error("Incorrect kNN method! Choose between 'hard' and 'soft'.");
        end
    end
end


%% HELPER FUNCTION FOR POSITION CALCULATION

function pos = position_calc(spikes_matrix, firing_mean, b, avg, curr_bin,reg_meth,polyd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates hand position using linear regression from spike activity
%
% Inputs:
%   spikes_matrix - spike vector (after preprocessing and reshaping)
%   firing_mean   - mean firing vector used to center the data
%   b             - regression coefficients (from PCA-reduced space)
%   avg           - average hand trajectory for the direction
%   curr_bin      - current time step
%   reg_meth      - regression method specified (standard, poly, ridge,
%   lasso)
%   polyd         - polynomial regression order
% Output:
%   pos           - estimated x or y position
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(reg_meth, 'poly')
    firingVector = spikes_matrix(1:length(b));  
    
    % Expand features with polynomial terms but maintain 70 features
    polyFiringVector = zeros(size(firingVector)); % Initialize same size as firingVector
    for d = 1:polyd
        polyFiringVector = polyFiringVector + (firingVector - mean(firing_mean)).^d;
    end

    % Predict position using polynomial regression
    pos = polyFiringVector' * b + avg;
    else
    pos = (spikes_matrix(1:length(b)) - mean(firing_mean))' * b + avg;
    end

    try
        pos = pos(curr_bin, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end
end
