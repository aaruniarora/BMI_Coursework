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
%   7. Trains a regression model (Kalman) to map spikes to (x,y)
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
    bin_group = 20; % hypertuned 
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
        [~, score, nPC] = perform_PCA(spikes_matrix, pca_threshold, 'nodebug');

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
    % [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, num_bins*bin_group);
    for c = 1:directions
        for r = 1:training_length
            % Fill missing values in hand position
            training_data(r,c).handPos(1,:) = fill_nan(training_data(r,c).handPos(1,:), 'handpos');
            training_data(r,c).handPos(2,:) = fill_nan(training_data(r,c).handPos(2,:), 'handpos');
        end
    end

    %% Kalman
    % selected_neurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98];
    selected_neurons = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 48, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 71, 72, 73, 74, 75, 77, 78, 80, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 94, 96, 97, 98];
    % selected_neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98];
    modelParameters.selected_neurons = selected_neurons;

    for dir_idx = 1:directions

        for tr = 1:training_length
            % positionEstimatorTraining_one_trial(training_data, selected_neurons, lag, num_bins, start_idx)
            [A,H,Q,W] = positionEstimatorTraining_one_trial(training_data(tr,dir_idx), ...
                selected_neurons, bin_group, start_idx);
            Parameters.A{tr}=A;
            Parameters.H{tr}=H;
            Parameters.Q{tr}=Q;
            Parameters.W{tr}=W;

        end
    
        % average over trials to obtain a final set of parameters 
        % (A_(dir), H_(dir), Q_(dir), W_(dir)) for each direction
        modelParameters.A{dir_idx}=sum(cat(3,Parameters.A{:}),3)./training_length; 
        modelParameters.H{dir_idx}=sum(cat(3,Parameters.H{:}),3)./training_length; 
        modelParameters.W{dir_idx}=sum(cat(3,Parameters.W{:}),3)./training_length; 
        modelParameters.Q{dir_idx}=sum(cat(3,Parameters.Q{:}),3)./training_length; 

    end
end

%% HELPER FUNCTIONS 

function [A,H,Q,W] = positionEstimatorTraining_one_trial(training_data, selected_neurons, ...
    lag, start_idx)
% Function for 'one trial' parameters estimation (A_(tr, dir), H_(tr, dir), 
% Q_(tr, dir), W_(tr, dir)) for a given angle

    % CONSTANTS
    nb_states = 4; % X Y Vx Vy 
    % NB: testing phase revealed that the inclusion of acceleration components in the state vector did not improved the performance of the decoder. 
    % Therefore, acceleration was excluded (only 4 states are used).
    time_max = length(training_data.handPos) - start_idx; 

    % Build observation matrix z
    num_bins = floor(time_max/lag); 
    for nr = 1:length(selected_neurons)
        neuron = selected_neurons(nr);
        spike = training_data.spikes(neuron,  start_idx+1:(start_idx+num_bins*lag));
        spike = reshape(spike, lag, num_bins);
        spike_count = sum(spike, 1);
        z(nr, :) = spike_count / lag;
    end


    % Build state matrix x over time bins
    x = zeros(nb_states,num_bins); % State Matrix 

    % Compute position every t=320+k*20ms 
    % x(1,:)=[X(320+20ms), X(320+40ms), ...., X(320+nb_bins*20ms)]
    % x(2,:)= [Y(320+20ms), Y(320+40ms), ...., Y(320+nb_bins*20ms)]

    for k = 1:num_bins
        x(1,k) = training_data.handPos(1,start_idx+k*lag); % X
        x(2,k) = training_data.handPos(2,start_idx+k*lag); % Y
    end

    Pos_0 = training_data.handPos(1:2,start_idx); %(x0, y0) store initial position at t=320 ms

    % Compute velocity every t=320+k*20ms 
    % Vx(1,:)=(1/20)*[X(320+20ms)-X(320ms), X(320+40ms)-X(320+20ms), ...., X(320+nb_bins*20ms)-X(320+(nb_bins-1)*20ms)]
    % Vy(2,:)= (1/20)*[Y(320+20ms)-Y(320ms), Y(320+40ms)-Y(320+20ms), ...., Y(320+nb_bins*20ms)-Y(320+(nb_bins-1)*20ms)]
    for k = 1:num_bins
        if k==1
            x(3,k) = (x(1,k)-Pos_0(1))/lag;
            x(4,k) = (x(2,k)-Pos_0(2))/lag;
        else
            x(3,k) = (x(1,k)-x(1,k-1))/lag;
            x(4,k) = (x(2,k)-x(2,k-1))/lag;
        end
    end

    % Parameters Estimations 
    % Reference: W. Wu, M. Black, Y. Gao, E. Bienenstock, M. Serruya, and J. Donoghue, "Inferring hand motion from multi-cell recordings in motor cortex 
    % using a kalman filter," (2002)

    % Useful Matrices  
    X1 = x(:,1:(end-1));
    X2 = x(:,2:end);

    % Compute A, H, W and Q
    A = X2*X1' / (X1*X1');
    W = X2*X2' - A*X1*X2';
    W = W / (num_bins-1);
    H = z*x' / (x*x');
    Q = (z*z' - H*x*z');
    Q = Q / (num_bins);

end

%% Preprocessing
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

%% Feature extraction
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

%% Dim reduction
function [coeff, score, nPC] = perform_PCA(data, threshold, debug)
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
        figure; plot(score);
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