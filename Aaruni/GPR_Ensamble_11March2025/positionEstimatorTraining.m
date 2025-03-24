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
    % assignin('base', 'preprocessed_data', preprocessed_data); 
    orig_neurons = size(preprocessed_data(1,1).rate, 1);

    %% Remove data from neurons with low firing rates.
    [spikes_mat, ~] = extract_features(preprocessed_data, orig_neurons, stop_idx/bin_group, 'nodebug');
    removed_neurons = remove_neurons(spikes_mat, orig_neurons, 'nodebug');
    neurons = orig_neurons - length(removed_neurons);
    modelParameters.removeneurons = removed_neurons;
    clear spikes_mat

    %% Dimensionality parameters
    pca_threshold = 40; % =40 for cov and =0.95 for svd
    lda_dim = 6;
    % % t-SNE
    % tsne_dim     = 6;         % or whatever dimensionality you want
    % perplexity   = 30;        % typical t-SNE perplexity
    % max_iter     = 1000;      % number of gradient-descent iterations

    for curr_bin = 1: length(num_bins)
        %% Extract features/restructure data for further analysis
        [spikes_matrix, labels] = extract_features(preprocessed_data, orig_neurons, num_bins(curr_bin), 'nodebug');
        
        %% Remove data from neurons with low firing rates.
        spikes_matrix(removed_neurons, : ) = [];
        % disp(['At bin ' num2str(curr_bin) ' spikes ' num2str(size(spikes_matrix))]);

        %% PCA for dimensionality reduction of the neural data
        [coeff, score, nPC] = perform_PCA(spikes_matrix, pca_threshold, 'cov', 'nodebug');
        % tsne_Y = perform_tSNE(spikes_matrix, tsne_dim, perplexity, max_iter, false);
        % disp(['score ' num2str(size(tsne_Y))]);

        %% LDA to maximise class separability across different directions
        [outputs, weights] = perform_LDA(spikes_matrix, score, labels, lda_dim, training_length, 'nodebug');
        % [outputs, weights] = perform_LDA(spikes_matrix, tsne_Y, labels, lda_dim, training_length, 'nodebug');

        %% kNN training: store samples in LDA space with corresponding hand positions
        % modelParameters.classify(curr_bin).dPCA_kNN = nPC;
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
        nPC = threshold;
        data_centred = data - mean(data,2);
        % C = cov(data_centred);
        C = data_centred' * data_centred;
        [V, D] = eig(C);
        [d, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        score = data_centred * V * diag(1./sqrt(d));
        score = score(:, 1:nPC);
        coeff = V(:, 1:nPC);

        % Normalize each principal component (each column) to have unit norm
        % normFactors = sqrt(sum(score.^2, 1));  % 1 x nPC vector, norm of each column
        % score = score ./ repmat(normFactors, size(score, 1), 1);
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

%% t-SNE
function Y = perform_tSNE(X, no_dims, perplexity, max_iter, debugFlag)
% perform_tSNE - A barebones implementation of t-SNE in pure MATLAB
%
%   Y = perform_tSNE(X, no_dims, perplexity, max_iter, debugFlag)
%
% Inputs:
%   X          : [D x N] data matrix, D = # features, N = # samples
%   no_dims    : desired dimensionality of the output (e.g. 2, 3, 6, etc.)
%   perplexity : t-SNE perplexity (controls local/global trade-off)
%   max_iter   : number of gradient descent iterations
%   debugFlag  : true/false for printing intermediate info
%
% Output:
%   Y          : [no_dims x N] the dimension-reduced data

    % 1) Transpose data if needed
    % We want a matrix with samples in rows. If X is [D x N], we can keep it that way
    % but be consistent about how we measure distances. We'll convert X to [N x D].
    % if size(X, 1) > size(X, 2)
    %     % In your pipeline, you often have rows=features, columns=samples. That’s fine,
    %     % but the typical t-SNE approach expects NxD. We'll just transpose to be safe.
    %     X = X';
    % end
    
    % Now X is [N x D]. N = # samples, D = # features
    [N, D] = size(X);

    % 2) Normalize or scale data (optional but often helps)
    X = X - mean(X, 1);
    X = X ./ std(X, [], 1);

    % 3) Compute pairwise Euclidean distances
    %  We get a NxN distance matrix
    sum_X  = sum(X.^2, 2);
    dist   = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2*(X*X')));
    dist(dist < 0) = 0;    % numerical guard

    % 4) Find the kernel bandwidth via binary search, so that each row has the given perplexity
    %     We'll store the final conditional probabilities in P, an NxN matrix
    P = zeros(N, N);
    logPerplexity = log(perplexity);

    for i = 1:N
        beta = 1;        % 1 / (2sigma^2)
        betamin = -Inf;
        betamax = Inf;
        
        % Exclude diagonal distances
        Di = dist(i, [1:i-1,i+1:end]);

        % Evaluate whether perplexity is correct
        [H, thisP] = computePerplexity(Di, beta);
        Hdiff = H - logPerplexity;
        tries = 0;

        % If Hdiff is too large, we want a larger sigma -> smaller beta
        while abs(Hdiff) > 1e-5 && tries < 50
            if Hdiff > 0
                betamin = beta; 
                if isinf(betamax)
                    beta = beta * 2;
                else
                    beta = (beta + betamax) / 2;
                end
            else
                betamax = beta;
                if isinf(betamin)
                    beta = beta / 2;
                else
                    beta = (beta + betamin) / 2;
                end
            end
            [H, thisP] = computePerplexity(Di, beta);
            Hdiff = H - logPerplexity;
            tries = tries + 1;
        end
        
        % Store row i of P
        P(i, [1:i-1,i+1:end]) = thisP;
    end

    % Symmetrize P, then normalize so it sums to 1
    P = 0.5 * (P + P');
    P = max(P, realmin);
    P = P / sum(P(:));
    
    % Early exaggeration
    P = P * 4;

    % 5) Initialize the solution Y
    Y = 0.0001 * randn(N, no_dims);

    % 6) Gradient descent
    eta           = 500;      % learning rate
    momentum      = 0.8;      % initial momentum
    min_gain      = 0.01;
    gains         = ones(size(Y));
    dY            = zeros(size(Y));
    old_dY        = zeros(size(Y));
    final_momentum = 0.8;
    momentum_switch_iter = 250;   % iteration to increase momentum
    % typical t-SNE: 1000-2000 iterations

    for iter = 1:max_iter
        
        % Compute pairwise affinities in low-dimensional map
        sum_Y  = sum(Y.^2, 2);
        num    = -2 * (Y * Y');
        num    = bsxfun(@plus, sum_Y, bsxfun(@plus, sum_Y', num));
        num    = 1 ./ (1 + num);
        num(1:N+1:end) = 0; % zero out diagonal
        Q = num / sum(num(:));
        Q = max(Q, realmin);

        % Compute gradient of the Kullback-Leibler divergence
        PQ       = P - Q;          % NxN
        for i = 1:N
            % The gradient on point i w.r.t. all other points
            dY(i,:) = sum( bsxfun(@times, (PQ(:,i) .* num(:,i)), (Y(i,:) - Y)), 1 );
        end

        % Apply momentum & update
        gains = (sign(dY) ~= sign(old_dY)) .* (gains + 0.2) + ...
                (sign(dY) == sign(old_dY)) .* (gains * 0.8);

        gains(gains < min_gain) = min_gain;

        dY   = momentum * old_dY - eta * (gains .* dY);
        Y    = Y + dY;
        old_dY = dY;

        % Decrease early exaggeration
        if iter == 100
            P = P / 4;
        end

        % Switch momentum
        if iter == momentum_switch_iter
            momentum = final_momentum;
        end

        % (Optional) Print some debugging info
        if debugFlag && (mod(iter, 50) == 0)
            cost = sum(P(:) .* log((P(:) + realmin)./(Q(:) + realmin)));
            fprintf('Iteration %d: error is %f\n', iter, cost);
        end
    end

    % Return as [no_dims x N] if you want to mirror the PCA’s shape
    % Y = Y';
end

function [H, thisP] = computePerplexity(Di, beta)
    % Di is a row of squared distances
    % beta = 1 / (2 sigma^2)
    P = exp(-Di * beta);
    sumP = sum(P);
    H = log(sumP) + beta * sum(Di .* P) / sumP;   % Shannon entropy
    thisP = P / sumP;
end


%%

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

    % Calculate regression coefficients for X and Y using the regression matrix
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

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rnnModel = trainElmanRNN(X, Y, hiddenSize, learningRate, nEpochs)
% trainElmanRNN - Train a simple Elman RNN (one hidden layer) via BPTT.
%
%   rnnModel = trainElmanRNN(X, Y, hiddenSize, learningRate, nEpochs)
%
% Inputs:
%   X            [DxT] matrix of inputs,  D = dimension of input, T = # of timesteps
%   Y            [OxT] matrix of targets, O = dimension of output, T = # of timesteps
%   hiddenSize   scalar, # of hidden units H
%   learningRate scalar, step size for gradient descent
%   nEpochs      scalar, number of epochs (full passes) over the sequence
%
% Output:
%   rnnModel     struct containing learned weights/biases:
%                W_xh, W_hh, W_ho, b_h, b_o
%                as well as final training loss trajectory
%
% Reference: 
%   - Elman, J. L. (1990). "Finding structure in time." Cognitive Science, 14(2), 179–211.
%   - Goodfellow, Bengio, Courville (2016). "Deep Learning." MIT Press.

  % ------------------------
  % 1. Dimensions & Init
  % ------------------------
  [inputSize, T]  = size(X); 
  outputSize       = size(Y,1);
  H               = hiddenSize;  % # hidden units

  % Random initialization (small random values)
  rng('default');
  W_xh = 0.01*randn(H, inputSize);  % input-to-hidden
  W_hh = 0.01*randn(H, H);          % hidden-to-hidden
  W_ho = 0.01*randn(outputSize, H); % hidden-to-output

  b_h  = zeros(H,1);
  b_o  = zeros(outputSize,1);

  % Store for debugging
  lossHistory = zeros(nEpochs,1);

  % ------------------------
  % 2. Training Loop
  % ------------------------
  for epoch = 1:nEpochs

      % Initialize total gradients to zero
      dW_xh = zeros(size(W_xh));
      dW_hh = zeros(size(W_hh));
      dW_ho = zeros(size(W_ho));
      db_h  = zeros(size(b_h));
      db_o  = zeros(size(b_o));

      % We will store hidden states for backprop
      h = zeros(H, T);   % hidden states for each t
      hprev = zeros(H,1);

      % ===== Forward pass: compute h_t and yhat_t for t=1..T =====
      yhat = zeros(outputSize, T);
      for t = 1:T
          x_t     = X(:,t);
          % hidden update
          h(:,t)  = tanh( W_xh*x_t + W_hh*hprev + b_h );
          % output
          yhat(:,t) = W_ho * h(:,t) + b_o;
          % update hprev for next time
          hprev = h(:,t);
      end

      % Compute cost (MSE)
      cost = 0.5 * mean(sum((Y - yhat).^2,1)); 
      lossHistory(epoch) = cost;

      % ===== Backward pass: BPTT to accumulate gradients =====
      % We'll propagate gradients backward in time from t=T..1
      dh_next = zeros(H,1); % gradient wrt hidden state at next time
      for t = T:-1:1
          % dL/dyhat
          dy = (yhat(:,t) - Y(:,t)); % O x 1
          % partial wrt W_ho, b_o
          dW_ho = dW_ho + dy * (h(:,t))';
          db_o  = db_o  + dy;

          % gradient wrt h(t)
          dh = (W_ho' * dy) + dh_next;  % add the gradient coming from next time step
          
          % derivative of tanh: dtanh(z) = 1 - tanh^2(z)
          z_t = h(:,t);  % same as tanh(...) above
          dtanh_ = (1 - z_t.^2) .* dh;  % element-wise

          % partial wrt b_h
          db_h  = db_h  + dtanh_;

          % partial wrt W_xh, W_hh
          x_t = X(:,t);
          dW_xh = dW_xh + dtanh_ * x_t';
          
          % We need h(t-1). If t>1, h(t-1) is h(:,t-1), else 0
          if t>1
              h_tm1 = h(:,t-1);
          else
              h_tm1 = zeros(H,1);
          end
          dW_hh = dW_hh + dtanh_ * h_tm1';

          % gradient wrt h(t-1) to pass backward
          dh_next = W_hh' * dtanh_;
      end

      % ====================
      % 3. Gradient Update
      % ====================
      % (Simple gradient descent)
      W_xh = W_xh - learningRate * dW_xh / T;
      W_hh = W_hh - learningRate * dW_hh / T;
      W_ho = W_ho - learningRate * dW_ho / T;
      b_h  = b_h  - learningRate * db_h  / T;
      b_o  = b_o  - learningRate * db_o  / T;
      
      % (Optional) Display or store the cost for monitoring
      if mod(epoch, 10) == 0
          fprintf('Epoch %d/%d, MSE loss = %.4f\n', epoch, nEpochs, cost);
      end
  end

  % ------------------------
  % 4. Store Results
  % ------------------------
  rnnModel.W_xh = W_xh;
  rnnModel.W_hh = W_hh;
  rnnModel.W_ho = W_ho;
  rnnModel.b_h  = b_h;
  rnnModel.b_o  = b_o;
  rnnModel.loss = lossHistory;
end



% function weights = perform_RNN(binnedTrials, mu, coeff, Wlda, binSize)
%     % Train a simple linear RNN model using least squares.
%     % For each trial, for time steps t = 2:end, create input:
%     %   [LDA_features (current bin), previous hand position, 1]
%     % and target:
%     %   delta = current hand position - previous hand position.
%     nLDA = size(Wlda,2);
%     X_total = [];
%     Y_total = [];
% 
%     [numRows, numCols] = size(binnedTrials);
%     for i = 1:numRows
%         for j = 1:numCols
%             trial = binnedTrials(i,j);
%             numBins = size(trial.spikes,2);
%             if numBins < 2, continue; end
%             % Compute LDA features for each bin using PCA and LDA parameters
%             features_lda = zeros(numBins, nLDA);
%             for b = 1:numBins
%                 spike_bin = trial.spikes(:, b)'; % 1 x numNeurons
%                 spike_bin_centered = spike_bin - mu;
%                 pca_feat = spike_bin_centered * coeff;  % 1 x nPC
%                 features_lda(b,:) = pca_feat * Wlda;      % 1 x nLDA
%             end
%             % For time steps t=2...numBins, get input and target delta
%             for b = 2:numBins
%                 prevHand = trial.handPos(1:2, b-1)';  % previous (x,y)
%                 currentHand = trial.handPos(1:2, b)';   % current (x,y)
%                 delta = currentHand - prevHand;          % change in hand pos
%                 current_feat = features_lda(b, :);        % current LDA features
%                 X_sample = [current_feat, prevHand, 1];     % add bias term
%                 X_total = [X_total; X_sample];
%                 Y_total = [Y_total; delta];
%             end
%         end
%     end
%     % Solve for weights: (nLDA + 3) x 2 matrix
%     weights = pinv(X_total) * Y_total;
% end