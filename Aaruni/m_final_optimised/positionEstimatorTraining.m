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
    reaching_angles = [1/6, 7/18, 11/18, 15/18, 19/18, 23/18, 31/18, 35/18] .* pi;
    bin_group = 20;
    alpha = 0.35; % arbitrary value decided through multiple trials
    sigma = 50;   % standard deviation in ms
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
    num_bins = time_bins / bin_group;          % e.g. 16:28 if 560 is max

    % Store in modelParameters
    modelParameters = struct;
    modelParameters.start_idx = start_idx;
    modelParameters.stop_idx  = stop_idx;

    %% Spikes Preprocessing: Binning (20ms), Sqrt Transformation, EMA Smoothing
    preprocessed_data = preprocessing(training_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
    orig_neurons = size(preprocessed_data(1,1).rate, 1);

    %% Remove data from neurons with low firing rates (once at max bins).
    [spikes_mat, ~] = extract_features(preprocessed_data, orig_neurons, stop_idx/bin_group, 'nodebug');
    removed_neurons = remove_neurons(spikes_mat, orig_neurons, 'nodebug');
    neurons = orig_neurons - length(removed_neurons);
    modelParameters.removeneurons = removed_neurons;
    clear spikes_mat  % free memory but doesn't affect shapes

    %% For each bin count, extract features, run PCA & LDA, store parameters
    for curr_bin = 1:length(num_bins)
        % Extract features for this many bins
        [spikes_matrix, labels] = extract_features(preprocessed_data, orig_neurons, num_bins(curr_bin), 'nodebug');
        
        % Remove low-firing neurons from the local spikes matrix
        spikes_matrix(removed_neurons, : ) = [];

        % PCA
        pca_threshold = 40; % e.g. 40 principal components
        [coeff, score, nPC] = perform_PCA(spikes_matrix, pca_threshold, 'cov', 'nodebug');

        % LDA
        lda_dim = 6;
        [outputs, weights] = perform_LDA(spikes_matrix, score, labels, lda_dim, training_length, 'nodebug');

        % kNN training info
        modelParameters.classify(curr_bin).dPCA_kNN = nPC;
        modelParameters.classify(curr_bin).dLDA_kNN = lda_dim;

        modelParameters.classify(curr_bin).wTrain = weights;  % [lda_dim x #samples]
        modelParameters.classify(curr_bin).wTest  = outputs;  % [#samples x lda_dim]

        modelParameters.classify(curr_bin).mean_firing = mean(spikes_matrix, 2);
        modelParameters.classify(curr_bin).labels_kNN  = labels(:)';  
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
        for timeWindowIndex = 1:((stop_idx - start_idx)/bin_group)+1
            [regressionCoefficientsX, regressionCoefficientsY, windowedFiring] = ...
                calcRegressionCoefficients(timeWindowIndex, time_division, labels, ...
                                           directionIndex, spikes_matrix, pca_threshold, ...
                                           Interval, currentXPositions, currentYPositions);
            
            % Store in modelParameters
            modelParameters.pcr(directionIndex,timeWindowIndex).bx    = regressionCoefficientsX;
            modelParameters.pcr(directionIndex,timeWindowIndex).by    = regressionCoefficientsY;
            modelParameters.pcr(directionIndex,timeWindowIndex).fMean = mean(windowedFiring,1);

            % Store average positions
            modelParameters.averages(timeWindowIndex).avX = squeeze(mean(xPos,1));
            modelParameters.averages(timeWindowIndex).avY = squeeze(mean(yPos,1));
        end
    end
    
    %% (Optional) RNN code omitted here, same as your original

end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS (with minor speedups and no shape/logic changes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function preprocessed_data = preprocessing(training_data, bin_group, filter_type, alpha, sigma, method)
    [rows, cols] = size(training_data); 
    preprocessed_data = struct;  % pre-allocate a struct array of size [rows x cols]

    % (Optional) If you have Parallel Toolbox, you can do:
    % parfor c = 1:cols
    for c = 1:cols
        for r = 1:rows
            train = training_data(r,c);
            [neurons, timepoints] = size(train.spikes);
            num_bins = floor(timepoints / bin_group);

            binned_spikes   = zeros(neurons, num_bins);
            binned_handPos  = zeros(size(train.handPos,1), num_bins);

            for b = 1:num_bins
                start_time = (b-1)*bin_group + 1;
                end_time   = b*bin_group;
                if b == num_bins 
                    binned_spikes(:, b)  = sum(train.spikes(:, start_time:end), 2);
                    binned_handPos(:, b) = mean(train.handPos(:, start_time:end), 2);
                else
                    binned_spikes(:, b)  = sum(train.spikes(:, start_time:end_time), 2);
                    binned_handPos(:, b) = mean(train.handPos(:, start_time:end_time), 2);
                end
            end
            binned_handPos_centred = bsxfun(@minus, binned_handPos, mean(binned_handPos, 2));
            preprocessed_data(r,c).handPos = binned_handPos_centred;
            
            % sqrt transform
            sqrt_spikes = sqrt(binned_spikes);

            if strcmp(filter_type, 'Gaussian')
                gKernel = gaussian_filter(bin_group, sigma);
                gaussian_spikes = zeros(size(sqrt_spikes));
                for n = 1:neurons
                    gaussian_spikes(n,:) = conv(sqrt_spikes(n,:), gKernel, 'same')/(bin_group/1000);
                end
                preprocessed_data(r,c).rate = gaussian_spikes; 
            elseif strcmp(filter_type, 'EMA')
                ema_spikes = ema_filter(sqrt_spikes, alpha, neurons);
                preprocessed_data(r,c).rate = ema_spikes / (bin_group/1000); 
            end            
        end
    end
    
    if strcmp(method, 'debug')
        % debug plotting code...
    end
end

function ema_spikes = ema_filter(sqrt_spikes, alpha, num_neurons)
    [~, nBins] = size(sqrt_spikes);
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:nBins
            ema_spikes(n, t) = alpha*sqrt_spikes(n, t) + (1-alpha)*ema_spikes(n, t-1);
        end
    end
end

function gKernel = gaussian_filter(bin_group, sigma)
    gaussian_window = 10*(sigma/bin_group);
    e_std = sigma/bin_group;
    alpha = (gaussian_window - 1)/(2*e_std);

    time_window = -(gaussian_window-1)/2 : (gaussian_window-1)/2;
    gKernel = exp((-1/2)* (alpha * time_window/((gaussian_window-1)/2)).^2)';
    gKernel = gKernel / sum(gKernel);
end

function [spikes_matrix, labels] = extract_features(preprocessed_data, neurons, curr_bin, debug)
    [rows, cols] = size(preprocessed_data);
    total_cols = rows*cols;
    % Pre-allocate: we know final size is [neurons*curr_bin, rows*cols]
    spikes_matrix = zeros(neurons*curr_bin, total_cols);
    labels        = zeros(total_cols, 1);
    
    for r = 1:rows
        for c = 1:cols
            c_idx = (c-1)*rows + r;
            for k = 1:curr_bin
                r_start = neurons*(k-1)+1;
                r_end   = neurons*k;
                spikes_matrix(r_start:r_end, c_idx) = preprocessed_data(r,c).rate(:,k);
            end
            labels(c_idx) = c; 
        end
    end

    if strcmp(debug, 'debug')
        figure; plot(spikes_matrix);
        title(['Firing Rate for Bin ', num2str(curr_bin)]);
    end
end

function removed_neurons = remove_neurons(spike_matrix, neurons, debug)
    removed_neurons = [];
    for neuronIdx = 1:neurons
        avgFiringRate = mean(mean(spike_matrix(neuronIdx:neurons:end, :)));
        if avgFiringRate < 0.5
            removed_neurons = [removed_neurons, neuronIdx]; 
        end
    end
    if strcmp(debug, 'debug')
        disp('Removed neurons = ');
        disp(removed_neurons);
    end
end

function [coeff, score, nPC] = perform_PCA(data, threshold, method, debug)
    if strcmp(method, 'cov')
        nPC = threshold; % fixed # PCs
        data_centred = data - mean(data,2);
        C = data_centred' * data_centred;
        [V, D] = eig(C);
        [d, idx] = sort(diag(D), 'descend');
        V = V(:, idx);
        score = data_centred * V * diag(1./sqrt(d));
        score = score(:, 1:nPC);
        coeff = V(:, 1:nPC);

    elseif strcmp(method, 'svd')
        variance_threshold = threshold; % fraction of variance to keep
        Xc = data - mean(data,2);
        [U, S, V] = svd(Xc, 'econ');
        singular_values = diag(S);
        explained_variance = (singular_values.^2) / sum(singular_values.^2);
        cum_variance = cumsum(explained_variance);
        nPC = find(cum_variance >= variance_threshold, 1);
        coeff = V(:, 1:nPC);
        score = Xc * coeff;
    end

    if strcmp(debug, 'debug')
        figure; plot(score);
        title('PCA Score');
    end
end

function [outputs, weights] = perform_LDA(data, score, labels, lda_dim, training_length, debug)
    classes = unique(labels);
    overall_mean = mean(data, 2);
    scatter_within  = zeros(size(data,1));
    scatter_between = zeros(size(data,1));

    for i = 1:length(classes)
        indicies = training_length*(i-1)+1 : training_length*i;
        mean_dir = mean(data(:, indicies), 2);

        dev_within = data(:, indicies) - mean_dir;
        scatter_within = scatter_within + dev_within * dev_within';

        dev_between = mean_dir - overall_mean;
        scatter_between = scatter_between + training_length*(dev_between*dev_between');
    end
    
    project_within  = score'* scatter_within  * score;
    project_between = score'* scatter_between * score;

    [V_lda, D_lda] = eig(pinv(project_within)* project_between);
    [~, sortIdx] = sort(diag(D_lda), 'descend');
    V_lda = V_lda(:, sortIdx(1:lda_dim));

    outputs = score* V_lda; % [samples x lda_dim]
    weights = outputs'*(data - overall_mean);

    if strcmp(debug, 'debug')
        figure; plot(outputs);  title('LDA Outputs');
        figure; plot(weights'); title('LDA Weights');
    end
end

function [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, bin_group, bins)
    handPos_cells = {training_data.handPos}; 
    max_trajectory= max(cellfun(@(hp) size(hp, 2), handPos_cells));
    [rows, cols] = size(training_data);

    xPos = zeros(rows, max_trajectory, cols);
    yPos = zeros(rows, max_trajectory, cols);

    for c = 1:cols
        for r = 1:rows
            curr_x = training_data(r,c).handPos(1,:);
            curr_y = training_data(r,c).handPos(2,:);
            len_xy = length(curr_x);
            if len_xy < max_trajectory
                pad_size = max_trajectory - len_xy;
                if pad_size > 0
                    xPos(r, :, c) = [curr_x, repmat(curr_x(end), 1, pad_size)];
                    yPos(r, :, c) = [curr_y, repmat(curr_y(end), 1, pad_size)];
                else
                    xPos(r, :, c) = curr_x;
                    yPos(r, :, c) = curr_y;
                end
            else
                xPos(r, :, c) = curr_x;
                yPos(r, :, c) = curr_y;
            end
        end
    end
    formatted_xPos = xPos(:, bins, :);
    formatted_yPos = yPos(:, bins, :);
end

function [regressionCoefficientsX, regressionCoefficientsY, FilteredFiring] = ...
    calcRegressionCoefficients(timeWindowIndex, timeDivision, labels, directionIndex, ...
                               neuraldata, pca_dimension, Interval, ...
                               currentXPositions, currentYPositions)
    % Center the positions for the current time window
    centeredX = bsxfun(@minus, currentXPositions(:, timeWindowIndex), ...
                       mean(currentXPositions(:, timeWindowIndex)));
    centeredY = bsxfun(@minus, currentYPositions(:, timeWindowIndex), ...
                       mean(currentYPositions(:, timeWindowIndex)));
    
    % Filter firing data based on time and direction
    FilteredFiring = filterFiringData(neuraldata, timeDivision, Interval(timeWindowIndex), ...
                                      labels, directionIndex);

    % Center the firing data
    centeredWindowFiring = FilteredFiring - mean(FilteredFiring,1);

    % PCA on the centered firing data
    [~, principalVectors, ~] = perform_PCA(centeredWindowFiring, pca_dimension, 'cov','nodebug');
    principalComponents = principalVectors' * centeredWindowFiring;

    % Compute regression matrix
    regressionMatrix = pinv(principalComponents * principalComponents') * principalComponents;

    regressionCoefficientsX = principalVectors * regressionMatrix * centeredX;
    regressionCoefficientsY = principalVectors * regressionMatrix * centeredY;
end

function FilteredFiring = filterFiringData(neuraldata, timeDivision, interval, labels, directionIndex)
    timeFilter      = (timeDivision <= interval);
    directionFilter = (labels == directionIndex);

    FilteredFiring  = neuraldata(timeFilter, :);
    % Center by subtracting mean over the selected trials
    FilteredFiring  = FilteredFiring(:, directionFilter) - ...
                      mean(FilteredFiring(:, directionFilter), 1);
end
