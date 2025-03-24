function preprocessAndTrainPadPCA_Visualize()
    % This function loads the training data, applies the following preprocessing:
    %   - Bandpass filtering (0.5–100 Hz)
    %   - Gaussian smoothing
    %   - Computes the average starting hand position (first 300 ms)
    %   - Pads each trial to the maximum trial length
    %   - Applies PCA to reduce dimensionality
    %
    % Then, it trains a linear regression model on the PCA-transformed data,
    % computes RMSE on the training data, and visualizes actual vs. predicted trajectories.
    
    % Preprocess data with padding
    [X_all, Y_all, startHandPos, maxTimeBins, X_cell, Y_cell] = preprocessDataPad();
    
    % Apply PCA to the global design matrix X_all.
    % X_all is (#samples x numNeurons). Perform PCA on X_all.
    [coeff, score, ~, ~, explained, mu] = pca(X_all);
    % Choose number of components to capture at least 95% of variance.
    cumExplained = cumsum(explained);
    numComponents = find(cumExplained >= 95, 1, 'first');
    fprintf('Using %d principal components to capture 95%% variance.\n', numComponents);
    
    % Transform the global design matrix to PCA space.
    X_all_pca = score(:, 1:numComponents);
    
    % Train a linear regression model in PCA space.
    % We solve: X_all_pca * W ≈ Y_all.
    W = (X_all_pca' * X_all_pca) \ (X_all_pca' * Y_all);
    
    % Predict on the global training data and compute RMSE.
    Y_pred = X_all_pca * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    fprintf('RMSE on training data (with PCA): %.3f\n', rmse_val);
    
    % Visualize actual vs. predicted trajectories for each trial.
    figure;
    hold on;
    for i = 1:length(X_cell)
        % Transform each trial's design matrix using the same PCA transformation.
        X_trial = X_cell{i};  % (maxTimeBins x numNeurons)
        X_trial_pca = (X_trial - mu) * coeff(:, 1:numComponents);
        % Predict hand positions for the current trial.
        Y_pred_trial = X_trial_pca * W;
        % Plot actual trajectory (blue) and predicted trajectory (red dashed).
        plot(Y_cell{i}(:,1), Y_cell{i}(:,2), 'b', 'LineWidth', 1);
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--', 'LineWidth', 1);
    end
    xlabel('X position (mm)');
    ylabel('Y position (mm)');
    title('Actual (blue) vs. Predicted (red dashed) Hand Trajectories');
    axis equal;
    grid on;
    hold off;
end

function [X_all, Y_all, startHandPos, maxTimeBins, X_cell, Y_cell] = preprocessDataPad()
    % This function loads monkeydata_training.mat and applies:
    %   1. Bandpass filtering (0.5–100 Hz) to each neuron's spike train.
    %   2. Gaussian smoothing to the filtered spike train.
    %   3. Computes the average starting hand position (first 300 ms).
    %   4. Pads each trial's data with zeros to the maximum trial length.
    %
    % Outputs:
    %   X_all       - Global design matrix (each row is a time bin sample across trials)
    %   Y_all       - Global target matrix ([x, y] positions per time bin)
    %   startHandPos- Cell array of starting hand positions (average over first 300 ms) for each trial.
    %   maxTimeBins - Maximum number of time bins across all trials.
    %   X_cell      - Cell array of design matrices (one per trial)
    %   Y_cell      - Cell array of target matrices (one per trial)
    
    load('monkeydata_training.mat');  % Loads variable "trial"
    
    % Determine data dimensions.
    numTrials = size(trial, 1);   % e.g., 100
    numAngles  = size(trial, 2);   % e.g., 8
    numNeurons = size(trial(1,1).spikes, 1);  % e.g., 98
    fs = 1000;  % Sampling frequency (1 ms bins)
    
    % Find the maximum trial length (in time bins) across all trials.
    maxTimeBins = 0;
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            T = size(trial(tr, angleIdx).spikes, 2);
            maxTimeBins = max(maxTimeBins, T);
        end
    end
    
    % Design a 2nd-order Butterworth bandpass filter (0.5-100 Hz).
    [b, a] = butter(2, [0.5, 100]/(fs/2), 'bandpass');
    
    % Set parameters for the Gaussian filter.
    gaussWindow = 11;   % window length (must be odd)
    gaussSigma = 2;     % standard deviation (in samples)
    gaussKernel = gausswin(gaussWindow);
    gaussKernel = gaussKernel / sum(gaussKernel);  % normalize kernel
    
    % Initialize cell arrays to store data from each trial.
    X_cell = {};
    Y_cell = {};
    startHandPos = {};
    
    % Loop over all trials (across angles and trial numbers).
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            % Extract spike and hand position data for the current trial.
            spikes = trial(tr, angleIdx).spikes;    % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos;      % (3 x T)
            T = size(spikes, 2);
            
            % Preallocate the processed spikes matrix.
            spikes_processed = zeros(size(spikes));
            for neuron = 1:numNeurons
                spk = double(spikes(neuron, :));
                % Apply bandpass filter.
                spk_filt = filtfilt(b, a, spk);
                % Apply Gaussian smoothing.
                spk_smooth = conv(spk_filt, gaussKernel, 'same');
                spikes_processed(neuron, :) = spk_smooth;
            end
            
            % Pad the trial's data with zeros to reach maxTimeBins.
            if T < maxTimeBins
                padAmount = maxTimeBins - T;
                spikes_padded = [spikes_processed, zeros(numNeurons, padAmount)];
                handPos_padded = [handPos, zeros(3, padAmount)];
            else
                spikes_padded = spikes_processed;
                handPos_padded = handPos;
            end
            
            % Build the design matrix for this trial (each row is a time bin).
            X_trial = spikes_padded';  % (maxTimeBins x numNeurons)
            % Build the target matrix using only x and y components.
            Y_trial = [handPos_padded(1,:)', handPos_padded(2,:)'];  % (maxTimeBins x 2)
            
            % Compute the starting hand position as the average over the first 300 ms.
            numSamplesForAvg = min(300, size(handPos_padded, 2));
            startPos = mean(handPos_padded(1:2, 1:numSamplesForAvg), 2);  % [x; y]
            
            % Save the processed data for the trial.
            X_cell{end+1} = X_trial;
            Y_cell{end+1} = Y_trial;
            startHandPos{end+1} = startPos;
        end
    end
    
    % Concatenate data from all trials to form global design and target matrices.
    X_all = vertcat(X_cell{:});
    Y_all = vertcat(Y_cell{:});
    
    fprintf('Preprocessing complete. Each trial padded to %d time bins.\n', maxTimeBins);
end
