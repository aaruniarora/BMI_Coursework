function preprocessAndTrainPadPCA_Ridge()
    % This script:
    %   1) Loads monkeydata_training.mat
    %   2) Applies low-pass filtering (100 Hz) and Gaussian smoothing
    %   3) Pads each trial to the maximum trial length
    %   4) Computes the average starting hand position (first 300 ms)
    %   5) Applies PCA to the neural features
    %   6) Adds a bias column and uses ridge regression for decoding
    %   7) Computes RMSE on training data and visualizes actual vs. predicted trajectories
    
    % --- Preprocessing and data construction ---
    [X_all, Y_all, startHandPos, maxTimeBins, X_cell, Y_cell, mu, coeff, numComponents] = preprocessDataPadPCA();
    
    % --- Ridge Regression ---
    lambda = 10;  % Ridge penalty parameter; tune as needed
    % Use the PCA-transformed global design matrix (X_all)
    X_all_pca = X_all;  % (#samples x numComponents)
    
    % Add a bias column after PCA
    X_all_pca = [X_all_pca, ones(size(X_all_pca,1), 1)];  % now (#samples x (numComponents+1))
    
    % Solve for W in ridge regression: (X'X + λI)⁻¹X'Y
    [nSamples, nFeatures] = size(X_all_pca);
    Ireg = eye(nFeatures);
    Ireg(end,end) = 0;  % do not regularize the bias term
    W = (X_all_pca' * X_all_pca + lambda * Ireg) \ (X_all_pca' * Y_all);
    
    % --- Evaluate on training data ---
    Y_pred = X_all_pca * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    fprintf('Preprocessing done. Padded to %d bins, selected %d PCs.\n', maxTimeBins, numComponents);
    fprintf('Training RMSE (with PCA + bias + ridge): %.3f\n', rmse_val);
    
    % --- Visualize all trials ---
    figure('Name','All Trials: Actual vs Predicted');
    hold on;
    for i = 1:length(X_cell)
        % Subtract the mean using bsxfun to ensure proper dimensions
        X_trial_centered = bsxfun(@minus, X_cell{i}, mu);  % (#timeBins x originalNeuronCount)
        % Project onto the top principal components
        X_trial_pca = X_trial_centered * coeff(:,1:numComponents);
        % Add bias column
        X_trial_pca = [X_trial_pca, ones(size(X_trial_pca,1),1)];
        
        % Predict hand positions for the current trial
        Y_pred_trial = X_trial_pca * W;
        
        % Plot actual (blue) vs predicted (red dashed) trajectories
        plot(Y_cell{i}(:,1), Y_cell{i}(:,2), 'b', 'LineWidth', 1);
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--', 'LineWidth', 1);
    end
    xlabel('X position (mm)'); ylabel('Y position (mm)');
    title('All Trials: Actual (blue) vs Predicted (red dashed)');
    axis equal; grid on;
    hold off;
end

function [X_all_pca, Y_all, startHandPos, maxTimeBins, X_cell, Y_cell, mu, coeff, numComponents] = preprocessDataPadPCA()
    % This function:
    %   - Loads monkeydata_training.mat
    %   - Low-pass filters (0-100 Hz) each neuron's spike train
    %   - Gaussian-smooths the filtered spike train
    %   - Pads each trial's data with zeros to the maximum trial length
    %   - Computes the average starting hand position (first 300 ms)
    %   - Builds global design (X_all_raw) and target (Y_all) matrices
    %   - Applies PCA to X_all_raw, returning X_all_pca along with mu and coeff
    %
    % Outputs:
    %   X_all_pca     - (#samples x numComponents) after PCA
    %   Y_all         - (#samples x 2)
    %   startHandPos  - cell array with each trial's average start position
    %   maxTimeBins   - maximum # of time bins across all trials
    %   X_cell        - cell array of design matrices (one per trial)
    %   Y_cell        - cell array of target matrices (one per trial)
    %   mu            - mean of X_all_raw (1 x numNeurons)
    %   coeff         - PCA loadings
    %   numComponents - # of components chosen to capture 95% variance
    
    load('monkeydata_training.mat');  % Loads variable "trial"
    
    fs = 1000;  % Sampling frequency (1 ms bins)
    numTrials = size(trial,1); 
    numAngles = size(trial,2);
    numNeurons = size(trial(1,1).spikes,1);
    
    % --- 1) Find max trial length for padding ---
    maxTimeBins = 0;
    for a = 1:numAngles
        for t = 1:numTrials
            T = size(trial(t,a).spikes,2);
            maxTimeBins = max(maxTimeBins, T);
        end
    end
    
    % --- 2) Low-pass filter design (cutoff 100 Hz) ---
    [b,a] = butter(2, 100/(fs/2), 'low');
    
    % --- 3) Gaussian smoothing kernel ---
    gaussWindow = 11;  
    gaussKernel = gausswin(gaussWindow);
    gaussKernel = gaussKernel / sum(gaussKernel);
    
    % Cell arrays to hold each trial's data
    X_cell = {};
    Y_cell = {};
    startHandPos = {};
    
    for a = 1:numAngles
        for t = 1:numTrials
            spikes = double(trial(t,a).spikes);   % (numNeurons x T)
            handPos = trial(t,a).handPos;           % (3 x T)
            T = size(spikes,2);
            
            % Process each neuron's spike train: filter and smooth
            spikes_processed = zeros(size(spikes));
            for nrn = 1:numNeurons
                spk = spikes(nrn,:);
                spk_filt = filtfilt(b, a, spk);
                spk_smooth = conv(spk_filt, gaussKernel, 'same');
                spikes_processed(nrn,:) = spk_smooth;
            end
            
            % Pad the trial's data with zeros to reach maxTimeBins
            if T < maxTimeBins
                padAmt = maxTimeBins - T;
                spikes_padded = [spikes_processed, zeros(numNeurons, padAmt)];
                handPos_padded = [handPos, zeros(3, padAmt)];
            else
                spikes_padded = spikes_processed;
                handPos_padded = handPos;
            end
            
            % Build the trial design matrix (each row is a time bin)
            X_trial_raw = spikes_padded';  % (maxTimeBins x numNeurons)
            % Build the trial target matrix using only x and y components
            Y_trial = [handPos_padded(1,:)', handPos_padded(2,:)'];  % (maxTimeBins x 2)
            
            % Compute starting hand position: average over first 300 ms
            nAvg = min(300, size(handPos_padded,2));
            startPos = mean(handPos_padded(1:2,1:nAvg),2);
            
            X_cell{end+1} = X_trial_raw;
            Y_cell{end+1} = Y_trial;
            startHandPos{end+1} = startPos;
        end
    end
    
    % --- 4) Concatenate all trials ---
    X_all_raw = vertcat(X_cell{:});  % (#samples x numNeurons)
    Y_all = vertcat(Y_cell{:});      % (#samples x 2)
    
    % --- 5) PCA on X_all_raw ---
    [coeff, score, ~, ~, explained, mu] = pca(X_all_raw);
    cumExplained = cumsum(explained);
    numComponents = find(cumExplained >= 95, 1, 'first');  % Components capturing 95% variance
    X_all_pca = score(:,1:numComponents);  % (#samples x numComponents)
    
    % For visualization: we can optionally reconstruct per-trial PCA projections
    % (Here we just slice X_all_pca according to the lengths stored in X_cell)
    % We keep X_cell as the raw trial matrices for later projection.
    
    fprintf('Preprocessing done. Padded to %d bins, selected %d PCs.\n', maxTimeBins, numComponents);
end
