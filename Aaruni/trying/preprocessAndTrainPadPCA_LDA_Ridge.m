function preprocessAndTrainPadPCA_LDA_Ridge()
    % This function implements a complete pipeline:
    % 1) Low-pass filtering (cutoff 100 Hz) and Gaussian smoothing.
    % 2) Padding each trial and computing average starting hand position.
    % 3) PCA on the neural features.
    % 4) LDA on the PCA features using reaching angle labels.
    % 5) Ridge regression (with a bias term) in the LDA space.
    % 6) Computes RMSE on training data and visualizes actual vs. predicted trajectories.
    
    % --- Preprocessing: Build design matrix, targets, and labels ---
    [X_all_pca, Y_all, startHandPos, maxTimeBins, X_cell, Y_cell, mu, coeff, numComponents, labels_all] = preprocessDataPadPCA_LDA();
    
    % --- Apply LDA to PCA features ---
    % Compute transformation matrix L (from PCA space to LDA space)
    L = computeLDA(X_all_pca, labels_all);
    % Transform global PCA features to LDA space
    X_all_lda = X_all_pca * L;
    
    % --- Add bias (intercept) column ---
    X_all_lda_bias = [X_all_lda, ones(size(X_all_lda,1), 1)];  % (#samples x (nLDA+1))
    
    % --- Ridge Regression ---
    lambda = 10;  % Ridge penalty parameter (tune as needed)
    [nSamples, nFeatures] = size(X_all_lda_bias);
    Ireg = eye(nFeatures);
    Ireg(end, end) = 0;  % Do not regularize the bias term
    W = (X_all_lda_bias' * X_all_lda_bias + lambda * Ireg) \ (X_all_lda_bias' * Y_all);
    
    % --- Evaluate on training data ---
    Y_pred = X_all_lda_bias * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    fprintf('Preprocessing done. Padded to %d bins, selected %d PCs, LDA reduced to %d dimensions.\n', maxTimeBins, numComponents, size(L,2));
    fprintf('Training RMSE (with PCA + LDA + bias + ridge): %.3f\n', rmse_val);
    
    % --- Visualization: For each trial, predict and plot trajectories ---
    figure('Name','All Trials: Actual vs Predicted');
    hold on;
    for i = 1:length(X_cell)
        % Center each trial's raw features using global mean mu.
        X_trial_centered = bsxfun(@minus, X_cell{i}, mu);  % (maxTimeBins x originalNeurons)
        % Project onto the same PCA space.
        X_trial_pca = X_trial_centered * coeff(:, 1:numComponents);
        % Project further onto the LDA space.
        X_trial_lda = X_trial_pca * L;
        % Add bias column.
        X_trial_lda_bias = [X_trial_lda, ones(size(X_trial_lda,1), 1)];
        % Predict hand positions.
        Y_pred_trial = X_trial_lda_bias * W;
        % Plot actual trajectory in blue and predicted in red dashed.
        plot(Y_cell{i}(:,1), Y_cell{i}(:,2), 'b', 'LineWidth', 1);
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--', 'LineWidth', 1);
    end
    xlabel('X position (mm)'); ylabel('Y position (mm)');
    title('Actual (blue) vs Predicted (red dashed) Hand Trajectories');
    axis equal; grid on;
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_all_pca, Y_all, startHandPos, maxTimeBins, X_cell, Y_cell, mu, coeff, numComponents, labels_all] = preprocessDataPadPCA_LDA()
    % This function loads monkeydata_training.mat and applies:
    % 1) Low-pass filtering (cutoff 100 Hz) to each neuron's spike train.
    % 2) Gaussian smoothing.
    % 3) Padding each trial with zeros to the maximum trial length.
    % 4) Computes average starting hand position (first 300 ms).
    % 5) Builds the global design matrix X_all_raw and target matrix Y_all.
    % 6) Also builds a labels vector (from the reaching angle index, 1..8).
    % 7) Applies PCA to X_all_raw, returning:
    %      X_all_pca, mu, coeff, numComponents.
    %
    % Outputs:
    %   X_all_pca     - Global PCA features (#samples x numComponents)
    %   Y_all         - Global target matrix (#samples x 2)
    %   startHandPos  - Cell array of average starting hand positions per trial.
    %   maxTimeBins   - Maximum number of time bins across all trials.
    %   X_cell        - Cell array of design matrices (raw, one per trial).
    %   Y_cell        - Cell array of target matrices (one per trial).
    %   mu            - Mean of X_all_raw (1 x originalNeurons).
    %   coeff         - PCA loadings.
    %   numComponents - Number of principal components selected (to capture 95% variance).
    %   labels_all    - Global vector of class labels (reaching angle index) for each sample.
    
    load('monkeydata_training.mat');  % Loads variable "trial"
    
    fs = 1000;  % sampling frequency (1 ms bins)
    numTrials = size(trial,1);
    numAngles = size(trial,2);
    numNeurons = size(trial(1,1).spikes,1);
    
    % --- 1) Determine maximum trial length for padding ---
    maxTimeBins = 0;
    for a = 1:numAngles
        for t = 1:numTrials
            T = size(trial(t,a).spikes,2);
            maxTimeBins = max(maxTimeBins, T);
        end
    end
    
    % --- 2) Design low-pass filter (cutoff 100 Hz) ---
    [b,a] = butter(2, 100/(fs/2), 'low');
    
    % --- 3) Set up Gaussian smoothing kernel ---
    gaussWindow = 11;
    gaussKernel = gausswin(gaussWindow);
    gaussKernel = gaussKernel / sum(gaussKernel);
    
    % Preallocate cell arrays.
    X_cell = {};
    Y_cell = {};
    startHandPos = {};
    labels_cell = {};
    
    % Loop over each trial (over angles and trial numbers).
    for a = 1:numAngles
        for t = 1:numTrials
            spikes = double(trial(t,a).spikes);    % (numNeurons x T)
            handPos = trial(t,a).handPos;            % (3 x T)
            T = size(spikes,2);
            
            % Filter and smooth each neuron's spike train.
            spikes_processed = zeros(size(spikes));
            for nrn = 1:numNeurons
                spk = spikes(nrn,:);
                spk_filt = filtfilt(b, a, spk);
                spk_smooth = conv(spk_filt, gaussKernel, 'same');
                spikes_processed(nrn,:) = spk_smooth;
            end
            
            % Pad trial data with zeros if needed.
            if T < maxTimeBins
                padAmt = maxTimeBins - T;
                spikes_padded = [spikes_processed, zeros(numNeurons, padAmt)];
                handPos_padded = [handPos, zeros(3, padAmt)];
            else
                spikes_padded = spikes_processed;
                handPos_padded = handPos;
            end
            
            % Build design matrix for this trial (each row is a time bin).
            X_trial_raw = spikes_padded';  % (maxTimeBins x numNeurons)
            % Build target matrix using only x and y components.
            Y_trial = [handPos_padded(1,:)', handPos_padded(2,:)'];  % (maxTimeBins x 2)
            
            % Compute starting hand position (average over first 300 ms).
            nAvg = min(300, size(handPos_padded,2));
            startPos = mean(handPos_padded(1:2, 1:nAvg),2);
            
            % Save the trial data.
            X_cell{end+1} = X_trial_raw;
            Y_cell{end+1} = Y_trial;
            startHandPos{end+1} = startPos;
            % Label: use the current reaching angle index (a) for all time bins in this trial.
            labels_cell{end+1} = a * ones(size(X_trial_raw,1), 1);
        end
    end
    
    % --- 4) Concatenate all trials to form global matrices ---
    X_all_raw = vertcat(X_cell{:});  % (#samples x numNeurons)
    Y_all = vertcat(Y_cell{:});      % (#samples x 2)
    labels_all = vertcat(labels_cell{:});  % (#samples x 1)
    
    % --- 5) Apply PCA to X_all_raw ---
    [coeff, score, ~, ~, explained, mu] = pca(X_all_raw);
    cumExplained = cumsum(explained);
    numComponents = find(cumExplained >= 95, 1, 'first');  % select PCs capturing 95% variance
    X_all_pca = score(:, 1:numComponents);  % (#samples x numComponents)
    
    fprintf('Preprocessing done. Padded to %d bins, selected %d PCs.\n', maxTimeBins, numComponents);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = computeLDA(X, labels)
    % Computes the LDA transformation matrix for data X with corresponding class labels.
    % Input:
    %   X      - (n x d) data matrix (here, the PCA features)
    %   labels - (n x 1) vector of class labels (e.g., 1...8 for reaching angles)
    % Output:
    %   L      - (d x (nClasses-1)) transformation matrix that projects X into LDA space.
    
    classes = unique(labels);
    nClasses = length(classes);
    d = size(X,2);
    mu_total = mean(X, 1);
    
    % Initialize within-class scatter (Sw) and between-class scatter (Sb)
    Sw = zeros(d, d);
    Sb = zeros(d, d);
    
    for i = 1:nClasses
        Xi = X(labels == classes(i), :);
        ni = size(Xi,1);
        mu_i = mean(Xi,1);
        % Within-class scatter: sum over classes of covariance weighted by class size.
        Sw = Sw + (ni * cov(Xi, 1));
        diff = (mu_i - mu_total)';
        Sb = Sb + ni * (diff * diff');
    end
    
    % Solve the generalized eigenvalue problem: Sb * v = lambda * Sw * v
    [V, D] = eig(Sb, Sw);
    % Sort eigenvectors by descending eigenvalues.
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    % Maximum dimensionality for LDA is (nClasses - 1)
    L = V(:, 1:(nClasses - 1));
end
