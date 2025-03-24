function pcaLdaLMS_noPreproc()
    % pcaLdaLMS_noPreproc:
    %   1) Loads monkeydata_training.mat and builds a design matrix (X_all_raw),
    %      target matrix (Y_all), and label vector (labels_all) from raw spike data.
    %   2) Applies PCA on X_all_raw (keeping components that capture 95% of the variance).
    %   3) Applies LDA on the PCA features using the reaching-angle labels.
    %   4) Uses an online LMS algorithm to learn a linear mapping from the LDA space
    %      (with a bias term) to predict (x,y) hand positions.
    %   5) Computes training RMSE and plots actual vs. predicted trajectories.
    
    % ----------------- Step 1: Build Data (no preprocessing) -----------------
    [X_all_raw, Y_all, labels_all, X_cell_raw, Y_cell, label_cell] = buildData_noPreproc();
    
    % ----------------- Step 2: PCA -----------------
    [coeff, score, ~, ~, explained, mu] = pca(X_all_raw);
    cumExplained = cumsum(explained);
    numComponents = find(cumExplained >= 95, 1, 'first');
    X_all_pca = score(:, 1:numComponents);  % (#samples x numComponents)
    
    % ----------------- Step 3: LDA -----------------
    L = computeLDA(X_all_pca, labels_all);  % L: (numComponents x (c-1)) where c is # classes
    X_all_lda = X_all_pca * L;              % (#samples x nLDA)
    
    % Add bias column to the global design matrix
    X_all_lda_bias = [X_all_lda, ones(size(X_all_lda,1),1)];
    
    % ----------------- Step 4: LMS Training -----------------
    [nSamples, nFeatures] = size(X_all_lda_bias);
    % Initialize weight matrix W (nFeatures x 2) for mapping to (x,y)
    W = zeros(nFeatures, 2);
    alpha = 1e-6;      % Learning rate (tune as needed)
    numEpochs = 100;   % Number of passes through the data
    
    for epoch = 1:numEpochs
        for i = 1:nSamples
            xi = X_all_lda_bias(i,:);   % 1 x nFeatures
            yi = Y_all(i,:);            % 1 x 2
            yhat = xi * W;              % 1 x 2
            error = yi - yhat;          % 1 x 2
            % LMS update rule: W = W + alpha * (xi' * error)
            W = W + alpha * (xi') * error;
        end
        % Optionally, compute and display RMSE for this epoch
        Y_pred_epoch = X_all_lda_bias * W;
        mse_epoch = mean(sum((Y_all - Y_pred_epoch).^2, 2));
        rmse_epoch = sqrt(mse_epoch);
        fprintf('Epoch %d, RMSE: %.3f\n', epoch, rmse_epoch);
    end
    
    % Final training RMSE
    Y_pred = X_all_lda_bias * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    fprintf('Final Training RMSE (PCA + LDA + LMS): %.3f\n', rmse_val);
    
    % ----------------- Step 5: Visualization -----------------
    figure('Name', 'Actual vs Predicted (PCA+LDA+LMS, no preproc)');
    hold on;
    % Create dummy lines for a clean legend
    h1 = plot(NaN, NaN, 'b', 'LineWidth', 1);
    h2 = plot(NaN, NaN, 'r--', 'LineWidth', 1);
    legend([h1, h2], {'Actual', 'Predicted'});
    
    idxStart = 1;
    for i = 1:length(X_cell_raw)
        T_i = size(X_cell_raw{i}, 1);  % Number of time bins in trial i
        idxEnd = idxStart + T_i - 1;
        X_trial_raw = X_all_raw(idxStart:idxEnd, :);  % (T_i x numNeurons)
        Y_trial = Y_all(idxStart:idxEnd, :);           % (T_i x 2)
        
        % Project trial raw data to PCA space (center using global mean mu)
        X_trial_centered = bsxfun(@minus, X_trial_raw, mu);
        X_trial_pca = X_trial_centered * coeff(:, 1:numComponents);
        % Project to LDA space
        X_trial_lda = X_trial_pca * L;
        % Add bias column
        X_trial_lda_bias = [X_trial_lda, ones(T_i,1)];
        % Predict with LMS weights
        Y_pred_trial = X_trial_lda_bias * W;
        
        % Plot actual trajectory (blue) and predicted trajectory (red dashed)
        plot(Y_trial(:,1), Y_trial(:,2), 'b', 'LineWidth', 1);
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--', 'LineWidth', 1);
        idxStart = idxEnd + 1;
    end
    xlabel('X position (mm)');
    ylabel('Y position (mm)');
    title('Actual (blue) vs Predicted (red dashed) Trajectories (PCA+LDA+LMS)');
    axis equal; grid on;
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_all_raw, Y_all, labels_all, X_cell_raw, Y_cell, label_cell] = buildData_noPreproc()
    % buildData_noPreproc:
    %   Loads "monkeydata_training.mat" and concatenates the raw spike data and
    %   hand positions into global matrices without any preprocessing.
    % Outputs:
    %   X_all_raw  - (#samples x numNeurons) raw spike data (each row is one time bin)
    %   Y_all      - (#samples x 2) hand positions (using x and y only)
    %   labels_all - (#samples x 1) labels corresponding to the reaching angle (1..8)
    %   X_cell_raw - cell array where each cell contains a trial's design matrix
    %   Y_cell     - cell array where each cell contains a trial's target matrix
    %   label_cell - cell array where each cell contains the label vector for that trial
    
    load('monkeydata_training.mat');  % loads variable "trial"
    
    numTrials = size(trial, 1);
    numAngles = size(trial, 2);
    numNeurons = size(trial(1,1).spikes, 1);
    
    X_cell_raw = {};
    Y_cell = {};
    label_cell = {};
    
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            % Get raw spike data and hand positions
            spikes = trial(tr, angleIdx).spikes;    % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos;      % (3 x T)
            T = size(spikes, 2);
            
            % Transpose so each row is a time bin
            X_trial_raw = spikes';  % (T x numNeurons)
            Y_trial = [handPos(1,:)', handPos(2,:)'];  % (T x 2)
            labels_trial = angleIdx * ones(T, 1);
            
            X_cell_raw{end+1} = X_trial_raw;
            Y_cell{end+1} = Y_trial;
            label_cell{end+1} = labels_trial;
        end
    end
    
    X_all_raw = vertcat(X_cell_raw{:});
    Y_all = vertcat(Y_cell{:});
    labels_all = vertcat(label_cell{:});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = computeLDA(X, labels)
    % computeLDA:
    %   Computes the LDA transformation matrix for data X with corresponding labels.
    %   Input:
    %       X      - (#samples x d) data matrix in PCA space
    %       labels - (#samples x 1) class labels (e.g., 1...8 for reaching angles)
    %   Output:
    %       L      - (d x (nClasses-1)) transformation matrix (LDA space)
    
    classes = unique(labels);
    nClasses = length(classes);
    d = size(X, 2);
    mu_total = mean(X, 1);
    
    Sw = zeros(d, d);
    Sb = zeros(d, d);
    for i = 1:nClasses
        Xi = X(labels == classes(i), :);
        ni = size(Xi, 1);
        mu_i = mean(Xi, 1);
        Sw = Sw + ni * cov(Xi, 1);
        diff = (mu_i - mu_total)';
        Sb = Sb + ni * (diff * diff');
    end
    
    [V, D] = eig(Sb, Sw);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    L = V(:, 1:(nClasses - 1));
end
