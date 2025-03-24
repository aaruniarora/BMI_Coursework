function pcaLdaLR_noPreproc()
    % pcaLdaLR_noPreproc:
    %   1) Loads monkeydata_training.mat
    %   2) NO preprocessing (no filter, no smoothing, no padding)
    %   3) PCA on raw spikes
    %   4) LDA on PCA features
    %   5) Linear regression (least squares) to predict (x,y) positions
    %   6) Plots actual vs. predicted trajectories for the training data
    
    % Build the design matrix X_all, target Y_all, label vector labels_all
    [X_all_raw, Y_all, labels_all, X_cell_raw, Y_cell, label_cell] = buildData_noPreproc();
    
    % ------------------- PCA -------------------
    [coeff, score, ~, ~, explained, mu] = pca(X_all_raw);
    cumExplained = cumsum(explained);
    % For example, keep components that capture 95% variance
    numComponents = find(cumExplained >= 95, 1, 'first');
    X_all_pca = score(:, 1:numComponents);   % (#samples x numComponents)
    
    % ------------------- LDA -------------------
    L = computeLDA(X_all_pca, labels_all);   % transforms PCA space -> LDA space
    X_all_lda = X_all_pca * L;               % (#samples x nLDA)
    
    % ------------------- Linear Regression (LS) -------------------
    % We add a bias column
    X_all_lda_bias = [X_all_lda, ones(size(X_all_lda,1),1)];
    % Solve W in least squares: W = (X^T X)^(-1) X^T Y
    W = (X_all_lda_bias' * X_all_lda_bias) \ (X_all_lda_bias' * Y_all);
    
    % ------------------- Compute Training RMSE -------------------
    Y_pred = X_all_lda_bias * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    
    fprintf('No preprocessing. PCA -> %d comps, LDA -> %d dims.\n', ...
            numComponents, size(L,2));
    fprintf('Training RMSE (PCA + LDA + LR): %.3f\n', rmse_val);
    
    % ------------------- Visualization -------------------
    figure('Name','All Trials: Actual vs Predicted (No Preproc)');
    hold on;
    % "Dummy" lines for a clean legend
    h1 = plot(NaN, NaN, 'b', 'LineWidth',1);
    h2 = plot(NaN, NaN, 'r--','LineWidth',1);
    legend([h1 h2], {'Actual','Predicted'});
    
    idxStart = 1;
    for i = 1:length(X_cell_raw)
        T_i = size(X_cell_raw{i},1);  % number of time bins in this trial
        idxEnd = idxStart + T_i - 1;
        
        % Extract the raw portion for this trial from the global array
        X_trial_raw = X_all_raw(idxStart:idxEnd, :);  % (T_i x numNeurons)
        Y_trial = Y_all(idxStart:idxEnd, :);          % (T_i x 2)
        
        % Project to PCA
        X_trial_centered = bsxfun(@minus, X_trial_raw, mu);
        X_trial_pca = X_trial_centered * coeff(:,1:numComponents);
        % Then to LDA
        X_trial_lda = X_trial_pca * L;
        % Add bias
        X_trial_lda_bias = [X_trial_lda, ones(T_i,1)];
        % Predict
        Y_pred_trial = X_trial_lda_bias * W;
        
        % Plot
        plot(Y_trial(:,1), Y_trial(:,2), 'b','LineWidth',1);
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--','LineWidth',1);
        
        idxStart = idxEnd + 1;
    end
    xlabel('X position (mm)');
    ylabel('Y position (mm)');
    title('All Trials: Actual (blue) vs Predicted (red dashed), No Preproc');
    axis equal; grid on;
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_all_raw, Y_all, labels_all, X_cell_raw, Y_cell, label_cell] = buildData_noPreproc()
    % buildData_noPreproc:
    %   Loads "monkeydata_training.mat" and simply concatenates the raw spikes
    %   and hand positions, with a label for each angle. 
    %   NO smoothing, NO filtering, NO padding, NO averaging.
    %
    % Outputs:
    %   X_all_raw   - (#samples x numNeurons)
    %   Y_all       - (#samples x 2) hand positions
    %   labels_all  - (#samples x 1) angle labels (1..8)
    %   X_cell_raw  - cell array of design matrices (one per trial, dimension: T_i x numNeurons)
    %   Y_cell      - cell array of target matrices (one per trial, dimension: T_i x 2)
    %   label_cell  - cell array of label vectors (one per trial, dimension: T_i x 1)
    
    load('monkeydata_training.mat');  % loads "trial"
    
    numTrials = size(trial,1);   % e.g. 100
    numAngles = size(trial,2);   % e.g. 8
    numNeurons = size(trial(1,1).spikes,1);  % e.g. 98
    
    X_cell_raw = {};
    Y_cell = {};
    label_cell = {};
    
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            % Extract raw spikes and raw handPos
            spikes = trial(tr, angleIdx).spikes;   % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos; % (3 x T)
            T = size(spikes,2);
            
            % Transpose so each row is one time bin
            X_trial_raw = spikes';           % (T x numNeurons)
            Y_trial = [handPos(1,:)', handPos(2,:)'];  % (T x 2)
            
            % Label each time bin by the angle index
            labels_trial = angleIdx * ones(T,1);
            
            X_cell_raw{end+1} = X_trial_raw;
            Y_cell{end+1} = Y_trial;
            label_cell{end+1} = labels_trial;
        end
    end
    
    % Concatenate
    X_all_raw = vertcat(X_cell_raw{:});   % (#samples x numNeurons)
    Y_all = vertcat(Y_cell{:});           % (#samples x 2)
    labels_all = vertcat(label_cell{:});  % (#samples x 1)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = computeLDA(X, labels)
    % computeLDA:
    %   Standard LDA to reduce from PCA dimension to (c-1) dimension for c classes.
    % Inputs:
    %   X      - (#samples x d) data matrix in PCA space
    %   labels - (#samples x 1) class labels (1..8 for angles)
    % Output:
    %   L      - (d x (c-1)) transformation matrix
    
    classes = unique(labels);
    nClasses = length(classes);
    d = size(X,2);
    
    mu_total = mean(X,1);
    Sw = zeros(d,d);
    Sb = zeros(d,d);
    
    for i = 1:nClasses
        Xi = X(labels == classes(i), :);
        ni = size(Xi,1);
        mu_i = mean(Xi,1);
        Sw = Sw + ni * cov(Xi,1);  % within-class scatter
        diff = (mu_i - mu_total)';
        Sb = Sb + ni * (diff * diff');  % between-class scatter
    end
    
    [V, D] = eig(Sb, Sw);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    % Keep up to c-1 LDA dimensions
    L = V(:,1:(nClasses - 1));
end
