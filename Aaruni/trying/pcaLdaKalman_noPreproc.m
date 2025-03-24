function pcaLdaKalman_noPreproc()
    % pcaLdaKalman_noPreproc:
    %   1) Loads raw data from monkeydata_training.mat (no additional preprocessing).
    %   2) Builds the global design matrix (raw spike data), target matrix (hand positions),
    %      and a label vector (reaching angle) from the raw data.
    %   3) Applies PCA (retain components capturing 95% variance) and then LDA.
    %   4) For each trial, constructs a state vector [x; y; vx; vy] from the hand positions.
    %   5) Learns a measurement matrix H (mapping state to LDA features) via linear regression.
    %   6) Estimates process noise Q and measurement noise R.
    %   7) Runs a Kalman filter on each trial to estimate state (and hence hand position).
    %   8) Computes training RMSE and plots actual vs. predicted trajectories.
    
    % ---- Step 1: Build raw data and trial boundaries ----
    [X_all_raw, Y_all, labels_all, X_cell_raw, Y_cell, label_cell, state_cell] = buildData_noPreproc_withState();
    % X_all_raw: (#samples x numNeurons)
    % Y_all: (#samples x 2) [x,y]
    % labels_all: (#samples x 1) reaching angle index
    % state_cell: cell array with each trial's state trajectories (each row: [x,y,vx,vy])
    
    % ---- Step 2: PCA ----
    [coeff, score, ~, ~, explained, mu] = pca(X_all_raw);
    cumExplained = cumsum(explained);
    numComponents = find(cumExplained >= 95, 1, 'first');
    X_all_pca = score(:, 1:numComponents);  % (#samples x numComponents)
    
    % ---- Step 3: LDA ----
    L = computeLDA(X_all_pca, labels_all);   % L: (numComponents x (c-1))
    X_all_lda = X_all_pca * L;               % (#samples x nLDA)
    
    % ---- Step 4: Learn Measurement Matrix H ----
    % We want to relate the LDA features (observations) to the state.
    % For each sample: z = H * x, where:
    %   z: LDA feature vector (dimension nLDA)
    %   x: state vector [x; y; vx; vy] (dimension 4)
    % Build state_all by concatenating state trajectories from each trial:
    state_all = vertcat(state_cell{:});  % (#samples x 4)
    % Solve z = H*x in least-squares sense.
    % We have: X_all_lda (n x nLDA) and state_all (n x 4).
    % Solve for B in: state_all * B ≈ X_all_lda, then set H = B' so that X_all_lda ≈ H * state_all'
    B = (state_all' * state_all) \ (state_all' * X_all_lda);
    H = B';  % (nLDA x 4)
    
    % ---- Step 5: Define State Transition Model and Estimate Q, R ----
    % Use a constant-velocity model:
    A = [1 0 1 0;
         0 1 0 1;
         0 0 1 0;
         0 0 0 1];
     
    % Compute measurement residuals: r = X_all_lda - (H * state_all')'
    z_pred = (H * state_all')';  % predicted LDA features from state
    residuals = X_all_lda - z_pred;
    R = cov(residuals);  % measurement noise covariance (nLDA x nLDA)
    
    % Compute process noise Q: from state transitions: x_{t+1} - A*x_t
    state_diff = [];
    for i = 1:length(state_cell)
        x_trial = state_cell{i};  % (T x 4)
        if size(x_trial,1) > 1
            diff_trial = diff(x_trial,1,1);  % (T-1 x 4)
            predicted_diff = (A * x_trial(1:end-1,:)')';
            % The residual for each step: x_{t+1} - A*x_t
            res_trial = x_trial(2:end,:) - predicted_diff;
            state_diff = [state_diff; res_trial];
        end
    end
    Q = cov(state_diff);  % process noise covariance (4x4)
    
    % ---- Step 6: Run Kalman Filter for Each Trial ----
    numTrialsTotal = length(X_cell_raw);
    all_rmse = [];
    
    figure('Name', 'Kalman Filter: Actual vs Predicted (No Preproc)');
    hold on;
    % For legend (dummy lines)
    h1 = plot(NaN, NaN, 'b', 'LineWidth', 1);
    h2 = plot(NaN, NaN, 'r--', 'LineWidth', 1);
    legend([h1 h2], {'Actual','Predicted'});
    
    for i = 1:numTrialsTotal
        % For trial i, get raw LDA observation sequence:
        % Center and project trial raw data to PCA then LDA:
        X_trial_raw = X_cell_raw{i};  % (T x numNeurons)
        X_trial_centered = bsxfun(@minus, X_trial_raw, mu);
        X_trial_pca = X_trial_centered * coeff(:,1:numComponents);
        z_trial = X_trial_pca * L;  % (T x nLDA)
        
        % True state for trial:
        x_true = state_cell{i};  % (T x 4)
        
        % Kalman filter initialization:
        x0 = x_true(1,:)';  % initial state (column vector, 4x1)
        P0 = eye(4) * 100;  % large initial uncertainty
        
        % Run Kalman filter for the trial:
        [x_est, ~] = kalman_filter(z_trial, A, H, Q, R, x0, P0);
        x_est = x_est';  % now (T x 4)
        
        % Extract estimated positions:
        Y_pred_trial = x_est(:,1:2);  % first two components are x and y
        
        % Compute RMSE for this trial:
        rmse_trial = sqrt(mean(sum((x_true(:,1:2) - Y_pred_trial).^2,2)));
        all_rmse(end+1) = rmse_trial;
        
        % Plot actual (blue) vs predicted (red dashed)
        plot(x_true(:,1), x_true(:,2), 'b', 'LineWidth', 1);
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--', 'LineWidth', 1);
    end
    hold off;
    
    overall_rmse = mean(all_rmse);
    fprintf('Overall Training RMSE (PCA+LDA+Kalman): %.3f\n', overall_rmse);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X_all_raw, Y_all, labels_all, X_cell_raw, Y_cell, label_cell, state_cell] = buildData_noPreproc_withState()
    % buildData_noPreproc_withState:
    %   Loads monkeydata_training.mat and builds the global raw design matrix (X_all_raw)
    %   and target matrix (Y_all, using x and y positions) as well as a label vector
    %   (reaching angle). Additionally, for each trial, computes a state trajectory:
    %       state = [x, y, vx, vy]
    %   where velocity is approximated as the difference between consecutive positions (with 0 for the first sample).
    %
    % Outputs:
    %   X_all_raw  - (#samples x numNeurons) raw spike data (each row is one time bin)
    %   Y_all      - (#samples x 2) hand positions (x,y)
    %   labels_all - (#samples x 1) reaching angle labels (1..8)
    %   X_cell_raw - cell array of design matrices (one per trial, each row: one time bin)
    %   Y_cell     - cell array of target matrices (one per trial, each row: [x,y])
    %   label_cell - cell array of label vectors (one per trial)
    %   state_cell - cell array of state trajectories (one per trial, each row: [x, y, vx, vy])
    
    load('monkeydata_training.mat');  % loads variable "trial"
    
    numTrials = size(trial,1);
    numAngles = size(trial,2);
    numNeurons = size(trial(1,1).spikes,1);
    
    X_cell_raw = {};
    Y_cell = {};
    label_cell = {};
    state_cell = {};
    
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            spikes = trial(tr, angleIdx).spikes;   % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos;   % (3 x T)
            T = size(spikes,2);
            
            % Build design matrix (each row is a time bin)
            X_trial_raw = spikes';  % (T x numNeurons)
            % Use x and y positions (first two rows of handPos)
            Y_trial = [handPos(1,:)', handPos(2,:)'];  % (T x 2)
            
            % Build label vector (all time bins get the angle index)
            labels_trial = angleIdx * ones(T,1);
            
            % Compute state trajectory: state = [x, y, vx, vy]
            state_trial = zeros(T,4);
            state_trial(:,1:2) = Y_trial;
            if T > 1
                % Approximate velocity using first differences; first sample velocity set to 0
                state_trial(2:end,3:4) = diff(Y_trial,1,1);
            end
            
            X_cell_raw{end+1} = X_trial_raw;
            Y_cell{end+1} = Y_trial;
            label_cell{end+1} = labels_trial;
            state_cell{end+1} = state_trial;
        end
    end
    
    X_all_raw = vertcat(X_cell_raw{:});
    Y_all = vertcat(Y_cell{:});
    labels_all = vertcat(label_cell{:});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = computeLDA(X, labels)
    % computeLDA:
    %   Computes the LDA transformation matrix for data X given class labels.
    %   Inputs:
    %       X      - (#samples x d) data matrix in PCA space.
    %       labels - (#samples x 1) class labels (e.g., 1...8 for reaching angles).
    %   Output:
    %       L      - (d x (nClasses-1)) transformation matrix (projects data into LDA space).
    
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
        Sw = Sw + ni * cov(Xi,1);
        diff = (mu_i - mu_total)';
        Sb = Sb + ni * (diff * diff');
    end
    
    [V, D] = eig(Sb, Sw);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    L = V(:, 1:(nClasses - 1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_est, P_est] = kalman_filter(z, A, H, Q, R, x0, P0)
    % kalman_filter:
    %   Implements the standard Kalman filter.
    %   Inputs:
    %       z  - (T x m) sequence of observations (each row is observation vector).
    %       A  - (n x n) state transition matrix.
    %       H  - (m x n) measurement matrix.
    %       Q  - (n x n) process noise covariance.
    %       R  - (m x m) measurement noise covariance.
    %       x0 - (n x 1) initial state estimate.
    %       P0 - (n x n) initial estimate covariance.
    %   Outputs:
    %       x_est - (n x T) estimated state trajectory.
    %       P_est - (n x n x T) state covariance for each time step.
    
    T = size(z,1);
    n = length(x0);
    m = size(z,2);
    
    x_est = zeros(n, T);
    P_est = zeros(n, n, T);
    
    x_prev = x0;
    P_prev = P0;
    
    for t = 1:T
        % Prediction step
        x_pred = A * x_prev;
        P_pred = A * P_prev * A' + Q;
        
        % Update step
        K = P_pred * H' / (H * P_pred * H' + R);
        z_t = z(t,:)';
        x_curr = x_pred + K * (z_t - H * x_pred);
        P_curr = (eye(n) - K * H) * P_pred;
        
        x_est(:, t) = x_curr;
        P_est(:,:, t) = P_curr;
        
        x_prev = x_curr;
        P_prev = P_curr;
    end
end
