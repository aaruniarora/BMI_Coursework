function pcaLdaKalman_binned_flexible(binSizeMs)
    % pcaLdaKalman_binned_flexible(binSizeMs)
    %
    % Demonstrates a pipeline of:
    %   1) Splitting monkey data into train (1..50) & test (51..100) trials per angle
    %   2) Binning spikes in user-specified binSizeMs
    %   3) PCA (train only), LDA (train only)
    %   4) Constant-velocity Kalman filter for decoding test set
    %   5) Plotting actual vs predicted on the test set
    %
    % Example usage:
    %   pcaLdaKalman_binned_flexible(20);  % 20 ms bin size
    %
    % Adjust the scaling of Q and R if your filter diverges or is too stiff.

    %% ------------------------- 1) Load Data & Split -------------------------
    load('monkeydata_training.mat');  % variable "trial"
    numTrials = size(trial,1);  % e.g. 100
    numAngles = size(trial,2);  % e.g. 8
    % We'll assume each angle has 100 trials and do a 50/50 split
    trainIdx = 1:50;
    testIdx  = 51:100;

    %% ----------- 2) Build Binned Training & Testing Data -------------------
    % buildBinnedData: robust binning code
    [Xtrain_raw, stateTrain, labelsTrain, trainLens] = buildBinnedData(trial(trainIdx,:), binSizeMs);
    [Xtest_raw,  stateTest,  labelsTest,  testLens ] = buildBinnedData(trial(testIdx, :),  binSizeMs);

    % Xtrain_raw: (#trainSamples x #neurons)
    % stateTrain: (#trainSamples x 4)  -> [x, y, vx, vy]
    % labelsTrain: (#trainSamples x 1)
    % trainLens: (#trainTrials x 1), number of bins per trial

    %% ----------------- 3) PCA on Training Only -----------------
    [coeff, scoreTrain, ~, ~, explained, mu] = pca(Xtrain_raw);
    cumExp = cumsum(explained);
    numComponents = find(cumExp >= 95, 1, 'first');
    Xtrain_pca = scoreTrain(:,1:numComponents);

    %% ----------------- 4) LDA on PCA (Training) ----------------
    L = computeLDA(Xtrain_pca, labelsTrain);
    Xtrain_lda = Xtrain_pca * L;  % (#trainSamples x nLDA)
    z_train = Xtrain_lda;         % Observations for training

    % Our hidden state is x_train = stateTrain
    x_train = stateTrain;  % (#trainSamples x 4)

    %% ----------------- 5) Learn Kalman Parameters --------------
    % (a) State transition: constant velocity in binSizeMs increments
    A = [1 0 1 0;  % x_{t+1} = x_t + vx_t
         0 1 0 1;  % y_{t+1} = y_t + vy_t
         0 0 1 0;  % vx_{t+1} = vx_t
         0 0 0 1]; % vy_{t+1} = vy_t

    % (b) Measurement matrix H: we solve z = xB => B in R^(4 x nLDA)
    % Then H = B'. So z_train (#trainSamples x nLDA), x_train (#trainSamples x 4)
    B = (x_train' * x_train) \ (x_train' * z_train);  % => (4 x nLDA)
    H = B';  % => (nLDA x 4)

    % (c) Estimate Q, R and clamp them
    % Q from x(t+1) - A*x(t)
    x_pred = (A * x_train')';
    validIdx = 2:size(x_train,1);  % skip first sample if you want
    diffX = x_train(validIdx,:) - x_pred(validIdx,:);
    Q_raw = cov(diffX);
    Q = 0.01 * Q_raw;  % clamp by factor 0.01

    % R from z_train - H*x_train'
    z_pred = (H * x_train')';  % (#trainSamples x nLDA)
    diffZ = z_train - z_pred;
    R_raw = cov(diffZ);
    R = 0.01 * R_raw;

    %% ----------------- 6) Evaluate on Test Set -----------------
    % Project test data to PCA & LDA
    Xtest_centered = bsxfun(@minus, Xtest_raw, mu);
    scoreTest = Xtest_centered * coeff;         % (#testSamples x #allNeurons)
    Xtest_pca = scoreTest(:, 1:numComponents);  % (#testSamples x numComponents)
    Xtest_lda = Xtest_pca * L;                  % (#testSamples x nLDA)
    z_test = Xtest_lda;
    x_testTrue = stateTest;

    idxStart = 1;
    allRMSE = [];
    figure('Name',sprintf('Test Set: PCA+LDA+Kalman (bin=%d ms)',binSizeMs));
    hold on;
    h1 = plot(NaN, NaN, 'b', 'LineWidth',1);
    h2 = plot(NaN, NaN, 'r--','LineWidth',1);
    legend([h1,h2], {'Actual','Predicted'});

    for i = 1:length(testLens)
        T_i = testLens(i);
        idxEnd = idxStart + T_i - 1;
        z_trial = z_test(idxStart:idxEnd,:);
        x_true  = x_testTrue(idxStart:idxEnd,:);

        % Kalman initialization
        x0 = x_true(1,:)';  % 4x1
        P0 = eye(4)*10;     % initial covariance

        [x_est, ~] = kalman_filter(z_trial, A, H, Q, R, x0, P0);
        x_est = x_est';  % (T_i x 4)

        % Evaluate RMSE on positions
        pos_true = x_true(:,1:2);
        pos_est  = x_est(:,1:2);
        rmse_trial = sqrt(mean(sum((pos_true - pos_est).^2,2)));
        allRMSE(end+1) = rmse_trial;

        % Plot
        plot(pos_true(:,1), pos_true(:,2), 'b','LineWidth',1);
        plot(pos_est(:,1),  pos_est(:,2),  'r--','LineWidth',1);

        idxStart = idxEnd + 1;
    end
    axis equal; grid on;
    xlabel('X position (mm)');
    ylabel('Y position (mm)');
    title(sprintf('Test Set: Actual (blue) vs Predicted (red dashed), bin=%d ms', binSizeMs));
    hold off;

    overallRMSE = mean(allRMSE);
    fprintf('Test RMSE (Kalman, PCA+LDA, bin=%d ms): %.3f\n', binSizeMs, overallRMSE);
end

%% --------------------- SUBFUNCTIONS -------------------------

function [Xbinned, stateAll, labelsAll, trialLens] = buildBinnedData(trialStruct, binSizeMs)
    % buildBinnedData:
    %   Bins spikes in windows of binSizeMs, constructs [x, y, vx, vy],
    %   and returns angle labels. Also returns #bins for each trial.
    numTrials = size(trialStruct,1);
    numAngles = size(trialStruct,2);
    numNeurons = size(trialStruct(1,1).spikes,1);

    Xcell = {};
    stateCell = {};
    labelCell = {};
    trialLens = zeros(numTrials*numAngles,1);

    idx = 1;
    for angleIdx = 1:numAngles
        for t = 1:numTrials
            spk = trialStruct(t, angleIdx).spikes;    % (numNeurons x T)
            handPos = trialStruct(t, angleIdx).handPos; % (3 x T)
            T_full = size(spk,2);

            nBins = ceil(T_full / binSizeMs);
            spikes_binned = zeros(nBins, numNeurons);
            state_trial   = zeros(nBins, 4);

            for b = 1:nBins
                startIdx = (b-1)*binSizeMs + 1;
                endIdx   = min(b*binSizeMs, T_full);

                spikes_binned(b,:) = sum(spk(:, startIdx:endIdx),2)';
                x_mean = mean(handPos(1, startIdx:endIdx));
                y_mean = mean(handPos(2, startIdx:endIdx));
                state_trial(b,1) = x_mean;
                state_trial(b,2) = y_mean;

                if b > 1
                    state_trial(b,3) = state_trial(b,1) - state_trial(b-1,1); % vx
                    state_trial(b,4) = state_trial(b,2) - state_trial(b-1,2); % vy
                end
            end

            Xcell{end+1}    = spikes_binned;   % (nBins x numNeurons)
            stateCell{end+1}= state_trial;      % (nBins x 4)
            labelCell{end+1}= angleIdx*ones(nBins,1);
            trialLens(idx)  = nBins;
            idx = idx + 1;
        end
    end

    Xbinned   = vertcat(Xcell{:});
    stateAll  = vertcat(stateCell{:});
    labelsAll = vertcat(labelCell{:});
end

function L = computeLDA(X, labels)
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
        Sw = Sw + ni*cov(Xi,1);
        diff = (mu_i - mu_total)';
        Sb = Sb + ni*(diff*diff');
    end

    [V, D] = eig(Sb, Sw);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    L = V(:,1:(nClasses-1));  % c-1 dims
end

function [x_est, P_est] = kalman_filter(z, A, H, Q, R, x0, P0)
    % Standard Kalman filter
    T = size(z,1);
    n = length(x0);
    x_est = zeros(n,T);
    P_est = zeros(n,n,T);

    x_prev = x0;
    P_prev = P0;
    for t = 1:T
        % predict
        x_pred = A*x_prev;
        P_pred = A*P_prev*A' + Q;

        % update
        K = P_pred*H'/(H*P_pred*H' + R);
        z_t = z(t,:)';
        x_curr = x_pred + K*(z_t - H*x_pred);
        P_curr = (eye(n) - K*H)*P_pred;

        x_est(:,t) = x_curr;
        P_est(:,:,t) = P_curr;
        x_prev = x_curr;
        P_prev = P_curr;
    end
end
