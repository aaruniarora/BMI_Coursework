function [modelParameters, firingData] = positionEstimatorTraining(trainingData)
% positionEstimatorTraining   Train a Kalman filter with PCA-based dimensionality reduction.
%
% Steps:
%   1) Preprocess (bin + sqrt + smoothing).
%   2) Remove low-firing neurons.
%   3) Build a large neural observation matrix Z_all and corresponding states X_all.
%   4) Run manual PCA on Z_all. Keep enough PCs to explain ~95% of variance (configurable).
%   5) Estimate Kalman parameters (A, C, Q, R) on the dimension-reduced observations.
%   6) Compute the initial state x0, P0 from each trial’s first bin.
%   7) Store all parameters (including PCA info) in modelParameters.

    % -------------------------------------------------------------------------
    % 1) Basic parameters
    % -------------------------------------------------------------------------
    noDirections = 8;              % the data has 8 reach directions
    group       = 20;              % bin size (ms)
    win         = 50;              % smoothing window (ms)
    noTrain     = length(trainingData);
    startTime   = 320;             % training start time (ms)
    endTime     = 560;             % training end time (ms)
    TmaxBins    = endTime / group; % e.g., 560/20 = 28
    dt          = group / 1000;    % time step in seconds

    % -------------------------------------------------------------------------
    % 2) Preprocessing
    % -------------------------------------------------------------------------
    % Bin & sqrt
    trialProcess = bin_and_sqrt(trainingData, group, 1);   % 'to_sqrt' = 1
    % Gaussian smoothing -> firing rates in spikes/s
    trialFinal   = get_firing_rates(trialProcess, group, win);

    % We'll gather all trial data (from 0-560 ms). Then we pick out [320..560].
    noNeurons = size(trialFinal(1,1).rates,1);
    firingData_full = zeros(noNeurons * TmaxBins, noDirections * noTrain);

    for dir_i = 1:noDirections
        for tr_i = 1:noTrain
            for bin_i = 1:TmaxBins
                rowStart = noNeurons*(bin_i-1) + 1;
                rowEnd   = noNeurons*bin_i;
                colInd   = (dir_i-1)*noTrain + tr_i;
                firingData_full(rowStart:rowEnd, colInd) = ...
                    trialFinal(tr_i, dir_i).rates(:, bin_i);
            end
        end
    end

    % -------------------------------------------------------------------------
    % 3) Identify and remove low-firing neurons
    % -------------------------------------------------------------------------
    lowFirers = [];
    for n = 1:noNeurons
        rowIdx = n : noNeurons : size(firingData_full,1);
        meanRate = mean(mean(firingData_full(rowIdx,:)));
        if meanRate < 0.5
            lowFirers = [lowFirers, n]; %#ok<AGROW>
        end
    end
    modelParameters.kalman.lowFirers = lowFirers;

    % -------------------------------------------------------------------------
    % 4) Build big observation (Z_all) and state (X_all) matrices
    % -------------------------------------------------------------------------
    % We only use bins from startTime to endTime.
    timeBins = startTime:group:endTime;  % e.g., [320, 340, ..., 560]
    T = numel(timeBins);                 % e.g., 13 time bins

    % Downsample hand positions to match binning
    [xn, yn, xrs, yrs] = getEqualandSampled(trainingData, noDirections, noTrain, group);

    X_all = []; % 4 x (#all time points across all trials)
    Z_all = []; % M x (#all time points across all trials), M = (#neurons - #lowFirers)

    % We'll also track the first bin of each trial (for x0)
    X_init = [];  % 4 x (#trials * #directions)

    for dir_i = 1:noDirections
        for tr_i = 1:noTrain
            colInd = (dir_i-1)*noTrain + tr_i;

            % Extract the relevant portion from firingData_full
            neuralBlock = zeros(noNeurons, T);
            for k = 1:T
                binIndex = (startTime/group - 1) + k;  
                rowStart = noNeurons*(binIndex - 1) + 1;
                rowEnd   = noNeurons*binIndex;
                neuralBlock(:, k) = firingData_full(rowStart:rowEnd, colInd);
            end
            % Remove low-firing neurons
            neuralBlock(lowFirers,:) = [];

            % Get position data
            binStart = startTime/group;  % e.g., 16
            binEnd   = endTime/group;    % e.g., 28
            xVals = xrs(tr_i, binStart:binEnd, dir_i);
            yVals = yrs(tr_i, binStart:binEnd, dir_i);

            % Compute velocities
            vx = diff(xVals)/dt; % length T-1
            vy = diff(yVals)/dt; % length T-1

            % For each time bin k=1..(T-1)
            for k = 1:(T-1)
                X_k = [xVals(k); yVals(k); vx(k); vy(k)];  % 4x1
                Z_k = neuralBlock(:, k);                  % Mx1
                X_all = [X_all, X_k]; %#ok<AGROW>
                Z_all = [Z_all, Z_k]; %#ok<AGROW>

                % If it's the *first bin* in [320..560], store it in X_init
                if k == 1
                    % This is effectively the "starting" bin in the 320..560 window
                    X_init = [X_init, X_k]; %#ok<AGROW>
                end
            end
        end
    end

    % Optionally, return firingData for debugging
    firingData = Z_all;

    % -------------------------------------------------------------------------
    % 5) Manual PCA on Z_all
    % -------------------------------------------------------------------------
    varToKeep = 15;  % percentage of variance to keep
    [W_pca, Z_pca, ~, mu_pca] = myPCA(Z_all, varToKeep);
    % W_pca: (M x K), Z_pca: (K x N), mu_pca: (M x 1)

    % -------------------------------------------------------------------------
    % 6) Fit Kalman filter in the PCA space
    % -------------------------------------------------------------------------
    % X_{t+1} = A X_t + w
    % z_pca(t) = C X_t + v
    X_current = X_all(:, 1:end-1);
    X_next    = X_all(:, 2:end);
    A = X_next * pinv(X_current);
    E_w = X_next - A * X_current;
    Q = (E_w * E_w.') / size(E_w,2);

    C = Z_pca * pinv(X_all);  % (K x 4)
    E_v = Z_pca - C * X_all;
    R = (E_v * E_v.') / size(E_v,2);

    % -------------------------------------------------------------------------
    % 7) Initial state from the *average* of each trial's first bin in [320..560]
    % -------------------------------------------------------------------------
    if isempty(X_init)
        % fallback if for some reason there's no data
        x0 = mean(X_all, 2);
        P0 = cov(X_all.');
    else
        x0 = mean(X_init, 2);
        P0 = cov(X_init.');
    end

    % -------------------------------------------------------------------------
    % 8) Store in modelParameters
    % -------------------------------------------------------------------------
    KF.A  = A;
    KF.C  = C;
    KF.Q  = Q;
    KF.R  = R;
    KF.x0 = x0;
    KF.P0 = P0;
    KF.dt = dt;

    KF.PCAmean  = mu_pca;     % (M x 1)
    KF.PCAcoeff = W_pca;      % (M x K)
    KF.K        = size(W_pca, 2);

    KF.lowFirers = lowFirers;

    modelParameters.kalman = KF;
end
