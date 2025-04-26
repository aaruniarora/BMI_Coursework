function modelParameters = positionEstimatorTraining(training_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSITION ESTIMATOR TRAINING FUNCTION
%
% Trains a full decoding model to estimate hand positions from neural spikes.
% Pipeline:
%   1. Preprocesses neural data:
%       - Pads all trials to max length
%       - Bins spikes in 20 ms intervals
%       - Applies smoothing filter (EMA or Gaussian)
%   2. Removes neurons with low firing rate (< 0.5 spk/s)
%   3. Extracts features and assigns direction labels
%   4. Applies PCA for dimensionality reduction
%   5. Applies LDA to find class-discriminative features
%   6. Stores features and labels for later kNN decoding
%   7. Trains a regression model (PCR - ridge and lasso optional) to map spikes to (x,y)
%
% Outputs:
%   modelParameters - struct storing all learned parameters:
%       - preprocessing config
%       - removed neurons
%       - PCA/LDA matrices
%       - regression coefficients
%       - training data in discriminative space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %% Parameters
    [training_length, directions] = size(training_data); 
    bin_group = 20; % hypertuned 
    alpha = 0.35; % hypertuned
    sigma = 50;  % standard deviation in ms
    start_idx = 300 + bin_group; 

    % Find min time length
    spike_cells = {training_data.spikes};  % Extract spike fields into a cell array
    min_time_length = min(cellfun(@(sp) size(sp, 2), spike_cells(:))); 
    clear spike_cells;
    
    % Calculate stop_index based on bin_group
    stop_idx = floor((min_time_length - start_idx) / bin_group) * bin_group + start_idx;
    time_bins = start_idx:bin_group:stop_idx;  % e.g. 320:20:560
    num_bins = time_bins / bin_group; % eg. 13

    % Store in modelParameters
    modelParameters = struct;
    modelParameters.start_idx = start_idx;
    modelParameters.stop_idx  = stop_idx;
    modelParameters.directions = directions;
    modelParameters.trial_id = 0;
    modelParameters.iterations = 0;

    % ---------- Kalman containers (one set per reach direction) -------------
    modelParameters.kalman = struct( ...
            'A',  [],  ...  % state‑transition
            'C',  [],  ...  % observation
            'Q',  [],  ...  % process‑noise covariance
            'R',  [],  ...  % observation‑noise covariance
            'Pi', [],  ...  % mean initial state  (x,y,vx,vy)
            'Vi', []);      % initial state covariance


    %% Spikes Preprocessing: Binning (20ms), Sqrt Transformation, EMA Smoothing
    preprocessed_data = preprocessing(training_data, bin_group, 'EMA', alpha, sigma, 'nodebug');
    orig_neurons = size(preprocessed_data(1,1).rate, 1);

    %% Remove data from neurons with low firing rates.
    [spikes_mat, ~] = extract_features(preprocessed_data, orig_neurons, stop_idx/bin_group, 'nodebug');
    removed_neurons = remove_neurons(spikes_mat, orig_neurons, 'nodebug');
    neurons = orig_neurons - length(removed_neurons);
    modelParameters.removeneurons = removed_neurons;
    clear spikes_mat

    %% Dimensionality parameters
    pca_threshold = 0.44; % =40 for cov and =0.44 for svd
    lda_dim = 6;
 
    for curr_bin = 1: length(num_bins)
        %% Extract features/restructure data for further analysis
        [spikes_matrix, labels] = extract_features(preprocessed_data, orig_neurons, num_bins(curr_bin), 'nodebug');
        
        %% Remove data from neurons with low firing rates.
        spikes_matrix(removed_neurons, : ) = [];

        %% PCA for dimensionality reduction of the neural data
        [~, score, nPC] = perform_PCA(spikes_matrix, pca_threshold, 'nodebug');

        %% LDA to maximise class separability across different directions
        [outputs, weights] = perform_LDA(spikes_matrix, score, labels, lda_dim, training_length, 'nodebug');

        %% kNN training: store samples in LDA space with corresponding hand positions
        modelParameters.class(curr_bin).PCA_dim = nPC;
        modelParameters.class(curr_bin).LDA_dim = lda_dim;

        modelParameters.class(curr_bin).lda_weights = weights;
        modelParameters.class(curr_bin).lda_outputs= outputs;
    
        modelParameters.class(curr_bin).mean_firing = mean(spikes_matrix, 2);
        modelParameters.class(curr_bin).labels = labels(:)';
    end

    %% Hand Positions Preprocessing: Binning (20ms), Centering, Padding
    [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, num_bins*bin_group);
    
    %% PCR
    poly_degree = 1;
    modelParameters.polyd = poly_degree;
    reg_meth = 'standard';
    modelParameters.reg_meth = reg_meth;

    time_division = kron(bin_group:bin_group:stop_idx, ones(1, neurons)); 
    time_interval = start_idx:bin_group:stop_idx;

    % modelling hand positions separately for each direction
    % CONSTANTS --------------------------------------------------------------
    dt = bin_group/1000;                 % 20 ms  → 0.02 s
    A_nominal = [1 0 dt 0; 0 1 0 dt;    % constant‑velocity model
                 0 0 1  0; 0 0 0  1];
    
    for dir_idx = 1:directions
        % ----------  Assemble per‑direction training matrices  --------------
        %  z(k)  = (binned EMA‑smoothed firing at time‑bin k)  [F x 1]
        %  x(k)  = [x; y; vx; vy]                              [4 x 1]
    
        Xk   = [];           % state at bin k      (4 × N)
        Xk1  = [];           % state at bin k+1
        Zk   = [];           % observation at k    (F × N)
    
        for tr = 1:training_length
            % extract hand position trajectory for this trial & direction
            pos = formatted_xPos(tr,:,dir_idx);   % x‑positions at all bins
            pos = [pos; formatted_yPos(tr,:,dir_idx)];        % 2×T
            vel = diff([pos(:,1) pos],1,2)/dt;                % finite diff
            vel(:,end) = vel(:,end-1);                        % same length
    
            X   = [pos ; vel];                                % 4×T
            Xk  = [Xk  X(:,1:end-1)];
            Xk1 = [Xk1 X(:,2:end)];
    
            % build observation matrix – **exactly the features already
            % computed for soft‑kNN** so nothing elsewhere changes
            bin_cnt = extract_features(preprocessed_data(tr,dir_idx), ...
                                       orig_neurons, size(X,2), 'nodebug');
            % bin_cnt(removed_neurons,:) = [];                  % keep neurons
            % Zk  = [Zk  reshape(bin_cnt, size(bin_cnt,1), [])];
            % For this trial, take binned firing rates
            rates = preprocessed_data(tr,dir_idx).rate;       % neurons × numBins
            rates(removed_neurons,:) = [];                     % drop low‐firing neurons
            rates = rates(:, num_bins);                   % F×T
            Zk = [Zk rates(:,1:end-1)];                       % align to Xk(:,1:end-1)
        end
    
        % ----------  Least‑squares estimation of Kalman matrices ------------
        % State transition A
        A = Xk1 * pinv(Xk);                                   % (4×4)
        % Force A to stay close to constant‑velocity model to prevent
        % pathological fits                       (optional but robust)
        lambda_A = 0.02;      % small ridge
        A = (1-lambda_A)*A + lambda_A*A_nominal;
    
        % Observation matrix C
        size(Zk), size(Xk)
        % C = Zk  .* pinv(Xk);                                   % (F×4)
        C = (Zk * Xk') / (Xk * Xk' + 1e-6*eye(4));
    
        % Noise covariances
        W = Xk1 - A*Xk;
        V = Zk  - C*Xk;
        Q = cov(W');                                          % (4×4)
        R = cov(V');                                          % (F×F)
        % small diagonal boost for numerical stability
        epsQ = 1e-4*eye(4);     Q = Q + epsQ;
        epsR = 1e-3*eye(size(R,1)); R = R + epsR;
    
        % Initial state mean/covariance
        Pi = mean(X(:,1:3),2);             % average of first three bins
        Vi = diag(var(X(:,1:3),0,2)+1);    % diagonal, inflated by +1
    
        % ----------  Store ---------------------------------------------------
        modelParameters.kalman(dir_idx).A  = A;
        modelParameters.kalman(dir_idx).C  = C;
        modelParameters.kalman(dir_idx).Q  = Q;
        modelParameters.kalman(dir_idx).R  = R;
        modelParameters.kalman(dir_idx).Pi = Pi;
        modelParameters.kalman(dir_idx).Vi = Vi;
    end
end