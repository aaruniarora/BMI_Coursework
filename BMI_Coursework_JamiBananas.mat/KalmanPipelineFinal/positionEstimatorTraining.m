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

    %% Kalman
    for dir_idx = 1:directions

        for tr = 1:training_length

            [A,H,Q,W] = positionEstimatorTraining_one_trial(training_data(tr,dir_idx), neurons);
            Parameters.A{tr}=A;
            Parameters.H{tr}=H;
            Parameters.Q{tr}=Q;
            Parameters.W{tr}=W;

        end
    
        % average over trials to obtain a final set of parameters 
        % (A_(dir), H_(dir), Q_(dir), W_(dir)) for each direction
        modelParameters.A{dir_idx}=sum(cat(3,Parameters.A{:}),3)./training_length; 
        modelParameters.H{dir_idx}=sum(cat(3,Parameters.H{:}),3)./training_length; 
        modelParameters.W{dir_idx}=sum(cat(3,Parameters.W{:}),3)./training_length; 
        modelParameters.Q{dir_idx}=sum(cat(3,Parameters.Q{:}),3)./training_length; 

    end
    
    %% PCR
    % poly_degree = 1;
    % modelParameters.polyd = poly_degree;
    % reg_meth = 'standard';
    % modelParameters.reg_meth = reg_meth;
    % 
    % time_division = kron(bin_group:bin_group:stop_idx, ones(1, neurons)); 
    % time_interval = start_idx:bin_group:stop_idx;
    % 
    % % modelling hand positions separately for each direction
    % for dir_idx = 1:directions
    % 
    %     % Extract the current direction's hand position data for all trials
    %     curr_X_pos = formatted_xPos(:,:,dir_idx);
    %     curr_Y_pos = formatted_yPos(:,:,dir_idx);
    % 
    %     % Loop through each time window to calculate regression coefficients that predict hand positions from neural data
    %     for win_idx = 1:((stop_idx-start_idx)/bin_group)+1
    % 
    %         % Calculate regression coefficients and the windowed firing rates for the current time window and direction
    %         [reg_coeff_X, reg_coeff_Y, win_firing] = calc_reg_coeff(win_idx, time_division, labels, ...
    %             dir_idx, spikes_matrix, pca_threshold, time_interval, curr_X_pos, curr_Y_pos,poly_degree, reg_meth);
    %         % figure; plot(regressionCoefficientsX, regressionCoefficientsY); title('PCR');
    % 
    %         % Store in model parameters
    %         modelParameters.pcr(dir_idx,win_idx).bx = reg_coeff_X;
    %         modelParameters.pcr(dir_idx,win_idx).by = reg_coeff_Y;
    %         modelParameters.pcr(dir_idx,win_idx).f_mean = mean(win_firing,2);
    % 
    %         % And store the mean hand positions across all trials for each time window   
    %         modelParameters.averages(win_idx).av_X = squeeze(mean(xPos,1));
    %         modelParameters.averages(win_idx).av_Y = squeeze(mean(yPos,1));
    % 
    %     end
    % end    
end

%% HELPER FUNCTIONS FOR PREPROCESSING OF SPIKES

function [A,H,Q,W] = positionEstimatorTraining_one_trial(training_data, selected_neurons, ...
    lag, num_bins, start_idx)
% Function for 'one trial' parameters estimation (A_(tr, dir), H_(tr, dir), 
% Q_(tr, dir), W_(tr, dir)) for a given angle

    % CONSTANTS
    nb_states = 4; % X Y Vx Vy 
    % NB: testing phase revealed that the inclusion of acceleration components in the state vector did not improved the performance of the decoder. 
    % Therefore, acceleration was excluded (only 4 states are used).

    % Build observation matrix z
    for nr = 1:length(selected_neurons)
        neuron = selected_neurons(nr);
        spike = training_data.spikes(neuron, start_idx+1:(start_idx+num_bins*lag));
        spike = reshape(spike, lag, num_bins);
        spike_count = sum(spike, 1);
        z(nr, :) = spike_count / lag;
    end


    % Build state matrix x over time bins
    x = zeros(nb_states,num_bins); % State Matrix 

    % Compute position every t=320+k*20ms 
    % x(1,:)=[X(320+20ms), X(320+40ms), ...., X(320+nb_bins*20ms)]
    % x(2,:)= [Y(320+20ms), Y(320+40ms), ...., Y(320+nb_bins*20ms)]

    for k = 1:num_bins
        x(1,k) = training_data.handPos(1,start_idx+k*lag); % X
        x(2,k) = training_data.handPos(2,start_idx+k*lag); % Y
    end

    Pos_0 = training_data.handPos(1:2,start_idx); %(x0, y0) store initial position at t=320 ms

    % Compute velocity every t=320+k*20ms 
    % Vx(1,:)=(1/20)*[X(320+20ms)-X(320ms), X(320+40ms)-X(320+20ms), ...., X(320+nb_bins*20ms)-X(320+(nb_bins-1)*20ms)]
    % Vy(2,:)= (1/20)*[Y(320+20ms)-Y(320ms), Y(320+40ms)-Y(320+20ms), ...., Y(320+nb_bins*20ms)-Y(320+(nb_bins-1)*20ms)]
    for k = 1:num_bins
        if k==1
            x(3,k) = (x(1,k)-Pos_0(1))/lag;
            x(4,k) = (x(2,k)-Pos_0(2))/lag;
        else
            x(3,k) = (x(1,k)-x(1,k-1))/lag;
            x(4,k) = (x(2,k)-x(2,k-1))/lag;
        end
    end

    % Parameters Estimations 
    % Reference: W. Wu, M. Black, Y. Gao, E. Bienenstock, M. Serruya, and J. Donoghue, "Inferring hand motion from multi-cell recordings in motor cortex 
    % using a kalman filter," (2002)

    % Useful Matrices  
    X1 = x(:,1:(end-1));
    X2 = x(:,2:end);

    % Compute A, H, W and Q
    A = X2*X1' / (X1*X1');
    W = X2*X2' - A*X1*X2';
    W = W / (num_bins-1);
    H = z*x' / (x*x');
    Q = (z*z' - H*x*z');
    Q = Q / (num_bins);

end