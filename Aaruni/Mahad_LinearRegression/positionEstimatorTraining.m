function [modelParameters, firingData] = positionEstimatorTraining(trainingData)
    % Parameters
    noDirections = 8;
    group = 20;          % Bin size in ms
    win = 50;            % Gaussian smoothing window in ms
    noTrain = length(trainingData);
    
    % Preprocess training data: bin spikes and apply square root transform
    trialProcess = bin_and_sqrt(trainingData, group, 1);
    
    % Compute firing rates with Gaussian smoothing
    trialFinal = get_firing_rates(trialProcess, group, win);
    
    % Reach angles for directions
    reachAngles = [30 70 110 150 190 230 310 350];
    
    % Initialize model parameters structure
    modelParameters = struct;
    
    % Time points for training (320ms to 560ms in 20ms steps)
    startTime = 320;
    endTime = 560;
    count = 1;
    timePoints = [startTime:group:endTime]/group;
    
    % Initialize low-firing neuron remover
    removers = {};
    noNeurons = size(trialFinal(1,1).rates, 1);
    
    % Build initial firingData for all trials and directions up to endTime
    for i = 1:noDirections
        for j = 1:noTrain
            for k = 1:endTime/group
                firingData(noNeurons*(k-1)+1:noNeurons*k, noTrain*(i-1)+j) = trialFinal(j,i).rates(:,k);
            end
        end
    end
    
    % Identify low-firing neurons (mean firing rate < 0.5 Hz)
    lowFirers = [];
    for x = 1:noNeurons
        check_rate = mean(mean(firingData(x:98:end, :)));
        if check_rate < 0.5
            lowFirers = [lowFirers, x];
        end
    end
    clear firingData
    removers{end+1} = lowFirers;
    modelParameters.lowFirers = removers;
    
    % Process data for each time point for classification
    for trimmer = timePoints
        noNeurons = size(trialFinal(1,1).rates, 1);
        % Build firingData up to current time point
        for i = 1:noDirections
            for j = 1:noTrain
                for k = 1:trimmer
                    firingData(noNeurons*(k-1)+1:noNeurons*k, noTrain*(i-1)+j) = trialFinal(j,i).rates(:,k);
                end
            end
        end
        
        % Remove low-firing neurons
        toRemove = [];
        for x = lowFirers
            toRemove = [toRemove, x:98:length(firingData)];
        end
        firingData(toRemove, :) = [];
        noNeurons = length(firingData)/(endTime/group);
        
        % Direction labels for classification
        dirLabels = [ones(1,noTrain), 2*ones(1,noTrain), 3*ones(1,noTrain), 4*ones(1,noTrain), ...
                     5*ones(1,noTrain), 6*ones(1,noTrain), 7*ones(1,noTrain), 8*ones(1,noTrain)];
        
        % Perform PCA
        % PCA dimensions
        pcaDim = 50;
        % [princComp, eVals] = getPCA(firingData);
        [princComp, eVals] = covPCA(firingData, pcaDim);
        % variance_threshold = 0.95; % Choose number of principal components to retain 95% variance
        % [princComp, eVals] = performPCA(firingData, variance_threshold);
        
        % Compute between-class and within-class scatter matrices
        matBetween = zeros(size(firingData,1), noDirections);
        for i = 1:noDirections
            matBetween(:,i) = mean(firingData(:, noTrain*(i-1)+1:i*noTrain), 2);
        end
        scatBetween = (matBetween - mean(firingData,2)) * (matBetween - mean(firingData,2))';
        scatGrand = (firingData - mean(firingData,2)) * (firingData - mean(firingData,2))';
        scatWithin = scatGrand - scatBetween;
        
        % LDA dimensions
        ldaDim = 6;
        
        % Compute LDA projection
        [eVectsLDA, eValsLDA] = eig(((princComp(:,1:pcaDim)' * scatWithin * princComp(:,1:pcaDim))^-1) * ...
                                    (princComp(:,1:pcaDim)' * scatBetween * princComp(:,1:pcaDim)));
        [~, sortIdx] = sort(diag(eValsLDA), 'descend');
        optimOut = princComp(:,1:pcaDim) * eVectsLDA(:, sortIdx(1:ldaDim));
        
        % Project firing data onto LDA space
        mean_all = mean(firingData, 2);
        W = optimOut' * (firingData - mean_all);
        
        % Store parameters for classification
        modelParameters.classify(count).wLDA_kNN = W;
        modelParameters.classify(count).dPCA_kNN = pcaDim;
        modelParameters.classify(count).dLDA_kNN = ldaDim;
        modelParameters.classify(count).wOpt_kNN = optimOut;
        modelParameters.classify(count).mFire_kNN = mean_all;
        count = count + 1;
    end
    
    % Resample position data to match binning
    [xn, yn, xrs, yrs] = getEqualandSampled(trainingData, noDirections, noTrain, group);
    xTestInt = xrs(:, [startTime:group:endTime]/group, :);
    yTestInt = yrs(:, [startTime:group:endTime]/group, :);
    
    % Train regression models for position estimation
    modelParameters.regression = struct;
    for j = 1:length(timePoints)
        % Get projected data W for this time point
        W = modelParameters.classify(j).wLDA_kNN;
        for i = 1:noDirections
            % Extract projected data for current direction
            W_dir = W(:, noTrain*(i-1)+1:noTrain*i)';
            
            % Get position data for current time point and direction
            x4pcr = xTestInt(:, j, i);
            y4pcr = yTestInt(:, j, i);
            
            % Center the position data
            mean_x = mean(x4pcr);
            mean_y = mean(y4pcr);
            
            % Train linear regression models using least squares
            beta_x = W_dir \ (x4pcr - mean_x);
            beta_y = W_dir \ (y4pcr - mean_y);
            
            % Store regression parameters
            modelParameters.regression(i,j).beta_x = beta_x;
            modelParameters.regression(i,j).beta_y = beta_y;
            modelParameters.regression(i,j).mean_x = mean_x;
            modelParameters.regression(i,j).mean_y = mean_y;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)
    % Re-bin the spike data to a new resolution and apply square-root transform.
    % Inputs:
    %   trial   - struct containing fields 'spikes' (neurons x time)
    %   group   - new binning resolution (in ms)
    %   to_sqrt - binary flag; if 1, apply sqrt to binned spikes
    %
    % Output:
    %   trialProcessed - struct with re-binned spikes
    
    trialProcessed = struct;
    for i = 1:size(trial,2)
        for j = 1:size(trial,1)
            all_spikes = trial(j,i).spikes;  % neurons x time points
            no_neurons = size(all_spikes,1);
            no_points = size(all_spikes,2);
            t_new = 1:group:(no_points + 1);
            spikes = zeros(no_neurons, numel(t_new)-1);
            for k = 1:(numel(t_new)-1)
                spikes(:, k) = sum(all_spikes(:, t_new(k):t_new(k+1)-1), 2);
            end
            if to_sqrt
                spikes = sqrt(spikes);
            end
            trialProcessed(j,i).spikes = spikes;
        end
    end
end

function trialFinal = get_firing_rates(trialProcessed, group, scale_window)
    % Compute firing rates using Gaussian smoothing.
    % Inputs:
    %   trialProcessed - struct output from bin_and_sqrt
    %   group          - binning resolution (ms)
    %   scale_window   - scaling parameter for the Gaussian kernel (ms)
    %
    % Output:
    %   trialFinal - struct containing smoothed firing rates in field 'rates'
    
    trialFinal = struct;
    win = 10*(scale_window/group);
    normstd = scale_window/group;
    alpha = (win-1)/(2*normstd);
    temp1 = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1/((win-1)/2)).^2)';
    gaussian_window = gausstemp/sum(gausstemp);
    for i = 1:size(trialProcessed,2)
        for j = 1:size(trialProcessed,1)
            hold_rates = zeros(size(trialProcessed(j,i).spikes,1), size(trialProcessed(j,i).spikes,2));
            for k = 1:size(trialProcessed(j,i).spikes,1)
                hold_rates(k,:) = conv(trialProcessed(j,i).spikes(k,:), gaussian_window, 'same')/(group/1000);
            end
            trialFinal(j,i).rates = hold_rates;
        end
    end
end

function [prinComp, evals, sortIdx, ev] = getPCA(data)
    % Perform Principal Component Analysis (PCA) on the data.
    % Inputs:
    %   data - matrix of firing rates (neurons x time/trials)
    %
    % Outputs:
    %   prinComp - projection of data onto principal components
    %   evals    - eigenvalues (sorted in descending order)
    %   sortIdx  - indices used for sorting eigenvalues
    %   ev       - eigenvectors corresponding to eigenvalues
    %
    % Subtract the cross-trial mean
    dataCT = data - mean(data, 2);
    % Calculate covariance matrix
    covMat = dataCT' * dataCT / size(data, 2);
    % Get eigenvalues and eigenvectors
    [evects, evalsMat] = eig(covMat);
    % Sort eigenvalues and eigenvectors in descending order
    [~, sortIdx] = sort(diag(evalsMat), 'descend');
    evects = evects(:, sortIdx);
    % Project firing rate data onto the new basis
    prinComp = dataCT * evects;
    % Normalize
    prinComp = prinComp ./ sqrt(sum(prinComp.^2));
    % Extract sorted eigenvalues
    evalsDiag = diag(evalsMat);
    evals = diag(evalsDiag(sortIdx));
    % Return eigenvectors as well
    ev = evects;
end

function [score, V] = covPCA(X, nPC)
    % Compute PCA: center the data, get covariance, then the top nPC eigenvectors.
    mu = mean(X,1);
    Xc = X - mu;
    C = cov(Xc);
    [V, D] = eig(C);
    [d, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    coeff = V(:, 1:nPC);
    score = Xc * coeff;
    % Normalize each principal component (each column) to have unit norm
    normFactors = sqrt(sum(score.^2, 1));  % 1 x nPC vector, norm of each column
    score = score ./ repmat(normFactors, size(score, 1), 1);
end

function [score, singular_values] = performPCA(X, variance_threshold) %[coeff, score, mu, nPC] = 
    % Compute PCA using Singular Value Decomposition (SVD)
    % Center the data by subtracting the mean from each sample
    Xc = X - mean(X, 2);
    
    % Perform SVD on the centered data (using economy size decomposition)
    [U, S, V] = svd(Xc, 'econ');

    % Compute variance explained
    singular_values = diag(S);
    explained_variance = (singular_values.^2) / sum(singular_values.^2);
    cum_variance = cumsum(explained_variance);
    
    nPC = find(cum_variance >= variance_threshold, 1);

    % The principal component directions are given by the columns of V
    coeff = V(:, 1:nPC);
    
    % Reduce data dimensionality: Compute the projection (scores) of the data onto the principal components
    score = Xc * coeff;
end

function [xn, yn, xrs, yrs] = getEqualandSampled(data, noDirections, noTrain, group)
    % Resample and equalize position trajectories to match the binning resolution.
    % Inputs:
    %   data         - struct of training data containing hand position information
    %   noDirections - number of different reaching directions (typically 8)
    %   noTrain      - number of training samples per direction
    %   group        - new binning resolution (in ms)
    %
    % Outputs:
    %   xn, yn       - matrices of original x and y positions (dimensions: noTrain x maxLength x noDirections)
    %   xrs, yrs     - resampled position matrices according to the binning resolution
    
    % Find the maximum trajectory length
    trialHolder = struct2cell(data);
    sizes = [];
    for i = [2:3:noTrain*noDirections*3]
        sizes = [sizes, size(trialHolder{i}, 2)];
    end
    maxSize = max(sizes);
    clear trialHolder

    % Preallocate position matrices
    xn = zeros(noTrain, maxSize, noDirections);
    yn = zeros(noTrain, maxSize, noDirections);
    
    for i = 1:noDirections
        for j = 1:noTrain
            xn(j,:,i) = [data(j,i).handPos(1,:), data(j,i).handPos(1,end)*ones(1, maxSize - size(data(j,i).handPos,2))];
            yn(j,:,i) = [data(j,i).handPos(2,:), data(j,i).handPos(2,end)*ones(1, maxSize - size(data(j,i).handPos,2))];
            % Resample according to the binning size
            tempx = xn(j,:,i);
            tempy = yn(j,:,i);
            xrs(j,:,i) = tempx(1:group:end);
            yrs(j,:,i) = tempy(1:group:end);
        end
    end
end
