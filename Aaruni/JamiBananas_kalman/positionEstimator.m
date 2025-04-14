function [x, y, modelParameters] = positionEstimator(past_current_trial, modelParameters)
    % Parameters
    noDirections = 8;
    group = 20;          % Bin size in ms
    win = 50;            % Gaussian smoothing window in ms
    
    % Preprocess test trial: bin spikes and apply square root transform
    trialProcess = bin_and_sqrt(past_current_trial, group, 1);
    
    % Compute firing rates with Gaussian smoothing
    trialFinal = get_firing_rates(trialProcess, group, win);
    
    % Reach angles for directions (not used further in this code)
    reachAngles = [30 70 110 150 190 230 310 350];
    
    % Current time in ms
    T_end = size(past_current_trial.spikes, 2);
    noNeurons = size(trialFinal(1,1).rates, 1);
    
    % Determine time point index and process trial accordingly
    if T_end <= 560
        indexer = floor((T_end - 320)/group) + 1;
        if indexer < 1
            indexer = 1;
        elseif indexer > 13
            indexer = 13;
        end
        % Remove low-firing neurons
        lowFirers = modelParameters.lowFirers{1};
        trialFinal.rates(lowFirers, :) = [];
        % Prepare firing data (reshape into a vector)
        firingData = reshape(trialFinal.rates, [], 1);
        noNeurons = noNeurons - length(lowFirers);
        % Get projection parameters from the training model
        optimOut = modelParameters.classify(indexer).wOpt_kNN;
        mean_all = modelParameters.classify(indexer).mFire_kNN;
        % Ensure firingData size matches the model dimensions
        if size(firingData,1) > size(optimOut,1)
            firingData = firingData(1:size(optimOut,1));
        end
        % Project test data onto the LDA space
        WTest = optimOut' * (firingData - mean_all);
        % Classify direction using kNN
        WTrain = modelParameters.classify(indexer).wLDA_kNN;
        ldaDim = modelParameters.classify(indexer).dLDA_kNN;
        outLabel = getKNNs(WTest, WTrain, ldaDim, 8);
        modelParameters.actualLabel = outLabel;
    else
        % For T_end > 560 ms, use the last predicted direction
        outLabel = modelParameters.actualLabel;
        indexer = 13;  % Corresponds to 560 ms
        % Remove low-firing neurons
        lowFirers = modelParameters.lowFirers{1};
        trialFinal.rates(lowFirers, :) = [];
        % Prepare firing data
        firingData = reshape(trialFinal.rates, [], 1);
        noNeurons = noNeurons - length(lowFirers);
        % Get projection parameters from the training model
        optimOut = modelParameters.classify(indexer).wOpt_kNN;
        mean_all = modelParameters.classify(indexer).mFire_kNN;
        if size(firingData,1) > size(optimOut,1)
            firingData = firingData(1:size(optimOut,1));
        end
        % Project test data onto the LDA space
        WTest = optimOut' * (firingData - mean_all);
    end
    
    % Predict position using linear regression parameters stored in the model
    beta_x = modelParameters.regression(outLabel, indexer).beta_x;
    beta_y = modelParameters.regression(outLabel, indexer).beta_y;
    mean_x = modelParameters.regression(outLabel, indexer).mean_x;
    mean_y = modelParameters.regression(outLabel, indexer).mean_y;
    
    % Compute position estimates
    x = WTest' * beta_x + mean_x;
    y = WTest' * beta_y + mean_y ;
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

function [outLabels] = getKNNs(WTest, WTrain, dimLDA, nearFactor)
    % Perform k-nearest neighbors classification on projected data.
    % Inputs:
    %   WTest    - dimLDA x 1 vector from the test trial after projection
    %   WTrain   - dimLDA x N matrix from the training data after projection
    %   dimLDA   - number of LDA dimensions used
    %   nearFactor - parameter for kNN (not used in current implementation)
    %
    % Output:
    %   outLabels - predicted direction label (mode of nearest neighbors)
    
    trainMat = WTrain';
    testMat = WTest;
    trainSq = sum(trainMat .* trainMat, 2);
    testSq = sum(testMat .* testMat, 1);
    % Compute squared Euclidean distances (each row corresponds to a test point)
    allDists = trainSq(:, ones(1, length(testMat))) + testSq(ones(1, length(trainMat)), :) - 2 * trainMat * testMat;
    allDists = allDists';
    
    % Sort distances and select k nearest neighbors
    k = 25;
    [~, sorted] = sort(allDists, 2);
    nearest = sorted(:, 1:k);
    
    % Generate direction labels for training data (assumes equal number per direction)
    noTrain = size(WTrain, 2) / 8;
    dirLabels = [ones(1, noTrain), 2*ones(1, noTrain), 3*ones(1, noTrain), 4*ones(1, noTrain), ...
                 5*ones(1, noTrain), 6*ones(1, noTrain), 7*ones(1, noTrain), 8*ones(1, noTrain)]';
    nearestLabs = reshape(dirLabels(nearest), [], k);
    outLabels = mode(mode(nearestLabs, 2));
end
