function [modelParameters, firingData] = positionEstimatorTraining(trainingData)
% POSITIONESTIMATORTRAINING
% This function trains a model to predict hand position from neural data.
%
% Steps:
%   (1) Bin + sqrt transform spikes
%   (2) Gaussian smoothing
%   (3) Remove low-firing neurons
%   (4) For each direction & each time bin window:
%       - Extract data for that direction/time
%       - Perform PCA
%       - Regress x,y positions in the PCA space
%
% OUTPUTS:
%   modelParameters - struct containing PCA vectors, means, and regression parameters
%   firingData      - (optional) the full preprocessed firing data (before direction splits)
%
% By: [Your Name], following the plan to split data by direction, discard LDA,
% and do direction-specific PCA + regression.


%% ------------- 1. Set parameters and do initial preprocessing -------------
noDirections = 8;            % Typically 8 directions
group       = 20;            % Bin size in ms
win         = 50;            % Gaussian smoothing window in ms
noTrain     = length(trainingData);  % # of trials per direction

% Binning + sqrt transform
trialProcess = bin_and_sqrt(trainingData, group, 1);
% Gaussian smoothing
trialFinal   = get_firing_rates(trialProcess, group, win);

% Define time points of interest (in increments of 'group' ms)
startTime    = 320;
endTime      = 560;
timePoints   = [startTime:group:endTime] / group;  % e.g. 16:20:28 (if group=20 ms)
modelParameters = struct;   % main storage

%% ------------- 2. Build a big matrix up to endTime for removing low firers -------------
noNeurons = size(trialFinal(1,1).rates,1);
maxBins   = endTime / group;  % e.g. 560/20 = 28
firingData = [];             % Will be (neurons * maxBins) x (noTrain * noDirections)

% Concatenate all directions/trials (up to endTime) into one big matrix
for iDir = 1:noDirections
    for jTrial = 1:noTrain
        rates_ij = trialFinal(jTrial, iDir).rates; % (neurons x bins)
        % We only take columns up to maxBins
        rates_ij = rates_ij(:, 1:maxBins);
        % Place into big matrix (stacking bins in the row dimension)
        for kBin = 1:maxBins
            rowStart = (kBin-1)*noNeurons + 1;
            rowEnd   = kBin*noNeurons;
            firingData(rowStart:rowEnd, (iDir-1)*noTrain + jTrial) = rates_ij(:, kBin);
        end
    end
end

%% ------------- 3. Remove low-firing neurons -------------
% We'll define 'low firing' as mean rate < 0.5 Hz
% We check the average across all directions/trials/bins
lowFirers = [];
for nn = 1:noNeurons
    % Indices in firingData that correspond to this neuron across bins
    % i.e. nn: noNeurons : end
    bigMatrixIndices = nn:noNeurons:size(firingData,1);
    check_rate = mean(mean(firingData(bigMatrixIndices, :), 2), 1); 
    if check_rate < 0.5
        lowFirers = [lowFirers, nn];
    end
end
modelParameters.lowFirers = lowFirers;  % store for test-time removal
fprintf('Number of low-firing neurons removed: %d\n', length(lowFirers));

%% ------------- 4. Prepare position data (for regression) -------------
[xn, yn, xrs, yrs] = getEqualandSampled(trainingData, noDirections, noTrain, group);
% xrs, yrs have dimensions: noTrain x (#bins) x noDirections
% We'll focus on the time bins from startTime to endTime:
xTestInt = xrs(:, timePoints, :);  % noTrain x length(timePoints) x noDirections
yTestInt = yrs(:, timePoints, :);

%% ------------- 5. Train separate PCA + regression for each direction & time bin -------------
% We will store model parameters in a struct of the form:
%   modelParameters.direction(dir).timeBin(t).{pcaVectors, pcaMean, betaX, betaY, etc.}
lambda = 1;  % ridge penalty
numTimeBins = length(timePoints);

for iDir = 1:noDirections
    
    % Extract *all* bins for that direction (up to maxBins=28 if 560 ms / 20 ms)
    % from the big matrix, but only that direction's trials
    dirColStart = (iDir-1)*noTrain + 1;
    dirColEnd   = iDir*noTrain;
    
    % We will not do LDA or combine directions. Instead, for each time bin t,
    % we do PCA on the data up to that bin, *only for this direction*.
    
    for tIdx = 1:numTimeBins
        % timePoints(tIdx) is the bin index in [startTime:group:endTime]/group
        currentTimeBin = timePoints(tIdx);  % e.g. 16, 17, 18,... if (320->560)/20
        
        % 5a) Build firingData_dir up to current time bin for this direction
        %     (neurons x currentTimeBin) x noTrain
        firingData_dir = [];
        for trialIdx = 1:noTrain
            % rates( neurons x bins ) for that single trial
            rates_ij = trialFinal(trialIdx, iDir).rates; 
            rates_ij = rates_ij(:, 1:currentTimeBin);  % up to the tIdx bin
            % stack along row dimension
            for kBin = 1:currentTimeBin
                rowStart = (kBin-1)*noNeurons + 1;
                rowEnd   = kBin*noNeurons;
                firingData_dir(rowStart:rowEnd, trialIdx) = rates_ij(:, kBin);
            end
        end
        
        % 5b) Remove low-firing neurons from firingData_dir
        toRemove = [];
        for lf = lowFirers
            % lf, lf+ noNeurons, lf+2*noNeurons, ...
            toRemove = [toRemove, lf:noNeurons:size(firingData_dir,1)];
        end
        firingData_dir(toRemove, :) = [];
        modelParameters.direction(iDir).timeBin(tIdx).avgFiring = mean(firingData_dir,2);
        % 5c) PCA on direction-specific data
        %     We can store all data in a matrix dataDir ( nNeurons_dir * currentTimeBin ) x noTrain
        %     Then do standard PCA to get principal components
        [score, pcaStruct] = doPCA(firingData_dir);
        % 'score' is dimension: (neurons * currentTimeBin) x #PCs
        % pcaStruct returns .meanRate (the column mean we subtracted), .eigVec, .topPC = #PC used, etc.
        
        % 5d) Project each trial's firing (for that direction/time) to PCA space
        %     Then do ridge regression from PC scores -> x, y
        % Typically we'd choose how many PCs to keep, e.g. 30, but let's let pcaStruct keep them all,
        % or store a user-defined number (like pcaDim=30).
        pcaDim = 30;  % you can tune this
        pcaDim = min(pcaDim, size(score,2)); % in case #PC < 30
        PC_scores = score(:, 1:pcaDim);  % (neurons*timeBin) x pcaDim
        
        % Prepare the regression target for each trial
        % We want x and y at time bin tIdx:
        xTarget = xTestInt(:, tIdx, iDir);  % noTrain x 1
        yTarget = yTestInt(:, tIdx, iDir);  % noTrain x 1
        % Subtract mean (optional, can help center the data)
        mean_x  = mean(xTarget);
        mean_y  = mean(yTarget);
        xCtr    = xTarget - mean_x;
        yCtr    = yTarget - mean_y;
        
        % Build design matrix from PC_scores, which is dimension:
        %   #rows = (neurons * timeBins) but we actually want to combine trials along that dimension.
        % Actually each column in firingData_dir corresponds to one trial. We have PC_scores for each trial too.
        % So let's transpose PC_scores to get noTrain x pcaDim
        % But we must confirm that 'score' is (all bins x all trials)? 
        %   In doPCA, we typically do data as (rows=neurons/time) x (cols=trials). 
        % So let's handle it carefully in doPCA.  
        % For now, let's just do "scoresTrial = PC_scores' * (firingData_dir - pcaStruct.meanRate(:))" approach 
        % but to keep it simpler, let's do it the direct way:
        
        % We'll do trial-wise approach:
        Xdesign = zeros(noTrain, pcaDim); 
        for trialIdx = 1:noTrain
            % Grab the columns in firingData_dir that correspond to this single trial
            singleTrialVector = firingData_dir(:, trialIdx); 
            % Subtract PCA mean
            singleTrialVector = singleTrialVector - pcaStruct.meanRate;
            % Project onto top pcaDim components
            Xdesign(trialIdx, :) = singleTrialVector' * pcaStruct.eigVec(:, 1:pcaDim);
        end
        
        % Now we do ridge regression: beta = (X^T X + lambda I)^(-1) X^T y
        % for x, then for y
        % Xdesign: noTrain x pcaDim
        % We'll do a small identity for pcaDim
        regTerm = lambda * eye(pcaDim);
        beta_x  = (Xdesign' * Xdesign + regTerm) \ (Xdesign' * xCtr);
        beta_y  = (Xdesign' * Xdesign + regTerm) \ (Xdesign' * yCtr);
        
        % Store the parameters
        modelParameters.direction(iDir).timeBin(tIdx).pcaMean     = pcaStruct.meanRate;
        modelParameters.direction(iDir).timeBin(tIdx).eigVec      = pcaStruct.eigVec(:, 1:pcaDim);
        modelParameters.direction(iDir).timeBin(tIdx).pcaDim      = pcaDim;
        
        modelParameters.direction(iDir).timeBin(tIdx).beta_x      = beta_x;
        modelParameters.direction(iDir).timeBin(tIdx).beta_y      = beta_y;
        modelParameters.direction(iDir).timeBin(tIdx).mean_x      = mean_x;
        modelParameters.direction(iDir).timeBin(tIdx).mean_y      = mean_y;

    end
end

% Store some other metadata
modelParameters.group       = group;
modelParameters.smoothWin   = win;
modelParameters.startTime   = startTime;
modelParameters.endTime     = endTime;
modelParameters.timePoints  = timePoints;

end % end of main function


%% ---------------- HELPER FUNCTIONS ----------------

function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)
% Re-bin the spike data to 'group' ms resolution and optionally apply sqrt transform
% trial: struct of size (#trials x #directions)
% group: bin size (ms)
% to_sqrt: if 1, apply sqrt
    trialProcessed = struct;
    for i = 1:size(trial,2)      % directions
        for j = 1:size(trial,1)  % trials
            all_spikes = trial(j,i).spikes;  % [neurons x timePoints]
            no_neurons = size(all_spikes,1);
            no_points  = size(all_spikes,2);
            t_new      = 1:group:(no_points+1);
            
            spikes = zeros(no_neurons, numel(t_new)-1);
            for k = 1:numel(t_new)-1
                spikes(:, k) = sum(all_spikes(:, t_new(k):t_new(k+1)-1), 2);
            end
            if to_sqrt
                spikes = sqrt(spikes);
            end
            trialProcessed(j,i).spikes = spikes;
            trialProcessed(j,i).handPos = trial(j,i).handPos; 
        end
    end
end

function trialFinal = get_firing_rates(trialProcessed, group, scale_window)
% Gaussian smoothing of binned spikes -> firing rates (Hz)
% group = bin size (ms)
% scale_window = smoothing window (ms)
    trialFinal = struct;
    win    = 10 * (scale_window/group);  % kernel half-width in bins
    normstd = scale_window/group;
    alpha  = (win-1)/(2*normstd);
    temp1  = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2) * (alpha * temp1/((win-1)/2)) .^ 2)';
    gaussian_window = gausstemp / sum(gausstemp);  % normalize
    
    for i = 1:size(trialProcessed,2)       % directions
        for j = 1:size(trialProcessed,1)   % trials
            spikes = trialProcessed(j,i).spikes;  % [neurons x timeBins]
            no_neurons = size(spikes,1);
            smoothed = zeros(size(spikes));
            for nn = 1:no_neurons
                smoothed(nn,:) = conv(spikes(nn,:), gaussian_window, 'same') / (group/1000);
            end
            trialFinal(j,i).rates   = smoothed;
            trialFinal(j,i).handPos = trialProcessed(j,i).handPos;
        end
    end
end

function [prinComp, stats] = doPCA(dataMat)
% Basic PCA on dataMat
% dataMat is [rows x cols]: rows=features, cols=trials
% Output:
%  prinComp  -> the projected data ( same size as dataMat, but in new basis )
%  stats.meanRate -> mean across columns
%  stats.eigVec   -> eigenvectors (in original space)
%  stats.eigVals  -> sorted eigenvalues
%  stats.topPC    -> (optionally) how many PCs we keep
    % 1) subtract mean across columns
    meanRate = mean(dataMat,2);
    dataCT   = dataMat - meanRate;
    % 2) covariance matrix
    C = (dataCT * dataCT') / size(dataCT,2);  % [rows x rows]
    % 3) eig
    [V,D] = eig(C);
    eigVals = diag(D);
    [~, idx] = sort(eigVals, 'descend');
    eigVals  = eigVals(idx);
    V        = V(:, idx);
    % 4) project data
    prinComp = V' * dataCT;  % each col is new coordinate in PC space
    % store
    stats.meanRate = meanRate;
    stats.eigVec   = V;
    stats.eigVals  = eigVals;
    % stats.topPC    = ??? (We can choose later)
end

function [xn, yn, xrs, yrs] = getEqualandSampled(data, noDirections, noTrain, group)
% For each trial/direction, pad trajectory to max length, then downsample
% by 'group' to match binned spike resolution
    % find max trajectory length
    trialHolder = struct2cell(data);
    lengths = [];
    for i = 1:3:(noTrain*noDirections*3)
        % handPos is stored at indices 2, 5, 8, ... in that struct2cell
        % but a simpler approach is to do each direction/trial in a loop
    end
    
    sizes = [];
    for iDir = 1:noDirections
        for jTrial = 1:noTrain
            sizes(end+1) = size(data(jTrial,iDir).handPos,2);
        end
    end
    maxSize = max(sizes);

    % preallocate
    xn = zeros(noTrain, maxSize, noDirections);
    yn = zeros(noTrain, maxSize, noDirections);

    for iDir = 1:noDirections
        for jTrial = 1:noTrain
            hp = data(jTrial,iDir).handPos; % 2 x T
            T  = size(hp,2);
            xn(jTrial,1:T,iDir) = hp(1,:);
            yn(jTrial,1:T,iDir) = hp(2,:);
            % pad the tail with the last known position
            if T < maxSize
                xn(jTrial,T+1:maxSize,iDir) = hp(1,end);
                yn(jTrial,T+1:maxSize,iDir) = hp(2,end);
            end
        end
    end

    % now downsample by factor 'group'
    newLength = floor(maxSize / group);
    xrs = zeros(noTrain, newLength, noDirections);
    yrs = zeros(noTrain, newLength, noDirections);
    
    for iDir = 1:noDirections
        for jTrial = 1:noTrain
            xTemp = xn(jTrial,:,iDir);
            yTemp = yn(jTrial,:,iDir);
            % sample each group points
            xrs(jTrial,:,iDir) = xTemp(1:group:newLength*group);
            yrs(jTrial,:,iDir) = yTemp(1:group:newLength*group);
        end
    end
end
