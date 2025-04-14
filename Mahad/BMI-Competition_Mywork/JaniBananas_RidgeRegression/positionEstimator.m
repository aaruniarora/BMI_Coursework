function [x, y, modelParameters] = positionEstimator(past_current_trial, modelParameters)
% POSITIONESTIMATOR
%   Estimates the (x,y) hand position from the most recent neural data
%   using a direction-specific PCA + regression approach (no LDA).
%
% INPUT:
%   past_current_trial - struct with fields:
%       .spikes (neurons x time) up to the current time
%   modelParameters    - struct from training, containing fields:
%       .lowFirers: vector of neuron indices that fire <0.5 Hz (to remove)
%       .direction(dir).timeBin(t).avgFiring   : mean firing for classification
%       .direction(dir).timeBin(t).pcaMean     : mean used for PCA
%       .direction(dir).timeBin(t).eigVec      : PCA eigenvectors
%       .direction(dir).timeBin(t).pcaDim      : #PC retained
%       .direction(dir).timeBin(t).beta_x      : regression weight for x
%       .direction(dir).timeBin(t).beta_y      : regression weight for y
%       .direction(dir).timeBin(t).mean_x      : mean x for re-centering
%       .direction(dir).timeBin(t).mean_y      : mean y for re-centering
%       .currentDirection                      : (updated each time)
%
% OUTPUT:
%   x, y             - current position estimate
%   modelParameters  - updated to store .currentDirection

%% -------------------- Hyperparameters & Preprocessing --------------------
group       = modelParameters.group;     % same bin size from training
win         = modelParameters.smoothWin; % smoothing window from training
lowFirers   = modelParameters.lowFirers; % vector of low firing neuron indices

% Current time in ms
T_end  = size(past_current_trial.spikes, 2);
% Convert T_end to a "time bin index" used in training
startT = modelParameters.startTime;  % e.g. 320
endT   = modelParameters.endTime;    % e.g. 560

% Identify which time bin we should use
%   If T_end < startT, we just use the first bin. If T_end>560, use last bin, etc.
%   Example: indexer = floor((T_end - startT)/group) + 1
if T_end <= startT
    % not enough data to be in the "first" official bin (like code 1 might do)
    indexer = 1;
elseif T_end >= endT
    % past the last bin
    indexer = length(modelParameters.timePoints); 
else
    % within [320, 560], pick the matching bin
    indexer = floor((T_end - startT) / group) + 1;
    if indexer < 1
        indexer = 1;
    end
    if indexer > length(modelParameters.timePoints)
        indexer = length(modelParameters.timePoints);
    end
end

% Bin + sqrt + smooth the test trial (similar to training)
testProcess = bin_and_sqrt(past_current_trial, group, 1);
testFinal   = get_firing_rates(testProcess, group, win);

% testFinal is a struct array, but there's only 1 "trial" and 1 "direction"
% dimension if past_current_trial is a single trial. So we can reference .rates:
firingRatesTest = testFinal.rates;   % [neurons x #bins]
% We only need data up to the bin that corresponds to 'indexer'
if size(firingRatesTest,2) > modelParameters.timePoints(indexer)
    firingRatesTest = firingRatesTest(:,1:modelParameters.timePoints(indexer));
end
% Flatten into a single vector: (neurons * timeBins) x 1
noNeuronsTotal = size(firingRatesTest,1);
noBinsUsed     = size(firingRatesTest,2);
testVector = [];
for b = 1:noBinsUsed
   rowStart = (b-1)*noNeuronsTotal + 1;
   rowEnd   = b*noNeuronsTotal;
   testVector(rowStart:rowEnd,1) = firingRatesTest(:,b);
end

% Remove low-firing neurons
toRemove = [];
for lf = 1:length(lowFirers)
    % Indices in [testVector] that correspond to this neuron
    remIndices = lowFirers(lf) : noNeuronsTotal : length(testVector);
    toRemove   = [toRemove, remIndices];
end
testVector(toRemove,:) = [];

%% ------------------ Determine (or recall) the movement direction ------------------
if T_end <= endT
    % ~~~~~ Classification step: pick direction that best matches "avgFiring" ~~~~~
    % We do a simple "nearest mean" approach across all 8 directions
    bestDir    = 1;
    minDist    = inf;
    for iDir = 1:8
        % Retrieve that direction's average firing for the same bin 'indexer'
        avgFR = modelParameters.direction(iDir).timeBin(indexer).avgFiring; 
        % Euclidean distance
        distVal = norm(testVector - avgFR);
        if distVal < minDist
            minDist = distVal;
            bestDir = iDir;
        end
    end
    % Store the chosen direction
    modelParameters.currentDirection = bestDir;
else
    % If T_end > 560, keep the same direction as previously predicted
    bestDir = modelParameters.currentDirection;
end

%% ------------------ Project onto PCA space for that direction/time bin ------------------
pcaMean    = modelParameters.direction(bestDir).timeBin(indexer).pcaMean;    % vector
eigVec     = modelParameters.direction(bestDir).timeBin(indexer).eigVec;     % columns = PCs
pcaDim     = modelParameters.direction(bestDir).timeBin(indexer).pcaDim;

% Center the test data by subtracting the PCA mean from training
testVectorCtr = testVector - pcaMean;  
% Project onto top pcaDim principal components
testPCscores  = eigVec(:,1:pcaDim)' * testVectorCtr;  % (pcaDim x 1)

%% ------------------ Predict x,y using ridge regression parameters ------------------
beta_x  = modelParameters.direction(bestDir).timeBin(indexer).beta_x;
beta_y  = modelParameters.direction(bestDir).timeBin(indexer).beta_y;
mean_x  = modelParameters.direction(bestDir).timeBin(indexer).mean_x;
mean_y  = modelParameters.direction(bestDir).timeBin(indexer).mean_y;

% The predicted x,y is testPCscores' * beta + offset
x = testPCscores' * beta_x + mean_x;
y = testPCscores' * beta_y + mean_y;

end


%% ---------------- HELPER FUNCTIONS (same as in training) ----------------

function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)
% Bins the spikes into 'group'-ms bins, optionally sqrt-transform
    % If 'trial' is a single structure with .spikes of [neurons x time], 
    % we handle it accordingly:
    if ~isfield(trial,'spikes')
        error('Input "trial" must have a ".spikes" field.');
    end
    all_spikes = trial.spikes;  % [neurons x timePoints]
    no_neurons = size(all_spikes,1);
    no_points  = size(all_spikes,2);
    t_new      = 1:group:(no_points+1);
    binnedSpikes = zeros(no_neurons,numel(t_new)-1);
    for k = 1:(numel(t_new)-1)
        binnedSpikes(:, k) = sum(all_spikes(:, t_new(k):t_new(k+1)-1),2);
    end
    if to_sqrt
        binnedSpikes = sqrt(binnedSpikes);
    end
    trialProcessed.spikes = binnedSpikes;  % store in struct
end

function trialFinal = get_firing_rates(trialProcessed, group, scale_window)
% Gaussian smoothing of binned spikes -> firing rates (Hz)
    if ~isfield(trialProcessed,'spikes')
        error('Input "trialProcessed" must have a ".spikes" field.');
    end
    spikes = trialProcessed.spikes; % [neurons x timeBins]
    no_neurons = size(spikes,1);
    
    win    = 10*(scale_window/group);
    normstd = scale_window/group;
    alpha  = (win-1)/(2*normstd);
    temp1  = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2)*(alpha*temp1/((win-1)/2)).^2)';
    gaussian_window = gausstemp/sum(gausstemp);

    smoothed = zeros(size(spikes));
    for nn = 1:no_neurons
        smoothed(nn,:) = conv(spikes(nn,:), gaussian_window, 'same') / (group/1000);
    end
    trialFinal.rates = smoothed;
end