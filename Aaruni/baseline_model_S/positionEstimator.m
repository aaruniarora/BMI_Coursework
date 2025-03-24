function [x,y,modelParameters]= positionEstimator(testData, modelParameters)

    %%%%%% Initialise parameters %%%%%%
    
    start = 320;
    finish = 560;
    group = 20;
    ema_decay = 0.3; % value chosen through multiple test trials
    
    %%%%% 1. Preprocess the trial data %%%%%
    %This step ensures that 
    % the data is in a suitable format for classification and regression.
    
    trialFinal = process_and_smooth(testData, group, ema_decay);    
    % assignin('base', "trialFinal", trialFinal);
    
    %%%%%% 2.  Determine indexer based on T_end %%%%%
    % Determine the indexer, which is used to select the appropriate model 
    % parameters for classification and regression based on the current time point 
    % (T_end) in the trial. This allows for dynamic adaptation as more data becomes 
    % available during a trial.
    
    T_end = size(testData.spikes, 2); %disp('Tend'); disp(T_end);
    indexer = min(max(floor((T_end - start) / group) + 1, 1), length(modelParameters.classify));
    % disp('IDX'); disp(indexer);
    
    %%%%%% 3. Extract firing data for PCA and PCR %%%%%%
    neuraldata = reshape(trialFinal.rates, [], 1);
    % assignin('base', "neuraldata", neuraldata);

    %%%%%% 4. Use KNN to classify direction  %%%%%
    % Classification of movement direction using a KNN classifier. If the current 
    % time point is within the specified range, classification is performed. 
    % Otherwise, the last known direction label is used.
    
    if T_end <= finish
    % Classification is applicable within the time range
     train_weight = modelParameters.classify(indexer).wTrain;
     % assignin('base', 'train_weight', train_weight);

     meanFiringTrain = modelParameters.classify(indexer).mFiring;
     % disp('1.'); size(modelParameters.classify(indexer).wTest')
     % disp('2.'); size(neuraldata)
     % disp('3.'); size(meanFiringTrain)
     test_weight = modelParameters.classify(indexer).wTest' * (neuraldata - meanFiringTrain);
     outLabel = KNN_classifier(test_weight, train_weight, 8);
    else
       
    % Beyond maxTime, use the last known label without re-classification
     outLabel = modelParameters.actualLabel;
    end
    modelParameters.actualLabel = outLabel; % Update the actual label in model parameters
    
    %%%%%%% 5. Estimate position using PCR results for both within and
    %%%%%%% beyond maxTime %%%%%

    % Estimate the hand position using PCR, applicable for both within the specified 
    % time range and beyond. This step uses regression coefficients and average firing 
    % rates to calculate the X and Y coordinates of the hand position.
    
    avX = modelParameters.averages(indexer).avX(:,outLabel);
    avY =  modelParameters.averages(indexer).avY(:,outLabel);
    meanFiring = modelParameters.pcr(outLabel, indexer).fMean;
    bx = modelParameters.pcr(outLabel,indexer).bx;
    by = modelParameters.pcr(outLabel,indexer).by;
    
    x = calculatePosition(neuraldata, meanFiring, bx, avX);
    y = calculatePosition(neuraldata, meanFiring, by, avY);
    %predicted = outLabel; %The predicted direction of movement


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  HELPER FUNCTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function processed_data = process_and_smooth(trial, group, ema_decay)
% Processes trial data by re-binning, sqrt transformation, and smoothing firing rates with 
% an exponential moving average (EMA).
%
% Input: 
%  trial : input struct with spikes and handPos data.
%  group : new binning resolution in ms.
%  ema_decay : decay factor for EMA, controlling the degree of smoothing.

    % Initialize the output structure
    processed_data = struct;
    
    % Loop through each trial and reaching angle
    for i = 1:size(trial, 2)
        for j = 1:size(trial, 1)

            % Re-bin spikes
            spikes_total = trial(j, i).spikes;
            [neurons, points] = size(spikes_total);
            t_new = 1:group:points + 1;
            spikes_new = zeros(neurons, numel(t_new) - 1);

            for k = 1:length(t_new) - 1
                if k == length(t_new) - 1
                    spikes_new(:, k) = sum(spikes_total(:, t_new(k):end), 2); % Ensure we include all remaining data in the last bin
                else
                    spikes_new(:, k) = sum(spikes_total(:, t_new(k):t_new(k + 1) - 1), 2);
                end
            end

            % Apply sqrt transformation 
            spikes_new = sqrt(spikes_new);
            
            % Apply EMA smoothing
            ema_rates = zeros(size(spikes_new));
            for n = 1: neurons
                for t = 2:size(spikes_new, 2)
                    ema_rates(n, t) = ema_decay * spikes_new(n, t) + (1 - ema_decay) * ema_rates(n, t - 1);
                end
            end

            % Store processed data in the output structure
            processed_data(j, i).rates = ema_rates / (group / 1000); % Adjust rates to spikes per second
        end
    end
end

function [output_lbl] = KNN_classifier(test_weight, train_weight, NN_num)

% Input:
%  test_weight: Testing dataset after projection 
%  train_weight: Training dataset after projection
%  NN_num: Used to determine the number of nearest neighbors
 
    trainlen = size(train_weight, 2) / 8; 
    k = max(1, round(trainlen / NN_num)); 

    output_lbl = zeros(1, size(test_weight, 2));

    for i = 1:size(test_weight, 2)
        distances = sum(bsxfun(@minus, train_weight, test_weight(:, i)).^2, 1);
        [~, indices] = sort(distances, 'ascend');
        nearestIndices = indices(1:k);

    
        trainLabels = ceil(nearestIndices / trainlen); 
        modeLabel = mode(trainLabels);
        output_lbl(i) = modeLabel;
    end

end

% Helper function for position calculation

function pos = calculatePosition(neuraldata, meanFiring, b, av)
    pos = (neuraldata(1:length(b)) - mean(meanFiring))' * b + av;
    pos = adjustFinalPosition(pos, T_end);
end

% Adjusting function for the final position

function pos = adjustFinalPosition(pos, T_finish)
    try
        pos = pos(T_finish, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end
end

end
