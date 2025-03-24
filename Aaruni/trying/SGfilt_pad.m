function preprocessAndTrain()
    % This function loads the monkey data, applies preprocessing:
    %   - Bandpass filtering (0.5-100 Hz)
    %   - Savitzky–Golay smoothing
    %   - Averages hand positions up until 300 ms for the starting point
    %   - Data padding (all trials padded to the same length)
    %
    % Then it trains a simple linear regression model and prints the RMSE.
    
    % Preprocess data to obtain design matrix (X_all) and target (Y_all)
    [X_all, Y_all, startHandPos] = preprocessData();
    
    % Train linear regression model (without an intercept term)
    % Solve W = (X_all'*X_all) \ (X_all'*Y_all) so that X_all*W approximates Y_all.
    W = (X_all' * X_all) \ (X_all' * Y_all);
    
    % Predict hand positions using the trained model
    Y_pred = X_all * W;
    
    % Compute RMSE (Root Mean Squared Error)
    % Calculate error for each sample (each row has two values: x and y)
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    
    fprintf('RMSE on training data: %.3f\n', rmse_val);
end

function [X_all, Y_all, startHandPos] = preprocessData()
    % This function loads monkeydata_training.mat and applies the following
    % preprocessing steps:
    %   1. Bandpass filtering (0.5–100 Hz) to each neuron’s spike train.
    %   2. Savitzky–Golay smoothing to each filtered spike train.
    %   3. Averages the hand positions (x,y) up until 300 ms for the start position.
    %   4. Pads each trial’s data with zeros so that all trials have equal length.
    %
    % Outputs:
    %   X_all       - Global design matrix (each row is one time bin from a trial)
    %   Y_all       - Global target matrix ([x, y] positions for each time bin)
    %   startHandPos- Cell array containing the averaged starting hand position for each trial.
    
    % Load the training data file (ensure monkeydata_training.mat is in your path)
    load('monkeydata_training.mat');  % Variable "trial" is expected.
    
    % Data dimensions (trial is 100 x 8)
    numTrials = size(trial, 1);   % e.g., 100
    numAngles  = size(trial, 2);   % e.g., 8
    numNeurons = size(trial(1,1).spikes, 1);  % e.g., 98
    fs = 1000;  % Sampling frequency (1 ms bins)
    
    % Determine the maximum number of time bins across all trials for padding.
    maxTimeBins = 0;
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            T = size(trial(tr, angleIdx).spikes, 2);
            maxTimeBins = max(maxTimeBins, T);
        end
    end
    
    % Design a 2nd order Butterworth bandpass filter (0.5-100 Hz)
    [b, a] = butter(2, [0.5, 100]/(fs/2), 'bandpass');
    
    % Parameters for the Savitzky–Golay filter (e.g., polynomial order 2, window length 11 samples)
    sgolayOrder = 2;
    sgolayWindow = 11;
    
    % Initialize cell arrays to hold preprocessed data from each trial.
    X_cell = {};  % Each cell: (time bins x numNeurons) preprocessed spike data
    Y_cell = {};  % Each cell: (time bins x 2) corresponding hand positions [x,y]
    startHandPos = {};  % Each cell: starting hand position computed as the average over first 300 ms.
    
    % Process each trial (across angles and trial numbers)
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            % Extract spike data and hand positions for the current trial.
            spikes = trial(tr, angleIdx).spikes;   % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos;     % (3 x T)
            T = size(spikes, 2);
            
            % Preallocate matrix for processed spikes.
            spikes_processed = zeros(size(spikes));
            
            % Process each neuron's spike train:
            for neuron = 1:numNeurons
                spk = double(spikes(neuron, :));
                % 1. Bandpass filter the spike train.
                spk_filt = filtfilt(b, a, spk);
                % 2. Apply Savitzky–Golay smoothing.
                spk_smooth = sgolayfilt(spk_filt, sgolayOrder, sgolayWindow);
                spikes_processed(neuron, :) = spk_smooth;
            end
            
            % Data Padding: pad the trial's data with zeros to reach maxTimeBins.
            if T < maxTimeBins
                padAmount = maxTimeBins - T;
                spikes_padded = [spikes_processed, zeros(numNeurons, padAmount)];
                handPos_padded  = [handPos, zeros(3, padAmount)];
            else
                spikes_padded = spikes_processed;
                handPos_padded  = handPos;
            end
            
            % Build design matrix for this trial: each row is a time bin.
            X_trial = spikes_padded';  % (maxTimeBins x numNeurons)
            % Target matrix: use only the x and y components.
            Y_trial = [handPos_padded(1,:)', handPos_padded(2,:)'];  % (maxTimeBins x 2)
            
            % Compute the starting hand position as the average over the first 300 ms.
            numSamplesForAvg = min(300, size(handPos_padded, 2));
            startPos = mean(handPos_padded(1:2, 1:numSamplesForAvg), 2);  % [x; y]
            
            % Save the processed data.
            X_cell{end+1} = X_trial;
            Y_cell{end+1} = Y_trial;
            startHandPos{end+1} = startPos;
        end
    end
    
    % Concatenate data from all trials.
    X_all = vertcat(X_cell{:});
    Y_all = vertcat(Y_cell{:});
    
    fprintf('Preprocessing complete. Each trial padded to %d time bins.\n', maxTimeBins);
end
