function preprocessAndTrainTruncate_Visualize()
    % Load and preprocess data, train a linear regression model,
    % compute RMSE, and visualize actual vs. predicted trajectories for each trial.
    
    % Preprocess data using truncation
    [X_all, Y_all, startHandPos, minTimeBins, X_cell, Y_cell] = preprocessDataTruncate();
    
    % Train linear regression model
    % Solve: W = (X_all'*X_all) \ (X_all'*Y_all)
    W = (X_all' * X_all) \ (X_all' * Y_all);
    
    % Predict on the global training data for RMSE computation
    Y_pred = X_all * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    fprintf('Preprocessing complete. Each trial truncated to %d time bins.\n', minTimeBins);
    fprintf('RMSE on training data (truncation): %.3f\n', rmse_val);
    
    % Visualize actual vs. predicted trajectories for each trial
    figure;
    hold on;
    % Loop over each trial in the cell arrays (each cell is one trial)
    for i = 1:length(X_cell)
        % Compute prediction for the current trial
        Y_pred_trial = X_cell{i} * W;
        % Plot actual trajectory in blue
        plot(Y_cell{i}(:,1), Y_cell{i}(:,2), 'b', 'LineWidth', 1);
        % Plot predicted trajectory in red dashed line
        plot(Y_pred_trial(:,1), Y_pred_trial(:,2), 'r--', 'LineWidth', 1);
    end
    xlabel('X position (mm)');
    ylabel('Y position (mm)');
    title('Actual (blue) vs. Predicted (red dashed) Hand Trajectories');
    axis equal;
    grid on;
    hold off;
end

function [X_all, Y_all, startHandPos, minTimeBins, X_cell, Y_cell] = preprocessDataTruncate()
    % This function loads monkeydata_training.mat and applies:
    %   1. Bandpass filtering (0.5–100 Hz) to each neuron’s spike train.
    %   2. Savitzky–Golay smoothing.
    %   3. Computes the average starting hand position (first 300 ms).
    %   4. Truncates each trial to the minimum trial length.
    %
    % Outputs:
    %   X_all       - Global design matrix (each row is one time bin from a trial)
    %   Y_all       - Global target matrix ([x,y] positions per time bin)
    %   startHandPos- Cell array of averaged starting hand positions (per trial)
    %   minTimeBins - Minimum number of time bins across all trials
    %   X_cell      - Cell array of design matrices (one per trial)
    %   Y_cell      - Cell array of target matrices (one per trial)
    
    load('monkeydata_training.mat');  % Loads variable "trial"
    
    % Get dimensions: trial is 100 x 8
    numTrials = size(trial, 1);   
    numAngles  = size(trial, 2);  
    numNeurons = size(trial(1,1).spikes, 1);  
    fs = 1000;  % Sampling frequency (1 ms bins)
    
    % Determine the minimum number of time bins over all trials
    minTimeBins = inf;
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            T = size(trial(tr, angleIdx).spikes, 2);
            minTimeBins = min(minTimeBins, T);
        end
    end
    
    % Design a 2nd-order Butterworth bandpass filter (0.5-100 Hz)
    [b, a] = butter(2, [0.5, 100]/(fs/2), 'bandpass');
    
    % Parameters for Savitzky–Golay filter (2nd order, window length 11 samples)
    sgolayOrder = 2;
    sgolayWindow = 11;
    
    % Preallocate cell arrays for trial data
    X_cell = {};
    Y_cell = {};
    startHandPos = {};
    
    % Loop over all trials (across angles and trial numbers)
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            spikes = trial(tr, angleIdx).spikes;    % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos;      % (3 x T)
            T = size(spikes, 2);
            
            % Preallocate processed spikes
            spikes_processed = zeros(size(spikes));
            for neuron = 1:numNeurons
                spk = double(spikes(neuron, :));
                spk_filt = filtfilt(b, a, spk);            % Bandpass filtering
                spk_smooth = sgolayfilt(spk_filt, sgolayOrder, sgolayWindow);  % Smoothing
                spikes_processed(neuron, :) = spk_smooth;
            end
            
            % Truncate data to minTimeBins
            spikes_truncated = spikes_processed(:, 1:minTimeBins);
            handPos_truncated  = handPos(:, 1:minTimeBins);
            
            % Build design matrix (each row is one time bin)
            X_trial = spikes_truncated';  % (minTimeBins x numNeurons)
            Y_trial = [handPos_truncated(1,:)', handPos_truncated(2,:)'];  % (minTimeBins x 2)
            
            % Compute starting hand position: average over first 300 ms (or fewer)
            numSamplesForAvg = min(300, minTimeBins);
            startPos = mean(handPos_truncated(1:2, 1:numSamplesForAvg), 2);  % [x; y]
            
            % Save data into cell arrays
            X_cell{end+1} = X_trial;
            Y_cell{end+1} = Y_trial;
            startHandPos{end+1} = startPos;
        end
    end
    
    % Concatenate all trials to form global design and target matrices
    X_all = vertcat(X_cell{:});
    Y_all = vertcat(Y_cell{:});
    
    fprintf('Preprocessing complete. Each trial truncated to %d time bins.\n', minTimeBins);
end
