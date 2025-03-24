function preprocessAndTrainTruncate_Visualize_Gauss()
    % Load and preprocess data, train a linear regression model,
    % compute RMSE, and visualize actual vs. predicted trajectories for each trial.
    
    % Preprocess data using truncation with Gaussian smoothing instead of Savitzky–Golay
    [X_all, Y_all, startHandPos, minTimeBins, X_cell, Y_cell] = preprocessDataTruncate_Gauss();
    
    % Train linear regression model
    % Solve: W = (X_all'*X_all) \ (X_all'*Y_all)
    W = (X_all' * X_all) \ (X_all' * Y_all);
    
    % Predict on the global training data for RMSE computation
    Y_pred = X_all * W;
    mse_val = mean(sum((Y_all - Y_pred).^2, 2));
    rmse_val = sqrt(mse_val);
    fprintf('Preprocessing complete. Each trial truncated to %d time bins.\n', minTimeBins);
    fprintf('RMSE on training data (truncation with Gaussian smoothing): %.3f\n', rmse_val);
    
    % Visualize actual vs. predicted trajectories for each trial
    figure;
    hold on;
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

function [X_all, Y_all, startHandPos, minTimeBins, X_cell, Y_cell] = preprocessDataTruncate_Gauss()
    % This function loads monkeydata_training.mat and applies the following
    % preprocessing steps:
    %   1. Bandpass filtering (0.5–100 Hz) to each neuron’s spike train.
    %   2. Gaussian smoothing applied to the filtered spike train.
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
    
    % Data dimensions: trial is 100 x 8
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
    
    % Set parameters for the Gaussian filter:
    gaussWindow = 11;     % number of samples (must be odd)
    gaussSigma = 2;       % standard deviation in samples
    % Create a Gaussian kernel (column vector), then normalize to sum to 1.
    gaussKernel = gausswin(gaussWindow);
    gaussKernel = gaussKernel / sum(gaussKernel);
    
    % Preallocate cell arrays for trial data
    X_cell = {};
    Y_cell = {};
    startHandPos = {};
    
    % Process each trial (across angles and trial numbers)
    for angleIdx = 1:numAngles
        for tr = 1:numTrials
            spikes = trial(tr, angleIdx).spikes;    % (numNeurons x T)
            handPos = trial(tr, angleIdx).handPos;      % (3 x T)
            T = size(spikes, 2);
            
            % Preallocate processed spikes matrix
            spikes_processed = zeros(size(spikes));
            for neuron = 1:numNeurons
                spk = double(spikes(neuron, :));
                % 1. Apply bandpass filter
                spk_filt = filtfilt(b, a, spk);
                % 2. Apply Gaussian smoothing via convolution
                spk_smooth = conv(spk_filt, gaussKernel, 'same');
                spikes_processed(neuron, :) = spk_smooth;
            end
            
            % Truncate the trial's data to the minimum number of time bins.
            spikes_truncated = spikes_processed(:, 1:minTimeBins);
            handPos_truncated  = handPos(:, 1:minTimeBins);
            
            % Build design matrix for this trial: each row is one time bin.
            X_trial = spikes_truncated';  % (minTimeBins x numNeurons)
            Y_trial = [handPos_truncated(1,:)', handPos_truncated(2,:)'];  % (minTimeBins x 2)
            
            % Compute the starting hand position as the average over the first 300 ms.
            numSamplesForAvg = min(300, minTimeBins);
            startPos = mean(handPos_truncated(1:2, 1:numSamplesForAvg), 2);  % [x; y]
            
            % Save the processed data
            X_cell{end+1} = X_trial;
            Y_cell{end+1} = Y_trial;
            startHandPos{end+1} = startPos;
        end
    end
    
    % Concatenate data from all trials to form global matrices
    X_all = vertcat(X_cell{:});
    Y_all = vertcat(Y_cell{:});
    
    fprintf('Preprocessing complete. Each trial truncated to %d time bins.\n', minTimeBins);
end
