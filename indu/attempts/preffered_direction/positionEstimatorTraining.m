clc
clear
load ('monkeydata_training.mat')
% ix = randperm(length(trial));
% trainingData = trial(ix(1:10),:);
% testData = trial(ix(11:end),:);
num = 80;
trainingData = trial(1:num, :);   % First 10 trials for training
testData = trial(num+1:end, :);      % Remaining trials for testing
[modelParameters, firingData] = positionEstimatorTraining(trainingData,testData);

%%

function [modelParameters, firingData] = positionEstimatorTraining(trainingData,testData)
    %% Preprocess Data

    cropped_data = crop_data(trainingData);
    %assignin('base', 'td_temp', cropped_data);

    window = 20;
    no_train = length(trainingData);
    reach_angles = [1/6,7/18,11/18,15/18,19/18,23/18,31/18,35/18].*pi;

    firing_rates = computeFiringRate(cropped_data,window);

    min_firing_rate = 0.1; % Minimum firing rate in Hz to keep neurons
    processed_firing_rates = filterFiringRates(firing_rates,min_firing_rate);

    %assignin('base', 'firing_rate_temp', firing_rates);

    threshold = 45;
    preferred_directions = computePreferredDirection(processed_firing_rates,reach_angles,threshold);

    assignin('base', 'preffered_direction', preferred_directions);

    % Define the threshold (in Hz)
    threshold = 45;  % For example, we ignore firing rates below 5 Hz
    
    % Assume firing_rates is already defined as your cell array containing firing rates
    % for each trial and angle.
    
    % Define number of Gaussians for fitting (for multi-peaked tuning curves)
    num_gaussians = 1;  % For example, we try to fit 2 Gaussians for each neuron
    
    % Compute preferred directions using the Gaussian fitting
    preferred_directions_tune = computePreferredDirections(firing_rates, reach_angles, threshold, num_gaussians);

    assignin('base', 'preffered_direction_tune', preferred_directions_tune);

    trialIndex = 5;
    angleIndex = 2;
    plotTrajectoryComparison(preferred_directions_tune, preferred_directions, testData, trialIndex, angleIndex)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Helper%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function cropped_data = crop_data(trainingData)
        no_trials = size(trainingData, 1);
        no_angles = size(trainingData, 2);
        neuron_no = size(trainingData(1,1).spikes, 1);
        pos_no = 2;
        start = 301;
        end_time = 100;
        cropped_data = struct;

        for i = 1:no_trials
            for j = 1:no_angles
                for x = 1:neuron_no
                    cropped_data(i,j).spikes(x,:) = trainingData(i,j).spikes(x,start:end-end_time);
                end
            end
        end
    
        for i = 1:no_trials
            for j = 1:no_angles
                for x = 1:pos_no
                    cropped_data(i,j).handPos(x,:) = trainingData(i,j).handPos(x,start:end-end_time);
                end
            end
        end
    end

   function firing_rates = computeFiringRate(trial, bin_size)
    % Computes firing rate over bins for spike activity in trial.spikes
    % Handles trials of different lengths.
    %
    % INPUTS:
    % trial: (T x 8) struct, where trial(t, n).spikes is a (neurons × duration) matrix
    % bin_size: Size of the bins in seconds
    %
    % OUTPUT:
    % firing_rates: A {T x 8} cell array, where each cell is a (neurons × time_bins) matrix

    num_trials = size(trial, 1); % Number of trials
    num_angles = size(trial, 2); % Number of reaching angles
    num_neurons = size(trial(1,1).spikes, 1); % Neurons (from first struct entry)
    
    % Initialize cell array to store firing rates
    firing_rates = cell(num_trials, num_angles);
    
    % Loop over trials and reaching angles
    for t = 1:num_trials
        for n = 1:num_angles
            spike_data = trial(t, n).spikes; % Extract neuron × duration spike matrix
            
            if isempty(spike_data)
                firing_rates{t, n} = []; % Handle empty trials
                continue;
            end
            
            trial_duration = size(spike_data, 2); % Duration from number of columns
            bin_edges = 0:bin_size:trial_duration; % Define bin edges
            
            % Ensure at least one bin
            if length(bin_edges) < 2
                bin_edges = [0, trial_duration + bin_size]; % Add an extra bin
            end
            
            % Compute firing rate for each neuron
            num_bins = length(bin_edges) - 1;
            neuron_firing_rates = zeros(num_neurons, num_bins);
            
            for neuron = 1:num_neurons
                % Extract single neuron's spike times
                spike_times = find(spike_data(neuron, :) > 0); % Get indices of spikes
                
                if isempty(spike_times)
                    continue; % Skip if no spikes
                end
                
                % Compute histogram of spikes across bins
                spike_counts = histcounts(spike_times, bin_edges);
                
                % Convert to firing rate (spikes/sec)
                neuron_firing_rates(neuron, :) = (spike_counts / bin_size )*1000;
            end
            
            % Store result in cell array
            firing_rates{t, n} = neuron_firing_rates;
        end
    end
   end

 function processed_firing_rates = filterFiringRates(firing_rates, min_firing_rate)
    % Filters out low-firing neurons at each time bin based on a threshold
    %
    % INPUTS:
    % firing_rates: {T x 8} cell array, each cell is (neurons × time_bins)
    % min_firing_rate: Minimum firing rate (Hz) to keep a neuron in each time bin
    %
    % OUTPUT:
    % processed_firing_rates: {T x 8} cell array with filtered firing rates (neurons below threshold are set to 0)

    num_trials = size(firing_rates, 1);
    num_angles = size(firing_rates, 2);

    % Initialize processed firing rates
    processed_firing_rates = cell(num_trials, num_angles);

    % Apply threshold filtering
    for t = 1:num_trials
        for a = 1:num_angles
            if ~isempty(firing_rates{t, a})
                % Filter firing rates for each neuron and bin
                filtered_rates = firing_rates{t, a};  % Start with the original firing rates
                for neuron = 1:size(filtered_rates, 1)
                    for bin = 1:size(filtered_rates, 2)
                        if filtered_rates(neuron, bin) < min_firing_rate
                            filtered_rates(neuron, bin) = 0; % Set to 0 if below threshold
                        end
                    end
                end
                processed_firing_rates{t, a} = filtered_rates;
            else
                processed_firing_rates{t, a} = []; % Handle empty cells
            end
        end
    end
    end

 function preferred_directions = computePreferredDirection(firing_rates, reach_angle, threshold)
    % Computes the preferred direction(s) for each neuron based on firing rates.
    % A neuron can have multiple preferred directions if its firing rate is similar
    % at more than one reach angle.
    %
    % INPUTS:
    % firing_rates: {T x 8} cell array, each cell is (neurons x time_bins)
    % reach_angles: [1 x N] array of reaching angles (in radians)
    % threshold: Minimum firing rate (in Hz) to include in the calculation
    %
    % OUTPUT:
    % preferred_directions: A cell array where each cell contains a list of preferred
    %                       directions (in radians) for the corresponding neuron.
    
    num_trials = size(firing_rates, 1);
    num_angles = size(firing_rates, 2);
    num_neurons = size(firing_rates{1, 1}, 1); % Number of neurons based on firing rates
    
    % Initialize a cell array to store preferred directions for each neuron
    preferred_directions = cell(num_neurons, 1);

    % Loop over each neuron to compute its preferred directions
    for neuron = 1:num_neurons
        angle_firing_rates = zeros(1, num_angles);  % Store firing rates for each angle
        for a = 1:num_angles
            total_firing_rate = 0;
            
            for t = 1:num_trials
                % Get the firing rate for the current neuron at this trial and angle
                firing_rate_at_angle = sum(firing_rates{t, a}(neuron, :));
                
                % Only include firing rates above the threshold
                if firing_rate_at_angle >= threshold
                    total_firing_rate = total_firing_rate + firing_rate_at_angle;
                end
            end
            angle_firing_rates(a) = total_firing_rate;  % Store the total firing rate for this angle
        end
        
        % Find the maximum firing rate across all angles
        max_firing_rate = max(angle_firing_rates);
        
        % Find all angles where the firing rate is equal to the maximum firing rate
        preferred_angles = reach_angle(angle_firing_rates == max_firing_rate);
        
        % Store the list of preferred directions for the current neuron
        preferred_directions{neuron} = preferred_angles;
    end
end


function preferred_directions = computePreferredDirections(firing_rates, reach_angles, threshold, num_gaussians)
    % Computes the preferred direction for each neuron using multi-Gaussian fitting.
    % Without using any toolbox functions, and includes thresholding.
    %
    % INPUTS:
    % firing_rates: {T x 8} cell array, each cell is (neurons x time_bins)
    % reach_angles: [1 x N] array of reaching angles (in radians)
    % threshold: Minimum firing rate (in Hz) to include in the calculation
    % num_gaussians: Number of Gaussians to fit to the data (for multiple peaks)
    %
    % OUTPUT:
    % preferred_directions: A cell array where each cell contains the preferred
    %                       direction(s) for the corresponding neuron, derived from Gaussian fitting.
    
    num_trials = size(firing_rates, 1);
    num_angles = size(firing_rates, 2);
    num_neurons = size(firing_rates{1, 1}, 1); % Number of neurons based on firing rates
    
    % Initialize a cell array to store preferred directions for each neuron
    preferred_directions = cell(num_neurons, 1);

    % Number of neurons to plot per figure (groups of 10)
    neurons_per_figure = 10;
    figure_count = 1;

    % Loop over each neuron to compute its preferred direction
    for neuron = 1:num_neurons
        firing_rates_at_angles = zeros(1, num_angles);  % Store firing rates for each angle
        
        % Sum firing rates across all trials for each angle
        for a = 1:num_angles
            total_firing_rate = 0;
            for t = 1:num_trials
                % Get the firing rate for the current neuron at this trial and angle
                firing_rate_at_angle = sum(firing_rates{t, a}(neuron, :));
                
                % Only include firing rates above the threshold
                if firing_rate_at_angle >= threshold
                    total_firing_rate = total_firing_rate + firing_rate_at_angle;
                end
            end
            firing_rates_at_angles(a) = total_firing_rate;  % Store the total firing rate for this angle
        end
        
        % Apply thresholding to zero out any angles that are below the threshold
        firing_rates_at_angles(firing_rates_at_angles < threshold) = 0;

        % Now fit the multi-Gaussian model to the firing rates
        % Initialize parameters (Amplitude, Mean, Std Dev for each Gaussian)
        initial_params = [];
        for i = 1:num_gaussians
            initial_params = [initial_params, max(firing_rates_at_angles), reach_angles(round(length(reach_angles)/2)), std(reach_angles)];
        end
        
        % Apply a simple gradient descent or iterative method to optimize parameters
        params = initial_params;
        max_iter = 1000;  % Max number of iterations
        learning_rate = 0.01;  % Learning rate for gradient descent
        tol = 1e-5;  % Convergence tolerance
        
        for iter = 1:max_iter
            % Compute the predicted firing rates using the current parameters
            predicted = sumOfGaussians(params, reach_angles);
            
            % Compute the error (squared error)
            error = predicted - firing_rates_at_angles;
            squared_error = sum(error.^2);
            
            % Check for convergence (if error is small enough)
            if squared_error < tol
                break;
            end
            
            % Compute gradients for each parameter (derivative of squared error w.r.t. params)
            gradients = computeGradients(firing_rates_at_angles, reach_angles, params);
            
            % Update parameters using gradient descent
            params = params - learning_rate * gradients;
        end
        
        % Extract preferred angles (means of the Gaussians)
        preferred_angles = params(2:2:end);  % Extract means of each Gaussian
        
        % Ensure unique preferred angles
        preferred_angles = mod(unique(preferred_angles), 2*pi);
        
        % Store the list of preferred directions for the current neuron
        preferred_directions{neuron} = preferred_angles;
        
        % % --- Plotting in groups of 10 ---
        % % Create a new figure every time we reach a multiple of 10 neurons
        % if mod(neuron-1, neurons_per_figure) == 0
        %     figure;
        % end
        % 
        % % Plot the tuning curve and the fitted Gaussians for the current neuron
        % subplot(neurons_per_figure, 1, mod(neuron-1, neurons_per_figure) + 1);
        % plot(reach_angles, firing_rates_at_angles, 'ko', 'MarkerFaceColor', 'k'); % Plot the actual data
        % hold on;
        % 
        % % Create Gaussian curves based on the fitted parameters
        % x = linspace(min(reach_angles), max(reach_angles), 100);
        % y = zeros(size(x));
        % for i = 1:num_gaussians
        %     gauss = exp(-(x - preferred_angles(i)).^2 / (2 * 1^2)); % Standard deviation = 1 for visualization
        %     y = y + gauss;  % Sum of Gaussians
        % end
        % plot(x, y, 'r--'); % Plot the sum of Gaussians (tuning curve)
        % hold off;
        % 
        % % Plot the preferred directions
        % for i = 1:length(preferred_angles)
        %     line([preferred_angles(i), preferred_angles(i)], ylim, 'Color', 'b', 'LineStyle', '--');
        % end
        % xlabel('Reach Angle (radians)');
        % ylabel('Firing Rate (Hz)');
        % title(['Neuron ' num2str(neuron) ' Tuning Curve and Preferred Directions']);
    end
end

function predicted = sumOfGaussians(params, x)
    % Compute the sum of Gaussians based on the current parameters.
    % Each Gaussian has parameters: amplitude, mean, and std dev.
    
    num_gaussians = length(params) / 3;
    predicted = zeros(size(x));
    
    for i = 1:num_gaussians
        A = params(3*i-2);  % Amplitude
        mu = params(3*i-1);  % Mean
        sigma = params(3*i);  % Standard deviation
        
        predicted = predicted + A * exp(-(x - mu).^2 / (2 * sigma^2));
    end
end

function gradients = computeGradients(firing_rates, angles, params)
    % Compute gradients of the error with respect to each parameter (amplitude, mean, std dev)
    predicted = sumOfGaussians(params, angles);
    error = predicted - firing_rates;
    
    gradients = zeros(size(params));
    
    for i = 1:length(params) / 3
        % For amplitude (A)
        A = params(3*i-2);  % Amplitude
        mu = params(3*i-1);  % Mean
        sigma = params(3*i);  % Standard deviation
        
        % Gradient for amplitude (A)
        grad_A = sum(error .* exp(-(angles - mu).^2 / (2 * sigma^2)));
        gradients(3*i-2) = grad_A;
        
        % Gradient for mean (mu)
        grad_mu = sum(error .* A .* (angles - mu) .* exp(-(angles - mu).^2 / (2 * sigma^2)) / sigma^2);
        gradients(3*i-1) = grad_mu;
        
        % Gradient for standard deviation (sigma)
        grad_sigma = sum(error .* A .* (angles - mu).^2 .* exp(-(angles - mu).^2 / (2 * sigma^2)) / (sigma^3));
        gradients(3*i) = grad_sigma;
    end
end

function plotTrajectoryComparison(preferred_directions_tune, preferred_directions, testData, trialIndex, angleIndex, trainingData)
%PLOTTRAJECTORYCOMPARISON Compares predicted hand trajectories using two sets of preferred directions.
%
% INPUTS:
%   preferred_directions_tune : [98 x 1] or [1 x 98] vector of PDs (from tuning method)
%   preferred_directions      : [98 x 1] or [1 x 98] vector of PDs (from another method)
%   testData                  : (trials x 8) struct array, each cell has .spikes (98 x T) and .handPos (3 x T)
%   trialIndex                : which trial row to use
%   angleIndex                : which of the 8 angles/columns to use
%   trainingData              : (trials x 8) struct array, same format as testData, used to compute the average starting position
%
% OUTPUT:
%   A figure showing the actual (x,y) trajectory vs. two predicted trajectories.

    % 1) Extract the spikes and handPos for the specified trial & angle
    spikes  = testData(trialIndex, angleIndex).spikes;  % 98 x timeBins
    handPos = testData(trialIndex, angleIndex).handPos; % 3 x timeBins
            
    avg_startPos = [-12.65,2.1];  % average of all starting positions

    % 3) We'll ignore z (row 3) and just use x,y for actual trajectory
    actual_xy = handPos(1:2, :);  % 2 x timeBins

    % 4) Predict trajectory using the "tuning" PDs, starting from the average start position
    predicted_xy_tune = predictTrajectory(spikes, preferred_directions_tune, avg_startPos);

    assignin('base', 'tune', predicted_xy_tune);

    % 5) Predict trajectory using the "other" PDs, starting from the average start position
    predicted_xy_other = predictTrajectory(spikes, preferred_directions, avg_startPos);

    assignin('base', 'other', predicted_xy_other);

    % 6) Plot everything on the same figure
    figure('Name','Trajectory Comparison','Color','w');
    hold on;
    plot(actual_xy(1,:), actual_xy(2,:), 'k-', 'LineWidth', 2);        % Actual in black
    plot(predicted_xy_tune(1,:), predicted_xy_tune(2,:), 'r--','LineWidth',2);  % Tuning PD in red
    plot(predicted_xy_other(1,:), predicted_xy_other(2,:), 'b--','LineWidth',2); % Other PD in blue
    legend('Actual','Predicted (Tune)','Predicted (Other)','Location','Best');
    xlabel('X Position'); ylabel('Y Position');
    title(sprintf('Trial %d, Angle %d: Actual vs. Predicted', trialIndex, angleIndex));
    axis equal; grid on;

end

%% ------------------------ SUBFUNCTIONS ------------------------ %%
function predicted_xy = predictTrajectory(spikes, preferred_dirs, initial_position)
%PREDICTTRAJECTORY Predicts (x,y) trajectory given spike counts and PDs.
%
% INPUTS:
%   spikes        : [98 x timeBins], each row is a neuron
%   preferred_dirs: [98 x 1] vector of PDs (radians)
%   initial_position : [2 x 1], the initial (x, y) position to start the prediction
%
% OUTPUT:
%   predicted_xy  : [2 x timeBins], predicted (x,y) over time

    [num_neurons, timeBins] = size(spikes);
    predicted_xy = zeros(2, timeBins);

    % Initialize predicted position at the provided initial position (x0,y0)
    predicted_xy(:,1) = initial_position;

    % We can define a simple "speed" or scaling factor
    % or use the sum of firing rates as a speed estimate
    for t = 2:timeBins
        % 1) Get the spike count for each neuron at time t
        firing_rates = spikes(:, t);

        % 2) Compute the population direction via weighted average
        total_rate = sum(firing_rates);
        if total_rate > 0
            weights = firing_rates / total_rate;  % fraction from each neuron
        else
            weights = zeros(num_neurons,1);
        end

        % For each neuron, pick the first angle
        numeric_PD = zeros(num_neurons, 1);
        for n = 1:num_neurons
            numeric_PD(n) = preferred_dirs{n}(1);  % e.g., the first angle in that cell
        end

        pop_angle = sum(weights .* numeric_PD);  % Weighted average of PDs

        % 3) Convert angle to velocity (dx, dy)
        % For simplicity, let's use 'speed = total_rate * 0.001' or any scale
        speed = total_rate * 0.1;  % tune this factor as needed
        dx = speed * cos(pop_angle);
        dy = speed * sin(pop_angle);

        % 4) Integrate to get new position
        predicted_xy(:, t) = predicted_xy(:, t-1) + [dx; dy];
    end
end

firingData = processed_firing_rates;
modelParameters = 0;
    end

  

