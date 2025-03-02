clc
clear
load ('monkeydata_training.mat')
% ix = randperm(length(trial));
% trainingData = trial(ix(1:10),:);
% testData = trial(ix(11:end),:);
num = 80;
trainingData = trial(1:num, :);   % First 10 trials for training
testData = trial(num+1:end, :);      % Remaining trials for testing
[modelParameters, firingData] = train(trainingData);

%%

function [modelParameters, firingData] = train(trainingData)
    %% Preprocess Data
    cropped_data = crop_data(trainingData);
    %assignin('base', 'td_temp', cropped_data);

    group = 20;
    window = 20;
    no_train = length(trainingData);
    reach_angles = [1/6,7/18,11/18,15/18,19/18,23/18,31/18,35/18].*pi;

    firing_rates = computeFiringRate(cropped_data,window);

    assignin('base', 'firing_rates', firing_rates);

    min_firing_rate = 1; % Minimum firing rate in Hz to keep neurons
    processed_firing_rates = filterFiringRates(firing_rates,min_firing_rate);

    coeff = pca(processed_firing_rates);

    firingData = processed_firing_rates;
    modelParameters = 0;


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
    % Applies Gaussian smoothing to the computed firing rates.
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
    
    % Define Gaussian smoothing kernel
    sigma = bin_size / 2; % Standard deviation of Gaussian window
    window_size = round(5 * sigma); % Define window size (5 times sigma)
    x = -window_size:window_size;
    gauss_kernel = exp(-(x .^ 2) / (2 * sigma ^ 2));
    gauss_kernel = gauss_kernel / sum(gauss_kernel); % Normalize kernel
    
    % Determine the minimum number of bins across all trials
    min_bins = Inf; % Start with a very large number
    for t = 1:num_trials
        for n = 1:num_angles
            spike_data = trial(t, n).spikes; % Extract neuron × duration spike matrix
            
            if isempty(spike_data)
                continue; % Skip empty trials
            end
            
            trial_duration = size(spike_data, 2); % Duration from number of columns
            num_bins = ceil(trial_duration / bin_size); % Compute number of bins
            min_bins = min(min_bins, num_bins); % Update minimum number of bins
        end
    end
    
    % Loop over trials and reaching angles
    for t = 1:num_trials
        for n = 1:num_angles
            spike_data = trial(t, n).spikes; % Extract neuron × duration spike matrix
            
            if isempty(spike_data)
                firing_rates{t, n} = zeros(num_neurons, min_bins); % Empty trial case
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
            neuron_firing_rates = zeros(num_neurons, min_bins); % Initialize for min bins
            
            for neuron = 1:num_neurons
                % Extract single neuron's spike times
                spike_times = find(spike_data(neuron, :) > 0); % Get indices of spikes
                
                if isempty(spike_times)
                    continue; % Skip if no spikes
                end
                
                % Compute histogram of spikes across bins
                spike_counts = histcounts(spike_times, bin_edges);
                
                % Convert to firing rate (spikes/sec)
                firing_rate = (spike_counts / bin_size) * 1000;
                
                % Apply Gaussian smoothing
                smoothed_firing_rate = conv(firing_rate, gauss_kernel, 'same');
                
                % Ensure the firing rate does not exceed the actual size of smoothed_firing_rate
                valid_bins = min(min_bins, length(smoothed_firing_rate)); % Ensure the index is within bounds
                neuron_firing_rates(neuron, 1:valid_bins) = smoothed_firing_rate(1:valid_bins); % Truncate if necessary
            end
            
            % Store the firing rates for this trial and angle
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

    function [X_data,Y_data] = get_feature(processed_firing_rates,cropped_data)

       % Initialize matrices for spike trains and hand trajectories
        X_data = [];
        Y_data = [];
        
        for angle = 1:num_angles
            for t = 1:num_trials
        % Extract spike train and corresponding hand positions
        spikes = trial2(t, angle).spikes;   % 98 x T binary matrix
        handPos = cropped_data(t, angle).handPos; % 3 x T position matrix (X, Y, Z)

        % Truncate to match T_min
        spikes = spikes(:, 1:T_min);
        handPos = handPos(1:2, 1:T_min); % Only take X and Y

        % Flatten spike train into a feature vector
        spike_vector = reshape(spikes, [], 1)'; % (98*T_min) x 1

        % Store the data
        X_data = [X_data; spike_vector]; % Feature matrix
        Y_data = [Y_data; reshape(handPos, 1, [])]; % Flatten trajectory
    end
end




    end

  

