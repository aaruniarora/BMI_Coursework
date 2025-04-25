function preprocessed_data = preprocessing(training_data, bin_group, filter_type, alpha, sigma, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing function for spike trains
%
% Input:
%   training_data - original neural dataset with spikes and hand positions
%   bin_group     - number of ms per time bin (e.g., 20 ms)
%   filter_type   - 'EMA' or 'Gaussian'
%   alpha         - EMA smoothing constant (0 < alpha < 1)
%   sigma         - std deviation for Gaussian smoothing (in ms)
%   debug         - if 'debug', plots are shown
%
% Output:
%   preprocessed_data - struct with binned, smoothed firing rates per trial
%
% Steps:
%   1. Pads all trials to max trial time length
%   2. Bins spike counts over `bin_group` intervals
%   3. Applies square root transform
%   4. Applies either a recursive filter, exponential moving average (EMA),
%      or Gaussian smoothing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialise
    [rows, cols] = size(training_data); 
    preprocessed_data = struct;

    spike_cells = {training_data.spikes};
    max_time_length = max(cellfun(@(sc) size(sc, 2), spike_cells));
    clear spike_cells;

    % Fill NaNs with 0's and pad each trial’s spikes out to max_time_length
    for tl = 1:rows
        for dir = 1:cols
            curr_spikes = training_data(tl, dir).spikes; 
            curr_spikes = fill_nan(curr_spikes, 'spikes');
            [num, T] = size(curr_spikes);
            if T < max_time_length
                padNeeded = max_time_length - T;
                training_data(tl, dir).spikes = [curr_spikes, zeros(num, padNeeded)]; % repmat(curr_spikes(:, end), 1, padNeeded)
            end
        end
    end

    % Bin the spikes by summing counts over non-overlapping windows to get the firing rate
    for c = 1:cols
        for r = 1:rows
            train = training_data(r,c);
            [neurons, timepoints] = size(train.spikes);
            num_bins = floor(timepoints / bin_group); % 28

            binned_spikes = zeros(neurons, num_bins);

            for b = 1:num_bins
                start_time = (b-1)*bin_group + 1; % 1, 21, 41, ..., 541
                end_time = b*bin_group; % 20, 40, 60, ..., 560
                if b == num_bins % gets all the leftover points for the last bin
                    binned_spikes(:,b) = sum(train.spikes(:, start_time:end), 2);
                else
                    binned_spikes(:,b) = sum(train.spikes(:, start_time:end_time), 2);
                end
            end
            
            % Apply sqrt transformation 
            sqrt_spikes = sqrt(binned_spikes);

           % Apply gaussian smoothing
            if strcmp(filter_type, 'Gaussian')
                gKernel = gaussian_filter(bin_group, sigma);
                % Convolve each neuron's spike train with the Gaussian kernel.
                gaussian_spikes = zeros(size(sqrt_spikes));
                for n = 1:neurons
                    gaussian_spikes(n,:) = conv(sqrt_spikes(n,:), gKernel, 'same')/(bin_group/1000);
                end
                preprocessed_data(r,c).rate = gaussian_spikes; % spikes per millisecond
            end

            % Apply EMA smoothing
            if strcmp(filter_type, 'EMA')
                ema_spikes = ema_filter(sqrt_spikes, alpha, neurons);
                preprocessed_data(r,c).rate = ema_spikes / (bin_group/1000); % spikes per second
            end            

        end
    end

    if strcmp(debug, 'debug')
        plot_r = 1; plot_c = 1; plot_n =1;
        figure; sgtitle('After preprocessing');
        subplot(1,2,1); hold on;
        % plot(training_data(plot_r,plot_c).spikes(plot_n,:), DisplayName='Original', LineWidth=1.5); 
        plot(preprocessed_data(plot_r,plot_c).rate(plot_n,:), DisplayName='Preprocessed', LineWidth=1.5);
        xlabel('Bins'); ylabel('Firing Rate (spikes/s)');
        title('Spikes'); legend show; hold off;
    
        subplot(1,2,2); hold on;
        plot(preprocessed_data(plot_r,plot_c).handPos(1,:), preprocessed_data(plot_r,plot_c).handPos(2,:), DisplayName='Original', LineWidth=1.5); 
        xlabel('x pos'); ylabel('y pos');
        title('Hand Positions'); legend show; hold off;
    end
end


function gKernel = gaussian_filter(bin_group, sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates a normalized 1D Gaussian kernel for convolution
% Inputs:
%   bin_group - bin size in ms
%   sigma     - standard deviation of the Gaussian in ms
% Output:
%   gKernel   - 1D vector of Gaussian filter values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Create a 1D Gaussian kernel centered at zero.
    gaussian_window = 10*(sigma/bin_group);
    e_std = sigma/bin_group;
    alpha = (gaussian_window-1)/(2*e_std);

    time_window = -(gaussian_window-1)/2:(gaussian_window-1)/2;
    gKernel = exp((-1/2) * (alpha * time_window/((gaussian_window-1)/2)).^2)';
    gKernel = gKernel / sum(gKernel);
end


function ema_spikes = ema_filter(sqrt_spikes, alpha, num_neurons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Applies exponential moving average (EMA) smoothing to spike data
% Inputs:
%   sqrt_spikes  - sqrt-transformed spike matrix [neurons x time bins]
%   alpha        - smoothing factor (higher = more recent weight)
%   num_neurons  - number of input neurons
% Output:
%   ema_spikes   - smoothed spike matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:size(sqrt_spikes, 2)
            ema_spikes(n, t) = alpha * sqrt_spikes(n, t) + (1 - alpha) * ema_spikes(n, t - 1);
        end
    end
end


function data = fill_nan(data, data_type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fills NaN values in spike or hand position data
% For spikes the NaN values are replaced with 0's and for hand position
% data we perform a forward then a backward fill.
% Inputs:
%   data       - input vector/matrix
%   data_type  - 'spikes' or 'handpos'
% Output:
%   data       - cleaned data with NaNs filled
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if strcmp(data_type, 'spikes')
        data(isnan(data)) = 0;
    end
    
    if strcmp(data_type, 'handpos')
        % Forward fill
        for r = 2:length(data)
            if isnan(data(r))
                data(r) = data(r-1);
            end
        end
        % Backward fill for any remaining NaNs
        for r = length(data)-1:-1:1
            if isnan(data(r))
                data(r) = data(r+1);
            end
        end
    end
end