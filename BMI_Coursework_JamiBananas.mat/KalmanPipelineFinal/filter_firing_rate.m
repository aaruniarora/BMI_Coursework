function filtered_firing = filter_firing_rate(spikes_matrix, time_div, time_interval, labels, dir_idx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filters spike data by time and direction, centers trials
% Inputs:
%   spikes_matrix  - full feature matrix of preprocessed neural firing 
%                    rates [neurons*time x trials]
%   time_div       - time bin identifier for each row
%   time_interval  - current time bin (upto which the data should be
%                    filtered)
%   labels         - direction labels for each trial (column vector)
%   dir_idx        - target direction
% Output:
%   filtered_firing - centered data for this time and direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Trim the spikes data upto the time point of the time_interval
    trimmed_time = time_div <= time_interval;
    % Then filter the data for a particular direction 
    dir_filter = labels == dir_idx;
    filtered_firing  = spikes_matrix(trimmed_time, :);
    % Mean centre
    filtered_firing  = filtered_firing (:, dir_filter) - mean(filtered_firing(:, dir_filter), 1);
end
