function [spikes_matrix, labels] = extract_features(preprocessed_data, neurons, curr_bin, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Converts 2D spike data into a 2D matrix of features across bins
% In our case, rearranging data as:
% rows: 2744 time points --> 98 neurons x 28 bins
% cols: 800 --> 8 angles and 100 trials so angle 1, trial 1; angle 1, trial 2; ...; angle 8, Trial 100
%
% Inputs:
%   preprocessed_data - output from preprocessing function
%   neurons           - number of neurons before filtering
%   curr_bin          - number of bins to include (time window)
%   debug             - 'debug' enables plotting
%
% Outputs:
%   spikes_matrix     - matrix [neurons*curr_bin x trials]
%   labels            - direction labels for each column (trial)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [rows, cols] = size(preprocessed_data);
    labels = zeros(rows * cols, 1);
    
    for r = 1:rows
        for c = 1:cols
            for k = 1:curr_bin
                c_idx = rows * (c - 1) + r; % 100 (1 - 1) + 1 = 1; 1; 1...x13; 101; 
                r_start = neurons * (k - 1) + 1; % 98 (1 - 1) + 1 = 1; 99; 197;...
                r_end = neurons * k; % 98; 196;...
                spikes_matrix(r_start:r_end,c_idx) = preprocessed_data(r,c).rate(:,k);  
                labels(c_idx) = c; 
            end
        end
    end

    if strcmp(debug, 'debug')
        figure; title(['Firing Rate for Bin ' num2str(curr_bin)]);
        plot(spikes_matrix); 
    end
end