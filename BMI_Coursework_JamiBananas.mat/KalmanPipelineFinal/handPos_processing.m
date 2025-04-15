function [xPos, yPos, formatted_xPos, formatted_yPos] = handPos_processing(training_data, bins)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pads and extracts hand position data aligned to spike bins
%
% Inputs:
%   training_data - input dataset with handPos field
%   bins          - selected bin indices (e.g., 320:20:560)
%
% Outputs:
%   xPos, yPos          - padded x and y positions [trials x time x dirs]
%   formatted_xPos/yPos - same data but indexed by bin times
%
% Steps:
%   1. Forward and then backward fill NaN values 
%   2. Pads all trials with the last value to max trial time length
%   3. Bins hand positions over `bin_group` intervals
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    handPos_cells = {training_data.handPos};
    max_trajectory = max(cellfun(@(hp) size(hp, 2), handPos_cells));
    clear handPos_cells;

    [rows, cols] = size(training_data);
    
    xPos = zeros(rows, max_trajectory, cols);
    yPos = zeros(rows, max_trajectory, cols);

    % Pad each trial to padLength
    for c = 1:cols
        for r = 1:rows
            % Mean Centre
            curr_x = training_data(r,c).handPos(1,:);
            curr_y = training_data(r,c).handPos(2,:);
            
            % Fill missing values in hand position
            curr_x = fill_nan(curr_x, 'handpos');
            curr_y = fill_nan(curr_y, 'handpos');

            if size(training_data(r,c).handPos,2) < max_trajectory
                pad_size = max_trajectory - size(training_data(r,c).handPos,2);
                if pad_size > 0
                    % Reformat the data by repeating the last element for padding
                    xPos(r, :, c) = [curr_x, repmat(curr_x(end), 1, pad_size)];
                    yPos(r, :, c) = [curr_y, repmat(curr_y(end), 1, pad_size)];
                else
                    % For no padding, just copy the original data
                    xPos(r, :, c) = curr_x;
                    yPos(r, :, c) = curr_y;
                end
            end
        end
    end 
    formatted_xPos = xPos(:, bins, :);
    formatted_yPos = yPos(:, bins, :);
end