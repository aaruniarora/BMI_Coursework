%% HELPER FUNCTION FOR POSITION CALCULATION

function pos = position_calc(spikes_matrix, firing_mean, b, avg, curr_bin,reg_meth,polyd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates hand position using linear regression from spike activity
%
% Inputs:
%   spikes_matrix - spike vector (after preprocessing and reshaping)
%   firing_mean   - mean firing vector used to center the data
%   b             - regression coefficients (from PCA-reduced space)
%   avg           - average hand trajectory for the direction
%   curr_bin      - current time step
%   reg_meth      - regression method specified (standard, poly, ridge,
%   lasso)
%   polyd         - polynomial regression order
% Output:
%   pos           - estimated x or y position
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(reg_meth, 'poly')
    firingVector = spikes_matrix(1:length(b));  
    
    % Expand features with polynomial terms but maintain 70 features
    polyFiringVector = zeros(size(firingVector)); % Initialize same size as firingVector
    for d = 1:polyd
        polyFiringVector = polyFiringVector + (firingVector - mean(firing_mean)).^d;
    end

    % Predict position using polynomial regression
    pos = polyFiringVector' * b + avg;
    else
    pos = (spikes_matrix(1:length(b)) - mean(firing_mean))' * b + avg;
    end

    try
        pos = pos(curr_bin, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end
end
