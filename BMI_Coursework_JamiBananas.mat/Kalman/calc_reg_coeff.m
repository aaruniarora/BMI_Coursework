%% HELPER FUNCTION FOR PCR

function [reg_coeff_X, reg_coeff_Y, filtered_firing ] = ...
    calc_reg_coeff(win_idx, time_div, labels, dir_idx, ...
    spikes_matrix, pca_thresh, time_interval, curr_X_pos, curr_Y_pos,poly_degree,method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates regression coefficients mapping spikes to hand positions
% using Principal Component Regression (PCR)
%
% Inputs:
%   win_idx       - current time window index
%   time_div      - array of time divided into bins
%   labels        - trial direction labels
%   dir_idx       - current movement direction index
%   spikes_matrix - feature matrix of preprocessed neural firing rates
%   pca_thresh    - PCA variance threshold
%   time_interval - vector of bin centers
%   curr_X_pos    - current matrix of x-coordinates of hand pos for direction
%   curr_Y_pos    - current matrix of y-coordinates of hand pos for direction
%   poly_degree   - polynomial regression order
%   method        - regression type: 'standard', 'ridge', or 'lasso'or
%   'poly'
%
% Outputs:
%   reg_coeff_X/Y - learned regression coefficients for x and y hand
%   positions
%   filtered_firing - centered neural data for this bin and direction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Mean centering the hand positions for the current time window
    centered_X = bsxfun(@minus, curr_X_pos(:, win_idx), mean(curr_X_pos(:, win_idx)));
    centered_Y = bsxfun(@minus, curr_Y_pos(:, win_idx), mean(curr_Y_pos(:, win_idx)));
    
    % Filtering firing data based on time and direction and centering it
    filtered_firing = filter_firing_rate(spikes_matrix, time_div, time_interval(win_idx), labels, dir_idx);
    centered_win_firing = filtered_firing  - mean(filtered_firing ,1);

    % Performing PCA for dimensionality reduction and numerical stability
    [~, score, nPC] = perform_PCA(centered_win_firing, pca_thresh, 'nodebug');
    principal_components = score' * centered_win_firing;  % (n_components x n_samples)
    X = principal_components';                            % Transpose: (n_samples x n_components)

    % Project data onto top principal components for polynomial regression
    Z = score(:, 1:nPC)' * centered_win_firing;  % [pca_dim x samples]

    % Select regression method and calculate X and Y regression coefficients 
    if nargin < 10, method = 'standard'; end
    switch lower(method)
        case 'standard'
            reg_mat = pinv(principal_components * principal_components') * principal_components;
            reg_coeff_X = score * reg_mat * centered_X;
            reg_coeff_Y = score * reg_mat * centered_Y;

        case 'poly'
            % Expand X with polynomial features
            X_poly = [];
            for p = 1:poly_degree
                X_poly = [X_poly; Z.^p];  % Use , to concatenate along columns (features)
            end
            % Compute regression coefficients for centered positions
            reg_coeff_X = (X_poly * X_poly') \ (X_poly * centered_X);
            reg_coeff_Y = (X_poly * X_poly') \ (X_poly * centered_Y);
            modelParameters.poly_1 =  1;

        case 'ridge'
            lambda = 1; % regularisation parameter (to be tuned)
            % Ridge Regression: (X'X + λI)^(-1) X'y
            reg_coeff_X = score * ((X' * X + lambda * eye(size(X, 2))) \ (X' * centered_X));
            reg_coeff_Y = score * ((X' * X + lambda * eye(size(X, 2))) \ (X' * centered_Y));

        case 'lasso'
            lambda = 0.1; % regularisation parameter (to be tuned)
            % Lasso must be done column-wise because it solves for one response variable at a time
            % Bx = lasso(X, centered_X, 'Lambda', lambda); % stats toolbox
            % By = lasso(X, centered_Y, 'Lambda', lambda); % stats toolbox
            
            Bx = zeros(size(X, 2), 1);
            By = zeros(size(X, 2), 1);
            for d = 1:size(centered_X, 2)
                Bx(:, d) = custom_lasso(X, centered_X(:, d), lambda);
                By(:, d) = custom_lasso(X, centered_Y(:, d), lambda);
            end

            reg_coeff_X = score * Bx;
            reg_coeff_Y = score * By;

        otherwise
            error('Unknown regression method: choose standard, ridge, or lasso');
    end
end

function B = custom_lasso(X, y, lambda, max_iter, tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs Lasso regression using coordinate descent
% Inputs:
%   X        - [samples x features] predictor matrix
%   y        - [samples x 1] response vector
%   lambda   - regularisation parameter
%   max_iter - max iterations (default: 1000)
%   tol      - convergence threshold (default: 1e-4)
% Output:
%   B        - regression coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin < 4, max_iter = 1000; end
    if nargin < 5, tol = 1e-4; end

    [~, p] = size(X);
    B = zeros(p, 1);
    % Xy = X' * y;
    X_sq_sum = sum(X.^2);

    for iter = 1:max_iter
        B_old = B;

        for j = 1:p
            % Partial residual
            r_j = y - X * B + X(:, j) * B(j);

            % Update coordinate j
            rho = X(:, j)' * r_j;

            if rho < -lambda / 2
                B(j) = (rho + lambda / 2) / X_sq_sum(j);
            elseif rho > lambda / 2
                B(j) = (rho - lambda / 2) / X_sq_sum(j);
            else
                B(j) = 0;
            end
        end

        % Check for convergence
        if norm(B - B_old, 2) < tol
            break;
        end
    end
end
