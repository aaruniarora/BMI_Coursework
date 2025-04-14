function [W, Z_pca, explained, mu] = myPCA(Z_all, varToKeep)
% myPCA   PCA using an SVD-based approach
%
% USAGE:
%   [W, Z_pca, explained, mu] = myPCA(Z_all, varToKeep)
%
% INPUTS:
%   Z_all     (M x N) matrix, where M = number of features and N = number of samples
%   varToKeep (scalar) e.g. 95 means keep enough PCs to explain >=95% variance
%
% OUTPUTS:
%   W         (M x K) matrix of principal components (sorted by descending singular value)
%   Z_pca     (K x N) data projected into the reduced K-dimensional space
%   explained (1 x K) percentage of variance explained by each of the selected components
%   mu        (M x 1) mean of Z_all (used for centering)
%
% STEPS:
%   1) Center the data by subtracting the mean.
%   2) Compute the singular value decomposition (SVD) of the centered data.
%   3) Compute the variance explained by each singular value.
%   4) Select the minimum number of components required to reach the desired variance.
%   5) Project the data onto the selected principal components.

    % (1) Center the data
    mu = mean(Z_all, 2);
    Zc = Z_all - mu;
    
    % (2) SVD decomposition in 'econ' mode for efficiency
    [U, S, ~] = svd(Zc, 'econ');
    
    % (3) Compute the variance explained by each singular value
    singular_values = diag(S);
    totalVar = sum(singular_values.^2);
    explained_vals = 100 * (singular_values.^2) / totalVar;
    cum_explained = cumsum(explained_vals);
    
    % (4) Find the number of components needed to reach varToKeep percent variance
    K = find(cum_explained >= varToKeep, 1, 'first');
    if isempty(K)
        K = size(Z_all, 1); % fallback in case threshold is never reached
    end
    
    % (5) Select the top-K components and project the data
    W = U(:, 1:K);
    explained = explained_vals(1:K)';
    Z_pca = W' * Zc;
end
