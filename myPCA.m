function [W, Z_pca, explained, mu] = myPCA(Z_all, varToKeep)
% myPCA   Manual PCA without using toolboxes
% 
% USAGE:
%   [W, Z_pca, explained, mu] = myPCA(Z_all, varToKeep)
% 
% INPUTS:
%   Z_all     (M x N) = M features, N samples
%   varToKeep (scalar) e.g. 95 means keep enough PCs for >=95% variance
% 
% OUTPUTS:
%   W         (M x K) principal components (sorted by descending eigenvalue)
%   Z_pca     (K x N) data in the reduced dimension
%   explained (1 x K) % variance explained by each component
%   mu        (M x 1) row-wise mean of Z_all
%
% STEPS:
%   1) center data => mu
%   2) covariance => MxM
%   3) eigen decomp => sort descending
%   4) pick top-K for var coverage
%   5) project => Z_pca

    % (1) center data
    mu = mean(Z_all, 2);         % M x 1
    Zc = Z_all - mu;             % subtract mean from each column

    % (2) covariance
    [M, N] = size(Zc);
    CovMat = (Zc * Zc') / N;     % M x M

    % (3) eigen decomp
    [V, D] = eig(CovMat);
    eigVals = diag(D);           % M x 1
    [eigValsSorted, idx] = sort(eigVals, 'descend');
    V_sorted = V(:, idx);

    % (4) find how many PCs to keep
    totalVar = sum(eigValsSorted);
    cvar = 100 * cumsum(eigValsSorted) / totalVar; 
    K = find(cvar >= varToKeep, 1, 'first');
    if isempty(K)
        K = M; % fallback
    end

    % pick top-K
    W = V_sorted(:, 1:K);                 % M x K
    usedVals = eigValsSorted(1:K);
    explained = 100*(usedVals / totalVar)';  % fraction -> percentage

    % (5) project => Z_pca = W' * Zc
    Z_pca = W' * Zc;
end
