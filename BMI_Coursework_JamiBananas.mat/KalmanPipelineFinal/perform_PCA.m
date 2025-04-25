function [coeff, score, nPC] = perform_PCA(data, threshold, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs Principal Component Analysis (PCA)
%
% Inputs:
%   data        - neural feature matrix [neurons x trials]
%   threshold   - cumulative variance threshold (e.g., 0.44)
%   debug       - if 'debug', plot score
%
% Outputs:
%   coeff       - principal component coefficients (eigenvectors)
%   score       - projected data in PCA space
%   nPC         - number of PCs meeting variance threshold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % nPC = threshold;
    data_centred = data - mean(data,2);
    % C = cov(data_centred);
    C = data_centred' * data_centred;
    [V, D] = eig(C);
    [d, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    explained_variance = cumsum(d) / sum(d);
    nPC = find(explained_variance >= threshold, 1); % Find the number of PCs that explain at least 44% variance
    score = data_centred * V * diag(1./sqrt(d));
    score = score(:, 1:nPC);
    coeff = V(:, 1:nPC);

    if strcmp(debug, 'debug')
        figure; plot(score);
    end
end