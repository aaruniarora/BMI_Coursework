function [outputs, weights] = perform_LDA(data, score, labels, lda_dim, ...
    training_length, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Performs Linear Discriminant Analysis (LDA) on PCA-transformed data.
% This method is called MDF, most discriminant feature, extraction.
%
% Inputs:
%   data         - original neural data
%   score        - PCA-transformed data
%   labels       - movement direction labels
%   lda_dim      - desired number of LDA components
%   training_len - number of trials per direction
%   debug        - plot if 'debug'
%
% Outputs:
%   outputs      - LDA-transformed features
%   weights      - projection of original data into LDA space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute the LDA projection matrix.
    classes = unique(labels); % gets 1 to 8

    overall_mean = mean(data, 2); % zeros(size(data), length(classes));
    scatter_within = zeros(size(data,1)); % How much do the samples of each class vary around their own class mean?
    scatter_between = zeros(size(data,1)); % How different are the means of the classes from the overall mean?
    
    for i = 1:length(classes)
        % Calculate mean vectors for each direction
        indicies = training_length*(i-1)+1 : i*training_length; % 1, 101, 201.. : 100, 200, 300... 

        % Mean of current direction
        mean_dir = mean(data(:, indicies), 2);

        % Scatter within (current direction)
        deviation_within = data(:, indicies) - mean_dir;
        scatter_within = scatter_within + deviation_within * deviation_within';

        % Scatter between (current direction)
        deviation_between = mean_dir - overall_mean;
        scatter_between = scatter_between + training_length * (deviation_between * deviation_between');
    end
    
    % Reduce the size of the matrices with PCA to improve numerical stability
    project_within = score' * scatter_within * score;  
    project_between = score' * scatter_between * score;
    
    % Sorting eigenvalues and eigenvectors in descending order
    [V_lda, D_lda] = eig(pinv(project_within) * project_between);
    [~, sortIdx] = sort(diag(D_lda), 'descend'); 
    
    % Selects the given lda_dimension eigenvectors to form the final
    % projection (from original feature space to LDA space)
    V_lda = V_lda(:, sortIdx(1:lda_dim));
    outputs = score * V_lda;  % [features x lda_dimension]
    
    % Mapping the mean-centered neural data to the discriminative space
    weights = outputs' * (data - overall_mean);  % [lda_dimension x samples]

    if strcmp(debug, 'debug')
        figure; plot(outputs); title('Output');
        figure; plot(weights); title('Weight');
    end
end