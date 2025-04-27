%% HELPER FUNCTION TO PERFORM CLASSIFICATION WITH kNN

function output_label = KNN_classifier(directions, test_weight, train_weight, NN_num, pow, alp, method, type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predicts direction label using either hard or soft k-nearest neighbours (kNN).
%
% Inputs:
%   test_weight  - [lda_dim x 1] test sample in LDA space
%   train_weight - [lda_dim x n_samples] training set in LDA space
%   NN_num       - neighbor divisor for k calculation
%   pow          - exponent for inverse-distance weighting (soft)
%   alp          - alpha for exponential soft weighting
%   method       - 'hard' or 'soft'
%   type         - 'dist' (1/distance^pow) or 'exp' (exp(-alpha * d))
%
% Output:
%   output_lbl   - predicted movement direction label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    train_len = size(train_weight, 2) / directions; 
    k = max(1, round(train_len / NN_num)); 

    output_label = zeros(1, size(test_weight, 2));
    
    for i = 1:size(test_weight, 2)
        % For the i-th test sample:
        distances = sum((train_weight - test_weight(:, i)).^2, 1);

        % Sort and get top-k nearest neighbors
        [sorted_dist, sorted_idx] = sort(distances, 'ascend');
        nearest_idx    = sorted_idx(1:k);
        nearest_dist   = sorted_dist(1:k);

        % Convert index -> direction label
        % If train_weight is grouped angle-by-angle, we do this:
        train_labels = ceil(nearest_idx / train_len);  % each from 1..8

        switch method
            case 'soft'

                % Compute distance-based weights, e.g. 1/d^2
                if strcmp(type, 'dist')
                    weights = 1 ./ (nearest_dist.^pow + eps);
                
                elseif strcmp(type, 'exp')
                    weights = exp(-alp .* nearest_dist);
                end
        
                % Sum up weights for each angle
                angle_weights = zeros(1, directions);
                
                for nn = 1:k
                    angle = train_labels(nn); 
                    angle_weights(angle) = angle_weights(angle) + weights(nn);
                end
                
                % % Final predicted label is the angle with the highest sum of weights
                % [~, best_angle] = max(angleWeights);
                % output_label(i) = best_angle;
        
                % Or we can use probability distribution
                p = angle_weights / sum(angle_weights);
                [~, best_angle] = max(p);
                output_label(i) = best_angle;
        
            case 'hard'
                output_label(i) = mode(train_labels);
        
            otherwise
                error("Incorrect kNN method! Choose between 'hard' and 'soft'.");
        end
    end
end