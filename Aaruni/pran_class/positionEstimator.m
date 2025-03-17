function [x, y] = positionEstimator(test_data, modelParameters)
    % Predict hand position for a single test sample using kNN regression

    % Extract stored training data from modelParameters
    train_data = modelParameters.train_data;
    hand_pos_x_train = modelParameters.hand_pos_x_train;
    hand_pos_y_train = modelParameters.hand_pos_y_train;
    k = modelParameters.k;

    % Extract current test sample's neural activity
    test_sample = mean(test_data.spikes, 2);

    % Perform kNN regression (WITHOUT external function)
    x = knn_regression_manual(train_data, hand_pos_x_train, test_sample, k);
    y = knn_regression_manual(train_data, hand_pos_y_train, test_sample, k);

    function predicted_position = knn_regression_manual(train_data, train_labels, test_sample, k)
        % Compute Euclidean distances between test sample and all training samples
        distances = sum((train_data - test_sample).^2, 1); % Squared Euclidean distance
        [~, sorted_indices] = sort(distances); % Sort distances
    
        % Get the k nearest neighbors
        nearest_neighbors = sorted_indices(1:k);
    
        % Predict position as the mean of k-nearest neighbors' positions
        predicted_position = mean(train_labels(nearest_neighbors));
    end
end
