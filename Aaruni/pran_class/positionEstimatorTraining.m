function modelParameters = positionEstimatorTraining(trainingData)
    % Train the model using trainingData and store parameters in modelParameters
    disp(size(trainingData))
    disp(size(trainingData(1, 1).spikes, 1))
    disp(trainingData(1, 1))

    num_neurons = size(trainingData(1, 1).spikes, 1);
    num_trials = size(trainingData, 1);
    num_angles = size(trainingData, 2);
    angle_labels = [30, 70, 110, 150, 190, 230, 310, 350];
    angles = repelem(angle_labels, num_trials);

    pop_activity = zeros(num_neurons, num_trials * num_angles);
    hand_pos_x_train = zeros(num_trials * num_angles, 1);
    hand_pos_y_train = zeros(num_trials * num_angles, 1);
    
    trial_idx = 1;
    for angle_num = 1:num_angles
        for trial_num = 1:num_trials
            % Extract average neural activity
            spike_data = trainingData(trial_num, angle_num).spikes;
            pop_activity(:, trial_idx) = mean(spike_data, 2);

            % Extract final hand position
            hand_pos_x_train(trial_idx) = trainingData(trial_num, angle_num).handPos(1, end);
            hand_pos_y_train(trial_idx) = trainingData(trial_num, angle_num).handPos(2, end);

            trial_idx = trial_idx + 1;
        end
    end

    modelParameters.train_data = pop_activity;
    modelParameters.hand_pos_x_train = hand_pos_x_train;
    modelParameters.hand_pos_y_train = hand_pos_y_train;
    modelParameters.k = 25;
    

end
