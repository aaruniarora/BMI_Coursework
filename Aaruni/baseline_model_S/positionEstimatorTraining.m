function  modelParameters = positionEstimatorTraining(trainingData)

% clc; close all; clear;
% load('monkeydata_training.mat');
% trainingData = trial;
% rng(2013);

%%%%%% Initialise parameters %%%%%%%
directions = 8;
traininglen =  length(trainingData);
group = 20;
ema_decay = 0.35; %arbitrary value decided through multiple trials

%%%%%% 1. Preprocess data  %%%%%%
trialFinal = process_and_smooth(trainingData, group, ema_decay); % 98 x 33

%%%%%% 2. Rearrange Data for Analysis %%%%%%
modelParameters = struct;
start = 320;
finish = 560;

timepoints = [start:group:finish]/group; % 16 to 28
neuron_num = size(trialFinal(1,1).rates,1);

% arranging as:
% rows: 2744 time points --> 98 neurons x 28 bins
% cols: 800 --> 8 angles and 100 trials so angle 1, trial 1; angle 1, trial 2; ...; angle 8, Trial 100
count = 1; % goes till 13 --> 13 bins
for time = timepoints % 16, 17, ...28
    % disp(count);

    for i = 1: traininglen
        for j = 1: directions
            for k = 1: time % runs till 16, then 17, then 18 ... 28
                colIndex = traininglen * (j - 1) + i; % 100 (1 - 1) + 1 = 1; 1; 1...x13; 101; 
                % disp(['colIndex: ' num2str(colIndex)]);
                rowIndexStart = neuron_num * (k - 1) + 1; % 98 (1 - 1) + 1 = 1; 99; 197;...
                rowIndexEnd = neuron_num * k; % 98; 196;...
                neuraldata(rowIndexStart:rowIndexEnd,colIndex) = trialFinal(i,j).rates(:,k);  
            end
        end
    end
    % figure;
    % plot(neuraldata);

%%%%%% 3. Most Discriminant Feature Analysis : PCA + LDA %%%%%%

    pca_dimension = 40; % arbitrary value decided throughout multiple trials 
    lda_dimension = 6;  % arbitrary value decided throughout multiple trials 
    [output, weight] = MDF(neuraldata, directions, traininglen, pca_dimension, lda_dimension);
    % figure; plot(output); title('Output');
    % figure; plot(weight); title('Weight');

%%%%%% 4. KNN Classifier %%%%%%

% Setting up a k-Nearest Neighbors (KNN) classifier for classifying neural data into
% predicted movement directions based on the processed features. The KNN classifier uses weights and other parameters derived
% from Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) to classify neural signals.

   % 'wTrain' holds the weights for the training data. These weights are derived from the feature extraction and selection
    % process, which may involve PCA, LDA, or both.

    modelParameters.classify(count).wTrain = weight;

     % 'wTest' stores the classifier's output or predictions. This might be used for validation or testing purposes, allowing
    % comparison between the predicted labels and the true labels of the data.

    modelParameters.classify(count).wTest= output;

    % 'dPCA_kNN' and 'dLDA_kNN' store the dimensions used for PCA and LDA, respectively. These parameters indicate how many
    % principal components and discriminants were retained in the feature reduction process, impacting the classifier's input space.
    modelParameters.classify(count).dPCA_kNN = pca_dimension;
    modelParameters.classify(count).dLDA_kNN = lda_dimension;

     % 'mFiring' records the mean firing rate across all neurons in the dataset.
    modelParameters.classify(count).mFiring = mean(neuraldata,2);

    % Increment 'count' to move to the next set of parameters or the next fold. This is essential for storing parameters
    % across multiple iterations of training and validation, especially in cross-validation scenarios.

    count = count+1;

end

%%%%%% 5. PCR to get hand positions %%%%%%

% The goal of using Principal Component Regression (PCR) here is to estimate hand positions from neural data. PCR allows us to
% deal effectively with the high dimensionality of neural firing rate data.
% By first applying Principal Component Analysis (PCA) to reduce the dimensionality of the data, we focus on the most significant
% variance components. Then, we use these principal components in a regression model to predict hand positions.


% Create supervised labels for each direction, which will be used in the PCR model. 
labels = repmat(1:directions, traininglen, 1); % Repeat each direction label 'traininglen' times.
labels = labels(:)'; 

% Initialize and resample hand position matrices to ensure consistency in input dimensions.
% This function returns both the original formatted hand positions and their resampled versions for specified time intervals.

[xTestingInterval, yTestingInterval, formattedX, formattedY] = initializeHandPositionMatrices(trainingData, traininglen, directions, group, start, finish);
% disp('X'); size(formattedX)
% disp('Y'); size(formattedY)
% Generate a time division array using the Kronecker product to replicate the time interval for each neuron.

timeDivision = kron(group:group:finish, ones(1, neuron_num)); 
Interval = start:group:finish;

% Loop through each direction to model hand positions separately for each.

for directionIndex = 1: directions

    % Extract the current direction's hand position data for all trials.

    currentXPositions = xTestingInterval(:,:,directionIndex);
    currentYPositions = yTestingInterval(:,:,directionIndex);

 % Loop through each time window to calculate regression coefficients.
    % These coefficients are used to predict hand positions from neural data.

    for timeWindowIndex = 1:((finish-start)/group)+1

         % Calculate regression coefficients and the windowed firing rates for the current time window and direction.

        [regressionCoefficientsX, regressionCoefficientsY, windowedFiring] = calcRegressionCoefficients(timeWindowIndex, timeDivision, labels, directionIndex, neuraldata, pca_dimension, Interval, currentXPositions, currentYPositions);
        % figure; plot(regressionCoefficientsX, regressionCoefficientsY); title('PCR');
        
        % Store the calculated regression coefficients and the mean windowed firing rates in the model parameters structure.
        modelParameters.pcr(directionIndex,timeWindowIndex).bx = regressionCoefficientsX;
        modelParameters.pcr(directionIndex,timeWindowIndex).by = regressionCoefficientsY;
        modelParameters.pcr(directionIndex,timeWindowIndex).fMean = mean(windowedFiring,1);

        % Store the average hand positions across all trials for each time window.
        % These averages can be useful for evaluating the model's performance.

        modelParameters.averages(timeWindowIndex).avX = squeeze(mean(formattedX,1));
        modelParameters.averages(timeWindowIndex).avY = squeeze(mean(formattedY,1));
   
    end

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%  HELPER FUNCTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
function processed_data = process_and_smooth(trial, group, ema_decay)
% Processes trial data by re-binning, sqrt transformation, and smoothing firing rates with 
% an exponential moving average (EMA).

% Inputs:
%  trial = input struct with spikes and handPos data.
%  group = binning resolution (ms).
%  ema_decay = decay factor for EMA, controlling the degree of smoothing.

    % Initialize the output structure
    processed_data = struct;
    
    % Loop through each trial and reaching angle
    for i = 1:size(trial, 2) % col
        for j = 1:size(trial, 1) % rows

            % Re-bin spikes
            all_spikes = trial(j, i).spikes; 
            neuron_num = size(all_spikes, 1); % row of spikes = 98
            points = size(all_spikes, 2); % timepoints = 600
            t_new = 1:group:points + 1; % binning time: 1x31
            spikes = zeros(neuron_num, numel(t_new) - 1); % new struct for binned spikes 98x30

            for k = 1:length(t_new) - 1 % 30
                if k == length(t_new) - 1 % for the last bin it gets all the leftover points
                    spikes(:, k) = sum(all_spikes(:, t_new(k):end), 2); % Ensure we include all remaining data in the last bin
                else % for all the bins it gets 20ms
                    spikes(:, k) = sum(all_spikes(:, t_new(k):t_new(k + 1) - 1), 2); % 2 ensures that we sum across the time dimension for each neuron
                end
            end

            % Apply sqrt transformation 
            spikes = sqrt(spikes); % 98, 30
            % spikes = log(spikes) - mean(spikes, 2);
            
            % Apply EMA smoothing
            ema_rates = zeros(size(spikes));
            for n = 1: neuron_num
                for t = 2:size(spikes, 2)
                    ema_rates(n, t) = ema_decay * spikes(n, t) + (1 - ema_decay) * ema_rates(n, t - 1);
                end
            end
            
            % Store processed data in the output structure
            processed_data(j, i).rates = ema_rates / (group / 1000); % Adjust rates to spikes per second
            
        end
    end
    % figure;
    % plot(processed_data(1,1).rates(1,:))
end


function [output, weight] = MDF(neuraldata, directions, traininglen, pca_dimension, lda_dimension)

    % Most Discriminant Feature Method:
    % This function performs dimensionality reduction on neural data by first applying PCA to reduce the feature space, 
    % followed by LDA to maximize class separability across different directions or classes. 

    % Input:
    %  neuraldata: Matrix containing the neural firing data, with neurons in rows and time in columns
    %  directions: The number of distinct directions or classes in the data
    %  traininglen: The length of the training data for each direction
    %  pca_dimension: The number of principal components to retain in PCA
    %  lda_dimension: The number of dimensions to retain after performing LDA
    
    % Output:
    %  output: The projected data after applying PCA and LDA for dimensionality reduction
    %  weight: The weights calculated for the most discriminant features
   
    
    % Extract principal components from firing data
    U = getPCA(neuraldata);
    princComp = U(:, 1:pca_dimension); % Select the first pca_dimension principal components
    % figure; plot(princComp);

    % Initialize matrix to store mean vectors for each direction
    mat_between = zeros(size(neuraldata,1), directions);
    
    % Calculate mean vectors for each direction
    for i = 1:directions
        startind = traininglen*(i-1)+1; % 1, 101, 201...
        endind = i*traininglen; % 100, 200, 300... 
        mat_between(:,i) = mean(neuraldata(:, startind:endind), 2);
    end

    % Initialize scatter matrices
    scatter_within = zeros(size(neuraldata, 1));
    scatter_between = zeros(size(neuraldata, 1));
    
    for i = 1:directions
        % Indices for the current direction
        ind = (i-1)*traininglen + 1 : i*traininglen;
        
        % Mean of the current direction
        directionMean = mean(neuraldata(:, ind), 2);
        
        % Scatter within for current direction
        deviation_within = neuraldata(:, ind) - directionMean;
        scatter_within = scatter_within + deviation_within * deviation_within';
        
        % Scatter between calculation
        deviation_between = directionMean - mean(neuraldata, 2);
        scatter_between = scatter_between + traininglen * (deviation_between * deviation_between');
    end
    
    % Perform LDA on the projected data (PCA reduced)
    projectWithin = princComp' * scatter_within * princComp;
    projectBetween = princComp' * scatter_between * princComp;
    
    [eigenvect, eigenval] = eig(pinv(projectWithin) * projectBetween);
    
    % Sort eigenvalues and eigenvectors in descending order
    [~, sortedComp] = sort(diag(eigenval), 'descend');

    eigenvect = eigenvect(:, sortedComp(1:lda_dimension));
    
    % Select the lda_dimension dimensions that maximize class separability
    output = princComp * eigenvect;
    
    % Calculate the optimum projection using the Most Discriminant Feature method
    weight = output' * (neuraldata - mean(neuraldata, 2));
end


function coeff = getPCA(neuraldata)

% Principal Component Analysis (PCA) is utilized here to reduce the dimensionality of the neural data while retaining as much
% variability in the data as possible. This is crucial in neural data analysis, where datasets often contain a large number
% of variables (neural features) relative to the number of observations (trial counts). 

    % Center the data by subtracting the mean of each feature
    neuralDataCentered = bsxfun(@minus, neuraldata, mean(neuraldata, 2));

    % Compute the covariance matrix from the centered data
    C = neuralDataCentered' * neuralDataCentered;

    % Perform eigenvalue decomposition on the covariance matrix
    % V contains the eigenvectors (principal directions), and D contains the eigenvalues on its diagonal
    [V, D] = eig(C);
    
    % Sort the eigenvalues (and corresponding eigenvectors) in descending order.
    [eigenvalues, order] = sort(diag(D), 'descend');
    V = V(:, order);
    
    % Compute the PCA coefficients (scores) by projecting the original centered data onto the principal component axes.
    coeff = neuralDataCentered * V * diag(1./sqrt(eigenvalues));

    % disp('neuralDataCentered '); size(neuralDataCentered)
    % disp('V '); size(V)
    % disp('diag(1./sqrt(eigenvalues)) '); size(diag(1./sqrt(eigenvalues)))
end

function [xTestingInterval, yTestingInterval, formattedX, formattedY] = initializeHandPositionMatrices(neuraldata, traininglen, directions, group, start, finish)

% This function initializes and resamples hand position matrices based on neural data.
% It adjusts the size of each trial's hand position data to match the longest trial,
% ensuring consistent input dimensions for subsequent analyses.

% Input:

%   neuraldata : A struct array containing the neural data, where each element
%                has a field 'handPos' corresponding to hand position data.
%   traininglen : The number of training trials.
%   directions : The number of movement directions 
%   group : The binning interval (in milliseconds) used for resampling.
%   start : The start time (in milliseconds) for the analysis interval.
%   finish : The end time (in milliseconds) for the analysis interval.


% Output:

%   xTestingInterval : The resampled x-coordinates of hand positions.
%   yTestingInterval : The resampled y-coordinates of hand positions.
%   formattedX : The x-coordinates of hand positions, adjusted for length and reshaped.
%   formattedY : The y-coordinates of hand positions, adjusted for length and reshaped.

    % Determine the maximum trajectory size across all trials
    % figure; plot(neuraldata(1,1).handPos(1, :), neuraldata(1,1).handPos(2, :)); title('HandPos Before');
    maxTrajectorySize = max(arrayfun(@(x) size(x.handPos, 2), neuraldata(:)));
    % disp('Max traj'); disp(maxTrajectorySize);

    % Initialize formatted position matrices
    formattedX = zeros(traininglen, maxTrajectorySize, directions);
    formattedY = zeros(traininglen, maxTrajectorySize, directions);

    % Loop over each direction and trial
    for i = 1:directions
        for j = 1:traininglen
            currentSize = size(neuraldata(j, i).handPos, 2);
            paddingSize = maxTrajectorySize - currentSize;
            
            % Extract the hand positions for the current trial
            currentX = neuraldata(j, i).handPos(1, :);
            currentY = neuraldata(j, i).handPos(2, :);
            
            % If padding is necessary, replicate the last position element
            if paddingSize > 0
                % Replicate the last element for padding
                padX = repmat(currentX(end), 1, paddingSize);
                padY = repmat(currentY(end), 1, paddingSize);
                
                % Combine the original data with the padding
                formattedX(j, :, i) = [currentX, padX];
                formattedY(j, :, i) = [currentY, padY];
            else
                % If no padding is needed, just copy the data
                formattedX(j, :, i) = currentX;
                formattedY(j, :, i) = currentY;
            end
        end
    end

    % Resample the formatted matrices to create intervals based on the 'group' parameter
    % This step aligns the data with the temporal analysis window specified by 'start' and 'finish'
    resampledIndices = start:group:finish;
    xTestingInterval = formattedX(:, resampledIndices, :);
    yTestingInterval = formattedY(:, resampledIndices, :);
    % figure; plot(formattedX, formattedY); title('HandPos Before');
end


function [regressionCoefficientsX, regressionCoefficientsY, FilteredFiring ] = calcRegressionCoefficients(timeWindowIndex, timeDivision, labels, directionIndex, neuraldata, pca_dimension, Interval, currentXPositions, currentYPositions)

% This function calculates regression coefficients for predicting hand positions
% from neural data using Principal Component Analysis (PCA) and a regression model.

% Input:
%   timeWindowIndex : Index of the current time window for analysis.
%   timeDivision : Array indicating the division of time into bins.
%   labels : Array of direction labels corresponding to each trial.
%   directionIndex : Index indicating the current direction of movement being analyzed.
%   neuraldata : Matrix of neural firing rates, potentially filtered by previous steps.
%   pca_dimension : Number of principal components to retain in the PCA.
%   Interval : Array of time intervals for analysis.
%   currentXPositions : Matrix of x-coordinates of hand positions across trials.
%   currentYPositions : Matrix of y-coordinates of hand positions across trials.


% Output:
%   regressionCoefficientsX : Regression coefficients for predicting x-coordinates of hand positions.
%   regressionCoefficientsY : Regression coefficients for predicting y-coordinates of hand positions.
%   FilteredFiring : Neural data filtered by time and direction, used for regression.

    
    % Center the positions for the current time window

    centeredX = bsxfun(@minus, currentXPositions(:, timeWindowIndex), mean(currentXPositions(:, timeWindowIndex)));
    centeredY = bsxfun(@minus, currentYPositions(:, timeWindowIndex), mean(currentYPositions(:, timeWindowIndex)));
    
    % Filter firing data based on time and direction
    FilteredFiring = filterFiringData(neuraldata, timeDivision, Interval(timeWindowIndex), labels, directionIndex);
    % assignin('base', "FilteredFiring", FilteredFiring);

    % Center the firing data by subtracting the mean of each neuron's firing rate
    centeredWindowFiring = FilteredFiring  - mean(FilteredFiring ,1);
    % assignin('base', "centeredWindowFiring", centeredWindowFiring);

    % Perform PCA on the centered firing data to reduce dimensionality
    [principalVectors] = getPCA(centeredWindowFiring);
    principalComponents = principalVectors(:, 1:pca_dimension)' * centeredWindowFiring;

    % Calculate regression coefficients for X and Y using the regression matrix
    regressionMatrix = (principalComponents * principalComponents') \ principalComponents;
    regressionCoefficientsX = principalVectors(:, 1:pca_dimension) * regressionMatrix * centeredX;
    regressionCoefficientsY = principalVectors(:, 1:pca_dimension) * regressionMatrix * centeredY;

end

function FilteredFiring = filterFiringData(neuraldata, timeDivision, interval, labels, directionIndex)

% This function filters neural firing data based on specified time and direction criteria. 
% It first selects the firing data up to a given time point (interval) and then further 
% filters the data for a specific movement direction. The function finally centers the 
% filtered data by subtracting the mean firing rate across the selected trials for the specific direction.

% Inputs:
%   neuraldata : Matrix of neural firing rates
%   timeDivision : Array that maps each row in neuraldata to a time interval.
%   interval : Scalar specifying the time point up to which the data should be filtered. 
%   labels : Array of labels indicating the direction of movement associated with each column 
%            in 'neuraldata'. This is used to filter the data based on movement direction.
%   directionIndex : Scalar specifying the direction of movement to filter by. Only data columns
%                    (trials) that correspond to this movement direction will be selected.

% Output:
%   FilteredFiring - The resulting matrix of filtered neural firing rates, where the data has been 
%                    filtered to include only the time points up to 'interval' and trials that 
%                    correspond to the specified 'directionIndex'.


    % Filter the neural data to include only time points up to 'interval'
    timeFilter = timeDivision <= interval;
    % Further filter the data to include only trials corresponding to 'directionIndex'
    directionFilter = labels == directionIndex;
    FilteredFiring  = neuraldata(timeFilter, :);
    % Center the filtered data by subtracting the mean firing rate across the selected trials
    % for the specific direction. 
    FilteredFiring  = FilteredFiring (:, directionFilter) - mean(FilteredFiring(:, directionFilter), 1);
end