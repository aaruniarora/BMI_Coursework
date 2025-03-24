% Fjona Lutaj, Soranna Bacanu, Sakshi Singh, Cristina López Ruiz 

clear all
clc

load("monkeydata_training.mat")

%%%%%% Initialise parameters %%%%%%%
directions = 8;
noTrain =  100; 
group = 20;
ema_decay = 0.35; %arbitrary value decided through multiple trials


trialFinal = process_and_smooth(trial, group, ema_decay);

size_t = size(trialFinal, 1);        
indices = randperm(size_t);          % Create a random permutation of indices from 1 to tot
train_idx = indices(1:noTrain);   

trainData = trialFinal(train_idx, :); 

start = 320;
finish = 560;

timepoints = [start:group:finish]/group;
neuron_num = size(trialFinal(1,1).rates,1);

count = 1;
for time = timepoints
    for i = 1: noTrain
        for j = 1: directions
            for k = 1: time
                colIndex = noTrain * (j - 1) + i;
                rowIndexStart = neuron_num * (k - 1) + 1;
                rowIndexEnd = neuron_num * k;
                neuraldata(rowIndexStart:rowIndexEnd,colIndex) = trainData(i,j).rates(:,k);     
            end
        end
    end
end

% Principal Component Analysis 
princComp= getPCA(neuraldata);

matBetween = zeros(size(neuraldata,1),directions);
 
% Calculate mean vectors for each direction
for i = 1:directions
    startind = noTrain*(i-1)+1;
    endind = i*noTrain;
    mat_between(:,i) = mean(neuraldata(:, startind:endind), 2);
end

% Initialize scatter matrices
scatter_within = zeros(size(neuraldata, 1));
scatter_between = zeros(size(neuraldata, 1));

for i = 1:directions
    % Indices for the current direction
    ind = (i-1)*noTrain + 1 : i*noTrain;
    
    % Mean of the current direction
    directionMean = mean(neuraldata(:, ind), 2);
    
    % Scatter within for current direction
    deviation_within = neuraldata(:, ind) - directionMean;
    scatter_within = scatter_within + deviation_within * deviation_within';
    
    % Scatter between calculation
    deviation_between = directionMean - mean(neuraldata, 2);
    scatter_between = scatter_between + noTrain * (deviation_between * deviation_between');
end

% Perform LDA on the projected data (PCA reduced)
projectWithin = princComp' * scatter_within * princComp;
projectBetween = princComp' * scatter_between * princComp;

[eigenvect, eigenval] = eig(pinv(projectWithin) * projectBetween);

% Sort eigenvalues and eigenvectors in descending order
[~, sortedComp] = sort(diag(eigenval), 'descend');

% Select the lda_dimension dimensions that maximize class separability
lda_dimension = 6;
output = princComp * eigenvect(:, sortedComp(1:lda_dimension));

% Calculate the optimum projection using the Most Discriminant Feature method
weight = output' * (neuraldata - mean(neuraldata, 2));

colors = {[0.8500, 0.3250, 0.0980], [0, 0.4470, 0.7410], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560], ...
          [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330], [0.6350, 0.0780, 0.1840], [0.750, 0.750, 0]};

figure;
hold on;

% Iterate through each direction to plot data
for i = 1:directions
    plot(weight(1, noTrain*(i-1)+1:i*noTrain), weight(2, noTrain*(i-1)+1:i*noTrain), ...
        '.', 'Color', colors{i}, 'MarkerSize', 14); 
end

grid on;
title('Plotted Feature Space of PCA followed by LDA Results');
xlabel('LDA Dimension 1');
ylabel('LDA Dimension 2');
hold off;


legend('30/180π','70/180π','110/180π','150/180π','190/180π','230/180π','310/180π','350/180π');

%% 
% 

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

end

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
    for i = 1:size(trial, 2)
        for j = 1:size(trial, 1)

            % Re-bin spikes
            all_spikes = trial(j, i).spikes;
            neuron_num = size(all_spikes, 1);
            points = size(all_spikes, 2);
            t_new = 1:group:points + 1;
            spikes = zeros(neuron_num, numel(t_new) - 1);

            for k = 1:length(t_new) - 1
                if k == length(t_new) - 1
                    spikes(:, k) = sum(all_spikes(:, t_new(k):end), 2); % Ensure we include all remaining data in the last bin
                else
                    spikes(:, k) = sum(all_spikes(:, t_new(k):t_new(k + 1) - 1), 2);
                end
            end

            % Apply sqrt transformation 
            spikes = sqrt(spikes);
            
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
end