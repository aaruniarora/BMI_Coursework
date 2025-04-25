function removed_neurons = remove_neurons(spike_matrix, neurons, debug)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identifies and returns neurons with average firing rate < 0.5 Hz
% Remove neurons with very low average firing rate for numerical stability.
%
% Inputs:
%   spike_matrix - matrix of spike data [neurons*bin x trials]
%   neurons      - original number of neurons
% Output:
%   removed_neurons - indices of low-firing neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    removed_neurons = []; 
    for neuronIdx = 1:neurons
        avgFiringRate = mean(mean(spike_matrix(neuronIdx:neurons:end, :)));
        if avgFiringRate < 0.5
            removed_neurons = [removed_neurons, neuronIdx]; 
        end
    end

    if strcmp(debug, 'debug')
        disp(removed_neurons);
    end
end