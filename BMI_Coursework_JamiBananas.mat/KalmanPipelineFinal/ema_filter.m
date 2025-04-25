function ema_spikes = ema_filter(sqrt_spikes, alpha, num_neurons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Applies exponential moving average (EMA) smoothing to spike data
% Inputs:
%   sqrt_spikes  - sqrt-transformed spike matrix [neurons x time bins]
%   alpha        - smoothing factor (higher = more recent weight)
%   num_neurons  - number of input neurons
% Output:
%   ema_spikes   - smoothed spike matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    ema_spikes = zeros(size(sqrt_spikes)); 
    for n = 1:num_neurons
        for t = 2:size(sqrt_spikes, 2)
            ema_spikes(n, t) = alpha * sqrt_spikes(n, t) + (1 - alpha) * ema_spikes(n, t - 1);
        end
    end
end