function trialFinal = get_firing_rates(trialProcessed, group, scale_window)
% get_firing_rates  Gaussian smoothing of binned spikes => firing rates in Hz
%
% usage:
%   trialFinal = get_firing_rates(trialProcessed, group, scale_window)
%
% trialProcessed can be a [nTrials x nAngles] struct array, or single-struct.
% group = bin size in ms
% scale_window = typically 50 => gaussian smoothing window

    win = 10*(scale_window/group);
    normstd = scale_window/group;
    alpha = (win-1)/(2*normstd);
    temp1 = -(win-1)/2 : (win-1)/2;
    gausstemp = exp((-1/2)*(alpha * temp1/((win-1)/2)).^2)';
    gaussian_window = gausstemp/sum(gausstemp);

    if numel(trialProcessed) > 1
        [nTrials, nAngles] = size(trialProcessed);
        trialFinal = repmat(struct, nTrials, nAngles);

        for i = 1:nAngles
            for j = 1:nTrials
                bin_spikes = trialProcessed(j,i).spikes;  % (#neurons x #bins)
                no_neurons = size(bin_spikes,1);
                hold_rates = zeros(no_neurons, size(bin_spikes,2));

                for nn = 1:no_neurons
                    % convolve => convert spike counts to a rate (divide by bin_time in sec)
                    hold_rates(nn,:) = conv(bin_spikes(nn,:), gaussian_window, 'same') ...
                                       / (group/1000);
                end

                trialFinal(j,i).rates   = hold_rates;
                trialFinal(j,i).handPos = trialProcessed(j,i).handPos;
                trialFinal(j,i).bin_size= trialProcessed(j,i).bin_size;
            end
        end

    else
        % single struct scenario
        bin_spikes = trialProcessed.spikes;
        no_neurons = size(bin_spikes,1);
        hold_rates = zeros(no_neurons, size(bin_spikes,2));

        for nn = 1:no_neurons
            hold_rates(nn,:) = conv(bin_spikes(nn,:), gaussian_window, 'same') ...
                               / (group/1000);
        end

        trialFinal.rates   = hold_rates;
        trialFinal.handPos = trialProcessed.handPos;
        trialFinal.bin_size= trialProcessed.bin_size;
    end
end
