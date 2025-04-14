function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)
% bin_and_sqrt  Re-bin spike data from 1 ms to 'group' ms and optionally sqrt-transform
%
% usage:
%   trialProcessed = bin_and_sqrt(trial, group, to_sqrt)
%
% Inputs:
%   - trial: if multi-element, we assume trainingData format => trial(j,i).spikes
%   - group: bin width in ms
%   - to_sqrt: 1 => sqrt spike counts, 0 => no transform
%
% Output:
%   trialProcessed: same shape as 'trial', but with .spikes re-binned

    if numel(trial) > 1
        [nTrials, nAngles] = size(trial);
        trialProcessed = repmat(struct, nTrials, nAngles);

        for i = 1:nAngles
            for j = 1:nTrials
                all_spikes = trial(j,i).spikes;  % #neurons x #timepoints
                no_neurons = size(all_spikes,1);
                no_points  = size(all_spikes,2);

                t_new = 1:group:(no_points+1);
                spikes = zeros(no_neurons, numel(t_new)-1);

                for k = 1:numel(t_new)-1
                    spikes(:,k) = sum(all_spikes(:, t_new(k):(t_new(k+1)-1)), 2);
                end

                if to_sqrt
                    spikes = sqrt(spikes);
                end

                trialProcessed(j,i).spikes  = spikes;
                if isfield(trial(j,i), 'handPos')
                    trialProcessed(j,i).handPos = trial(j,i).handPos(1:2,:);
                else
                    trialProcessed(j,i).handPos = [];
                end
                trialProcessed(j,i).bin_size = group;
            end
        end

    else
        % single struct scenario
        all_spikes = trial.spikes;  % #neurons x #timepoints
        no_neurons = size(all_spikes,1);
        no_points  = size(all_spikes,2);

        t_new = 1:group:(no_points+1);
        spikes = zeros(no_neurons, numel(t_new)-1);

        for k = 1:numel(t_new)-1
            spikes(:,k) = sum(all_spikes(:, t_new(k):(t_new(k+1)-1)), 2);
        end

        if to_sqrt
            spikes = sqrt(spikes);
        end

        trialProcessed.spikes  = spikes;
        if isfield(trial, 'handPos')
            trialProcessed.handPos = trial.handPos;
        else
            trialProcessed.handPos = [];
        end
        trialProcessed.bin_size = group;
    end
end
