function [xn, yn, xrs, yrs] = getEqualandSampled(data, noDirections, noTrain, group)
% getEqualandSampled  Extract and downsample the hand positions 
%
% usage:
%   [xn, yn, xrs, yrs] = getEqualandSampled(data, noDirections, noTrain, group)
%
% Steps:
%   - Pad handPos to the max trial length.
%   - xrs, yrs => the downsampled positions matching bin size = group (ms).

    trialHolder = struct2cell(data);
    sizes = [];
    for i = 2:3:noTrain*noDirections*3
        sizes = [sizes, size(trialHolder{i}, 2)];
    end
    maxSize = max(sizes);

    xn = zeros(noTrain, maxSize, noDirections);
    yn = zeros(noTrain, maxSize, noDirections);

    for dir_i = 1:noDirections
        for tr_i = 1:noTrain
            thisPos = data(tr_i, dir_i).handPos;  % 3 x T
            len     = size(thisPos, 2);

            xn(tr_i, 1:len, dir_i) = thisPos(1, :);
            yn(tr_i, 1:len, dir_i) = thisPos(2, :);

            % If short, pad with final value
            if len < maxSize
                xn(tr_i, len+1:maxSize, dir_i) = thisPos(1, end);
                yn(tr_i, len+1:maxSize, dir_i) = thisPos(2, end);
            end
        end
    end

    xrs = xn(:, 1:group:end, :);
    yrs = yn(:, 1:group:end, :);
end
