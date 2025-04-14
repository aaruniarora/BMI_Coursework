function save_figure(figHandle, outFolder, name, format, type, replaceFlag)

    if nargin < 5 || isempty(type)
        type = 'vector';
    end
    if nargin < 6
        replaceFlag = false; % Default: do not replace
    end

    to_day = char(datetime('today', 'Format', 'ddMMMMyyyy'));

    % Get absolute folder path (relative to this script or current dir)
    outFolderFull = fullfile(pwd, outFolder);

    % Create folder if it doesn't exist
    if ~exist(outFolderFull, 'dir')
        mkdir(outFolderFull);
    end

    % Base filename without extension
    baseFileName = [name '_' to_day];
    filepath = fullfile(outFolderFull, [baseFileName '.' format]);

    % Check if file exists and handle versioning if replaceFlag is false
    if exist(filepath, 'file') && ~replaceFlag
        counter = 1;
        while true
            newFileName = sprintf('%s(%d).%s', baseFileName, counter, format);
            filepath = fullfile(outFolderFull, newFileName);
            if ~exist(filepath, 'file')
                break;
            end
            counter = counter + 1;
        end
    end

    % Save figure
    if format == "png"
        exportgraphics(figHandle, filepath, 'Resolution', 600);
    elseif any(format == ["pdf", "eps"])
        exportgraphics(figHandle, filepath, 'ContentType', type);
    else
        warning('Unsupported format: %s', format);
    end
end
