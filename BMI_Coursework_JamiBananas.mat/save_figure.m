function save_figure(figHandle, outFolder, name, format, type)

    if nargin<5
        type='vector';
    end

    to_day = char(datetime('today', 'Format', 'ddMMMMyyyy'));

    % if you want to save in the same folder, put '.'
    % Get absolute folder path (relative to this script or current dir)
    outFolderFull = fullfile(pwd, outFolder);

    % Create folder if it doesn't exist
    if ~exist(outFolderFull, 'dir')
        mkdir(outFolderFull);
    end

    % Build full file path
    filepath = fullfile(outFolderFull, [name '_' to_day '.' format]);

    if format == "png"
        exportgraphics(figHandle, filepath, 'Resolution', 600);
    elseif any(format == ["pdf", "eps"])
        exportgraphics(figHandle, filepath, 'ContentType', type);
    else
        warning('Unsupported format: %s', format);
    end
end