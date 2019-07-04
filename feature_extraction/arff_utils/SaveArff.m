% SaveArff.m
%
% Function to save ARFF data to file.
% 
% input:
%   arffFile    - name of the file to save data
%   data        - data to write in arff file
%   metadata    - metadata struct in
%   attributes  - nx2 cell array holding the attribute names
%   relation    - relation described in the file
%   comments    - (optional) nx1 cell array containing one comment line per cell

function SaveArff(arffFile, data, metadata, attributes, relation, comments)
    if (nargin < 6)
        comments = {};
    end
    % check input
    assert(isfield(metadata,'width_px'), 'metadata should contain "width_px" field');
    assert(isfield(metadata,'height_px'), 'metadata should contain "height_px" field');
    assert(isfield(metadata,'width_mm'), 'metadata should contain "width_mm" field');
    assert(isfield(metadata,'height_mm'), 'metadata should contain "height_mm" field');
    assert(isfield(metadata,'distance_mm'), 'metadata should contain "distance_mm" field');
    assert(size(relation,2)>0, 'relation should not be empty');
    assert(size(attributes,1)==size(data,2), 'attribute number should be the same with data');

    % start writing
    fid = fopen(arffFile, 'w+');

    % write relation
    fprintf(fid, '@RELATION %s\n\n', relation);

    % write metadata
    fprintf(fid, '%%@METADATA width_px %d\n', metadata.width_px);
    fprintf(fid, '%%@METADATA height_px %d\n', metadata.height_px);
    fprintf(fid, '%%@METADATA width_mm %.2f\n', metadata.width_mm);
    fprintf(fid, '%%@METADATA height_mm %.2f\n', metadata.height_mm);
    fprintf(fid, '%%@METADATA distance_mm %.2f\n\n', metadata.distance_mm);

    % write metadata extras. Those are data that vary between experiments
    for i=1:size(metadata.extra,1)
        fprintf(fid, '%%@METADATA %s %s\n', metadata.extra{i,1}, metadata.extra{i,2});
    end
    % print an empty line
    fprintf(fid, '\n');

    % write attributes and get their type
    % 1 = integer
    % 2 = numeric
    % 3 = nominal
    % -1 = other
    numAtts = size(attributes,1);
    attType = -1*ones(numAtts,1);
    numMaps = cell(numAtts,1);
    for i=1:numAtts
        fprintf(fid, '@ATTRIBUTE %s %s\n', attributes{i,1}, attributes{i,2});
        [isNom, ~, numericMap] = IsNomAttribute(attributes{i,2});

        % get type
        if (strcmpi(attributes{i,2},'integer')==1) 
            attType(i) = 1;
        elseif (strcmpi(attributes{i,2},'numeric')==1)
            attType(i) = 2;
        elseif (isNom)
            attType(i) = 3;
            numMaps{i,1} = numericMap;
        end
    end

    % write comments if they exist
    if (~isempty(comments))
        fprintf(fid, '\n');
        for i=1:length(comments)
            comment = comments{i};
            % check if % is the first character
            if (length(comment)>0 && comment(1)~='%')
                comment = ['%' comment];
            end

            fprintf(fid, '%s\n', comment);
        end
    end

    % write data keyword
    fprintf(fid,'\n@DATA\n');

    numEntries = size(data,1);
    % transpose data in order to allow one line writing because fprintf handles 
    % matrices column wise when writing in file
    data = num2cell(data');
    nomIndices = find(attType==3);
    for nomInd=nomIndices'
        if (isempty(nomInd))
            break;
        end

        % convert numbers to nominal values
        for ind=1:numEntries
            data{nomInd, ind} = numMaps{nomInd,1}(data{nomInd, ind});
        end
    end

    writeFormat = '';
    for ind=1:numAtts
        if (attType(ind) == 1)
            writeFormat = [writeFormat '%d'];
        elseif (attType(ind) == 2)
            writeFormat = [writeFormat '%.2f'];
        elseif (attType(ind) == 3)
            writeFormat = [writeFormat '%s'];
        else
            error(['Attribute type "' num2str(attType(ind)) '" is not recognised']);
        end

        if (ind<numAtts)
            writeFormat = [writeFormat ','];
        end
    end
    writeFormat = [writeFormat '\n'];

    % One line writing almost halves the writing time
    fprintf(fid, writeFormat, data{:});
    %for ind=1:numEntries
    %    fprintf(fid, writeFormat, data{ind,:});
    %end

    % close file
    fclose(fid);
end    
