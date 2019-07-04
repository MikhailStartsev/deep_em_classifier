% LoadArff.m
%
% Thi funciton loads data from an ARFF file and returns the data, metadata,
% attributes, relation and comments. All returned strings are lower case.
%
% input:
%   arffFile    - path to ARFF file to read
%   
% output:
%   data        - data stored in the ARFF file
%   metadata    - structure holding metadta in the form: metadata.{width_px, height_px, width_mm, height_mm, distance_mm} -1 if not available. Extra metadata are stored in  metadata.extra, which is an nx2 cell array holding name-value pairs
%   attributes  - nx2 cell array with attribute names and types, where n is the number of attributes
%   relation    - relation described in ARFF
%   comments    - nx1 cell array containing one comment line per cell

function [data, metadata, attributes, relation, comments] = LoadArff(arffFile)
    % initialize data
    data = [];
    % initialize metadata
    metadata.width_px = -1;
    metadata.height_px = -1;
    metadata.width_mm = -1; 
    metadata.height_mm = -1; 
    metadata.distance_mm = -1; 
    metadata.extra = {};
    attributes = {};
    relation = '';
    comments = {};

    % nominal attribute handling
    nomMat = logical([]);
    nomMaps = {};

    % read header
    numOfHeaderLines = 1;
    fid = fopen(arffFile, 'r');
    fline = fgetl(fid);
    while (ischar(fline))
        % split lines into words
        words = strsplit(fline,' ');
        % check for relation
        if (size(words,2)>1 && strcmpi(words{1,1},'@relation')==1)
            relation = lower(words{1,2});
        % check for width_px
        elseif (size(words,2)>2 && strcmpi(words{1,1},'%@metadata')==1 && strcmpi(words{1,2},'width_px')==1)
            metadata.width_px = str2num(words{1,3});
        % check for height_px
        elseif (size(words,2)>2 && strcmpi(words{1,1},'%@metadata')==1 && strcmpi(words{1,2},'height_px')==1)
            metadata.height_px = str2num(words{1,3});
        % check for width_mm
        elseif (size(words,2)>2 && strcmpi(words{1,1},'%@metadata')==1 && strcmpi(words{1,2},'width_mm')==1)
            metadata.width_mm = str2num(words{1,3});
        % check for height_mm
        elseif (size(words,2)>2 && strcmpi(words{1,1},'%@metadata')==1 && strcmpi(words{1,2},'height_mm')==1)
            metadata.height_mm = str2num(words{1,3});
        % check for distance_mm
        elseif (size(words,2)>2 && strcmpi(words{1,1},'%@metadata')==1 && strcmpi(words{1,2},'distance_mm')==1)
            metadata.distance_mm = str2num(words{1,3});
        % process the rest of the metadata
        elseif (size(words,2)>2 && strcmpi(words{1,1},'%@metadata')==1)
            pos = size(metadata.extra,1)+1;
            metadata.extra{pos,1} = words{1,2};
            metadata.extra{pos,2} = words{1,3};
        % check for attributes
        elseif (size(words,2)>2 && strcmpi(words{1,1},'@attribute')==1)
            index = size(attributes,1)+1;
            attributes{index,1} = lower(words{1,2});
            attributes{index,2} = words{1,3};
            [isNom, nominalMap] = IsNomAttribute(fline);
            nomMat = [nomMat; isNom];
            if (isNom)
                nomMaps = [nomMaps; {nominalMap}];
                attributes{index,2} = GetNomAttValue(fline);
            else
                nomMaps = [nomMaps; {[]}];
            end
        % check if it is a comment
        elseif (length(fline>0) && fline(1) == '%')
            comments{end+1} = fline;
        % check if data has been reached
        elseif (size(words,2)>0 && strcmpi(words{1,1},'@data')==1)
            break;
        end

        fline = fgetl(fid);
        numOfHeaderLines = numOfHeaderLines+1;
    end

    numAtts = size(attributes,1);
    readFormat = '';
    for ind=1:numAtts
        if (nomMat(ind))
            readFormat = [readFormat '%s '];
        else
            readFormat = [readFormat '%f '];
        end
    end
    lines = textscan(fid, readFormat, 'Delimiter', ',');

    nomIndices = find(nomMat);
    for nomInd=nomIndices'
        if (isempty(nomInd))
            break;
        end

        for ind=1:size(lines{1,nomInd},1)
            lines{1,nomInd}{ind} = nomMaps{nomInd,1}(lines{1,nomInd}{ind});
        end
        lines{1,nomInd} = cell2mat(lines{1,nomInd});
    end

    data = cell2mat(lines);

    fclose(fid);
end    
