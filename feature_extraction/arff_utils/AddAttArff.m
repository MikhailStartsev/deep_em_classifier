% function AddAttArff:
%
% This function adds the data and the name of the new attribute to the initial
% data as a new column.
%
% input:
%   data            - data of the initial arff file
%   attributes      - attributes of the initial arff file
%   attData         - attribute data to append at the data. When nominal attributes 
%                     are appended the attribute values should corespond to the enumeration
%                     equivalent
%   attName         - attribute name
%   attType         - attribute type (Integer, Numeric or nominal in the form '{val1,val2}')
%
% output:
%   newData         - data after addition of the new column
%   newAttributes   - attributes containing the addition of the new attribute

function [newData, newAttributes] = AddAttArff(data, attributes, attData, attName, attType)
    % are data and new attribute smae size
    assert(size(data,1)==size(attData,1), 'Provided attribute does not have same number of entries as initial data');
	
	% check if attribute already exists
    for i=1:size(attributes,1)
        if (strcmpi(attributes{i,1}, attName))
            error(['Attributes "' attName '" already exists. Cannot add it.']);
        end
    end

    % merge returned attributes
    newAttributes = attributes;
    index = size(attributes,1)+1;
    newAttributes{index,1} = attName;
    newAttributes{index,2} = attType;

    % concatenate attribute to the returned data
    newData = zeros(size(data,1), size(data,2)+1);
    newData(:,1:end-1) = data(:,:);
    newData(:,end) = attData(:);
end
