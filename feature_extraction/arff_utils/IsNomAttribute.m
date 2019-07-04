% IsNomAttribute.m
%
% This function checks if an attribute is of nominal type and returns true along
% with nominal and numeric maps. Otherwise it returns false.
%
% input:
%   attDatatype - the part that describes the attribute after its name
%
% output:
%   isNom       - boolean value denoting if nominal
%   nominalMap  - mapping of nominal values to doubles as in an C++ enumeration
%   numericMap  - mapping of doubles to nominal values

function [isNom, nominalMap, numericMap] = IsNomAttribute(attDatatype)
    openCurl = strfind(attDatatype, '{');
    closeCurl = strfind(attDatatype, '}');

    if (isempty(openCurl) && isempty(closeCurl))
        isNom = false;
        nominalMap = containers.Map;
        numericMap = containers.Map;
        return;
    end

    assert(length(openCurl) == 1, ['Invalid attribute datatype ' attDatatype]);
    assert(length(closeCurl) == 1, ['Invalid attribute datatype ' attDatatype]);
    attDatatype = attDatatype(openCurl+1:closeCurl-1);

    % remove spaces from nominal
    attDatatype = attDatatype(~isspace(attDatatype));

    keys = split(attDatatype, ',');
    values = 0:length(keys)-1;

    nominalMap = containers.Map(keys, values);

    % convert to simple when we have single key. Otherwise the type is invalid for map creation
    if (length(keys) == 1)
        keys = string(keys);
    end
    numericMap = containers.Map(values, keys);
    isNom = true;
end
