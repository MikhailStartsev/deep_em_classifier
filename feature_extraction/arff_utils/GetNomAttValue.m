% GetNomAttValue.m
%
% This function returns the value of a nominal attribute in its correct form without 
% spaces.
%
% input:
%   attDatatype - the part that describes the attribute after its name
%
% output:
%   attValue    - nominal attribute in its correct form

function [attValue] = GetNomAttValue(attDatatype)
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
    attValue = attDatatype(openCurl:closeCurl);

    % remove spaces from nominal
    attValue = attValue(~isspace(attValue));
end
