% function GetAttPositionArff:
%
% Gets a list of attributes returned from LoadArff and an attribute name to
% search.  If it finds the attribute returns its index otherwise it can raise
% an error.
%
% input:
%   arffAttributes  - attribute list returned from LoadArff
%   attribute       - attribute to search
%   check           - (optional) boolean to check if attribute exists. Default is true
%
% output:
%   attIndex        - index attribute of the attribute in the list if it was found. 
%                     Returns 0 if it wasn't found

function [attIndex] = GetAttPositionArff(arffAttributes, attribute, check)
    if (nargin < 3)
        check = true;
    end
    attIndex = 0;

    for i=1:size(arffAttributes,1)
        if (strcmpi(arffAttributes{i,1}, attribute) == 1)
            attIndex = i;
        end
    end

    % check index
    if (check)
        assert(attIndex>0, ['Attribute "' attribute '" not found']);
    end
end
