% GetAcceleration.m
%
% This function calculates the acceleration from the precomputed velocity.
%
% input:
%   data        - arff data
%   attributes  - attributes describing the data
%   attSpeed    - attribute name holding the speed
%   attDir      - attribute name holding the speed direction
%   windowWidth - step size
%
% output:
%   acceleration    - computed acceleration

function acceleration = GetAcceleration(data, attributes, attSpeed, attDir, windowWidth)
    c_minConf = 0.75;
    step = ceil(windowWidth/2);

    acceleration = zeros(size(data,1),1);

    timeInd = GetAttPositionArff(attributes, 'time');
    confInd = GetAttPositionArff(attributes, 'confidence');
    speedInd = GetAttPositionArff(attributes, attSpeed);
    dirInd = GetAttPositionArff(attributes, attDir);

	for i=1:size(data,1)
        if (data(i,confInd) < c_minConf)
            continue;
        end

        % get initial interval
        if (step == windowWidth)
            startPos = i - step;
            endPos = i;
        else
            startPos = i -step;
            endPos = i + step;
        end

        % fine tune intervals
        if (startPos < 1 || data(startPos,confInd) < c_minConf)
            startPos = i;
        end
        if (endPos > size(data,1) || data(endPos,confInd) < c_minConf)
            endPos = i;
        end

        % invalid interval
        if (startPos == endPos)
            continue;
        end
        
        velStartX = data(startPos,speedInd)*cos(data(startPos,dirInd));
        velStartY = data(startPos,speedInd)*sin(data(startPos,dirInd));

        velEndX = data(endPos,speedInd)*cos(data(endPos,dirInd));
        velEndY = data(endPos,speedInd)*sin(data(endPos,dirInd));

        deltaT = (data(endPos,timeInd)-data(startPos,timeInd))/1000000;

        accX = (velEndX-velStartX)/deltaT;
        accY = (velEndY-velStartY)/deltaT;

        acceleration(i) = sqrt(accX^2 + accY^2);
	end
end
