% AnnotateDataAll.m
%
% This function annotates all the gazsamples with velocity and acceleration data

function AnnotateDataAll()
    arffBasepath = '../data/inputs/GazeCom_ground_truth';
    outBasepath = '../data/inputs/GazeCom_features';  % already generated features should be in ../data/inputs/GazeCom_all_features

    dirList = glob([arffBasepath '/*']);
    for i=1:size(dirList,1)
        pos = strfind(dirList{i}, '/');
        name = dirList{i}(pos(end-1)+1:pos(end)-1);
        outputDir = [outBasepath '/' name];

        if (exist(outputDir) ~= 7)
            mkdir(outputDir);
        end

        arffFiles = glob([arffBasepath '/' name '/*.arff']);

        for arffInd=1:size(arffFiles)
            arffFile = arffFiles{arffInd,1};
            [arffDir, arffName, ext] = fileparts(arffFile);
            outputFile = [outputDir '/' arffName '.arff'];

            disp(['Processing ' arffFile]);
            AnnotateData(arffFile, outputFile);
        end
    end
end
