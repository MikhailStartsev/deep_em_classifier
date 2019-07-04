% AnnotateDataAll.m
%
% This function annotates all the gazsamples with velocity and acceleration data
% input:
%   arffBasepath    - folder containing the input .arff files (with gaze data)
%   outBasepath     - path to the folder where the resulting extracted
%                     feature .arff files will be written; does not have to
%                     exist already.
% Both arguments can be omitted, will then default to the folders intended
% to be used with the repository (see global README).
function AnnotateDataAll(arffBasepath, outBasepath)
    if nargin < 1
        arffBasepath = '../data/inputs/GazeCom_ground_truth';
    end
    if nargin < 2
        outBasepath = '../data/inputs/GazeCom_features';  % already generated features should appear in ../data/inputs/GazeCom_all_features
    end

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
