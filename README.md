# Deep eye movement (EM) classifier: a 1D CNN-BLSTM model

This is the implementation of the deep learning approach for eye movement classification from the "1D CNN with BLSTM for automated classification of fixations, saccades, and smooth pursuits" paper. If you use this code, please cite it as

    @Article{startsev2018cnn,
        author="Startsev, Mikhail
        and Agtzidis, Ioannis
        and Dorr, Michael",
        title="1D CNN with BLSTM for automated classification of fixations, saccades, and smooth pursuits",
        journal="Behavior Research Methods",
        year="2018",
        month="Nov",
        day="08",
        issn="1554-3528",
        doi="10.3758/s13428-018-1144-2",
        url="https://doi.org/10.3758/s13428-018-1144-2"
    }

The full paper is freely accessible via a [SharedIt link](https://rdcu.be/bbMo3).

Authors:

Mikhail Startsev, Ioannis Agtzidis, Michael Dorr

For feedback and collaboration you can contact Mikhail Startsev via mikhail.startsev@tum.de, or any of the authors above at < firstname.lastname > @tum.de.

## DEPENDENCIES

To make use of this software, you need to first install the [sp_tool](https://github.com/MikhailStartsev/sp_tool/). For its installation instructions see respective README!

If you want to use blstm_model.py script (to train/test models on GazeCom -- data to be found [here](http://michaeldorr.de/smoothpursuit/)), provide the correct path to the sp_tool folder via the `--sp-tool-folder /path/to/sp_tool/` argument.


You will also need to download and unzip the data archive from [here](https://drive.google.com/drive/folders/1SPGTwUKnvZCUFJO05CnYTqakv-Akdth-?usp=sharing). In particular, 

* The files in `data/models` contain the pre-trained models of two different architectures: the "standard" architecture with 3 Conv1D layers, 1 dense layer, and 1 BLSTM layer (described in the main paper from above) and the improved ("final") architecture that was presented at ECVP'18 (4 Conv1D layers, 2 BLSTM layers) presentation slides can be found [here](http://michaeldorr.de/smoothpursuit/ECVP2018_presentation_slides.pdf)). Note that the "improved" architecture performs better (see paper for the evaluation of the standard model or [the project page](http://michaeldorr.de/smoothpursuit/) for the final one). 
* The files in `data/inputs` need to be unzipped, if you want to use GazeCom data as input to the model
* The files in `data/outputs` need to be unzipped, if you want to examine the outputs of our models without having to run it





## USAGE


This is, perhaps, not the final version of the code, but the pipeline (steps 1 to 3) outlined below has been tested.


The files here provide the interface to our 1D CNN-BLSTM eye movement classification model.

blstm_model.py is the main script that contains all the necessary tools to train and test the model (currently -- on GazeCom).

If you simply want to run our best (so far) model(s) on external data, this is the pipeline you need to follow (using the blstm_model_run.py script):

1. Convert your data into .arff file format (see example_data/input.arff for an example file). Your .arff files need to contain some metadata about the eye tracking experiment (the values are to be set in accordance to your particular data set!):

        %@METADATA width_px 1280.0
        %@METADATA height_px 720.0
        %@METADATA width_mm 400.0
        %@METADATA height_mm 225.0
        %@METADATA distance_mm 450.0

    The fields indicate the width and height of the stimulus or monitor (in pixels), 
    the width and height of the stimulus or monitor in millimeters,
    the distance between the subjects' eyes and the stimulus or monitor (in millimeters).

    The data needs just 4 columns:
    * time (in microseconds)
    * x and y - the on-screen (or relative to the stimulus) coordinates (in piexls; will be converted to degrees of visual angle automatically - that is why we need the metadata)
    * confidence - the eye tracker confidence for the tracking of the subjects' eyes (1 means normal tracking, 0 means lost tracking). If you do not have this information, set to 1 everywhere.

    The format is better described in https://ieeexplore.ieee.org/abstract/document/7851169/ 

2. Such files can then be processed by the Matlab scripts in the `feature_extraction` folder to produce .arff files with features of the gaze (see an example file in example_data/features.arff). For single files, use AnnotateData.m (see respective README; usage: `AnnotateData('path/to/input/file.arff', 'path/to/output/file.arff')` ). You can alternatively use the model that only utilises x and y as feature, in which case you can skip this step.

3. Call `python blstm_model_run.py --input path/to/extracted/feature/file.arff --output folder/where/to/put/the/result.arff --model path/to/model/folder/or/file --feat <feature group name> <feature group name>`.

      Feature groups refer to the groups of features that are present in the path/to/extracted/feature/file.arff file. Can be the following: xy, speed, direction, acceleration. Any number of feature groups can be listed (use space as a separator, see example below). We found that using speed and direction as features performed best.

      Example command (run without quotation marks ` `): 
      `python blstm_model_run.py --feat speed direction --model example_data/model.h5 --in example_data/features.arff --out example_data/output_reproduced.arff`
