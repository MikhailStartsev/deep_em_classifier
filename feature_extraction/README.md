To extract features from GazeCom data, run AnnotateDataAll. 
If some alterations are needed, make sure to change paths in that file not to override the existing .arff feature files!

If you want to extract features from another data set, you can manually run the AnnotateData function in a loop for all 
the needed files (run as AnnotateData('path/to/input/file.arff', 'path/to/output/file.arff') ).

The .arff files to be processed need to have at least 4 columns:  
  - x,      		the x coordinate of gaze location (in pixels, in case of GazeCom)
  - y,      		the x coordinate of gaze location (in pixels, in case of GazeCom)
  - time,   		the timestamp of the respective sample (in microseconds)
  - confidence		the tracking confidence value (from the eye tracker).

In the feature extraction scripts, the "acceptable" confidence level is set to 0.75: 
every sample with confidence value below this threshold will be discarded (features 
set to zero). In GazeCom, the confidence values are either 0 or 1 (0 meaning lost tracking,
1 meaning normal tracking). If your data does not have such information, set confidence
to 1 for all samples.
