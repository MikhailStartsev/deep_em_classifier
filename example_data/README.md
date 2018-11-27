The .arff files correspond to the input file `GazeCom/holsten_gaze/GGM_holsten_gate.arff` (all input files can be downloaded [here](https://drive.google.com/drive/folders/1SPGTwUKnvZCUFJO05CnYTqakv-Akdth-?usp=sharing)).

The model corresponds to the "final architecture" (descriptor "4xvC@(32, 16, 8, 8)_0xD@()_2xB@(16, 16)",
speed and direction used as features, context size 257 samples), which was trained **without** the data of the `holsten_gate` video (more information on Leave-One-Video-Out cross-validation in the [corresponding paper](https://rdcu.be/bbMo3)).

You can reproduce this output example by running

> python blstm_model_run.py --feat speed direction --model example_data/model.h5 --in example_data/features.arff --out example_data/output_reproduced.arff
