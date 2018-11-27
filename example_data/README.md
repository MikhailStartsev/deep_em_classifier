The .arff files correspond to the input file GazeCom/holsten_gaze/GGM_holsten_gate.arff

The model corresponds to the "final architecture" (descriptor "4xvC@(32, 16, 8, 8)_0xD@()_2xB@(16, 16)",
speed and direction used as features, context size 257 samples).

You can reproduce this output by running

> python blstm_model_run.py --feat speed direction --model example_data/model.h5 --in example_data/features.arff --out example_data/output_reproduced.arff
