import argparse
import os
import sys
import glob
import tempfile
import numpy as np

import keras

from sp_tool.arff_helper import ArffHelper
import sp_tool.util as sp_util
from sp_tool.evaluate import CORRESPONDENCE_TO_HAND_LABELLING_VALUES
from sp_tool import recording_processor as sp_processor

import blstm_model
from blstm_model import zip_equal


def run(args):
    """
    Run prediction for a trained model on a set of .arff files (with features already extracted).
    See feature_extraction folder for the code to compute appropriate features.
    :param args: command line arguments
    :return: a list of tuples (corresponding to all processed files) that consist of
               - the path to an outputted file
               - predicted per-class probabilities
    """
    subfolders_and_fnames = find_all_subfolder_prefixes_and_input_files(args)
    out_fnames = get_corresponding_output_paths(subfolders_and_fnames, args)

    print('Processing {} file(s) from "{}" into "{}"'.format(len(out_fnames),
                                                             args.input,
                                                             args.output))

    arff_objects = [ArffHelper.load(open(fname)) for _, fname in subfolders_and_fnames]

    keys_to_keep = blstm_model.get_arff_attributes_to_keep(args)
    print('Will look for the following keys in all .arff files: {}. ' \
          'If any of these are missing, an error will follow!'.format(keys_to_keep))
    all_features = [get_features_columns(obj, args) for obj in arff_objects]

    model = keras.models.load_model(args.model_path,
                                    custom_objects={'f1_SP': blstm_model.f1_SP,
                                                    'f1_SACC': blstm_model.f1_SACC,
                                                    'f1_FIX': blstm_model.f1_FIX})

    # Guess the padding size from model input and output size
    window_length = model.output_shape[1]  # (batch size, window size, number of classes)
    padded_window_shape = model.input_shape[1]  # (batch size, window size, number of features)
    padding_features = (padded_window_shape - window_length) // 2
    print('Will pad the feature sequences with {} samples on each side.'.format(padding_features))

    keys_to_subtract_start = sorted({'x', 'y'}.intersection(keys_to_keep))
    if len(keys_to_subtract_start) > 0:
        print('Will subtract the starting values of the following features:', keys_to_subtract_start)
    keys_to_subtract_start_indices = [i for i, key in enumerate(keys_to_keep) if key in keys_to_subtract_start]

    predictions, _ = blstm_model.evaluate_test(model=model,
                                               X=all_features,
                                               y=None,  # no ground truth available or needed
                                               keys_to_subtract_start_indices=keys_to_subtract_start_indices,
                                               correct_for_unknown_class=False,
                                               padding_features=padding_features,
                                               split_by_items=True)

    CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE = {v: k for k, v in
                                                       CORRESPONDENCE_TO_HAND_LABELLING_VALUES.items()}
    print('Class names:', CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE)

    for original_obj, out_fname, predicted_labels in zip_equal(arff_objects, out_fnames, predictions['pred']):
        # Create folders that might not exist yet
        containing_folder = os.path.split(out_fname)[0]
        if not os.path.exists(containing_folder):
            os.makedirs(containing_folder)

        # Get labels from probabilities for each label
        labels_pred = np.argmax(predicted_labels, axis=-1)
        # We get outputs as windows of labels, so now need to assemble one whole sequence.
        # Also need to cut the result, since it contains only whole windows of data and was respectively mirror-padded
        labels_pred = np.concatenate(labels_pred)[:original_obj['data'].shape[0]]

        # Add a column containing predicted labels
        original_obj = ArffHelper.add_column(original_obj,
                                             name=sp_processor.EM_TYPE_ATTRIBUTE_NAME,
                                             dtype=sp_processor.EM_TYPE_ARFF_DATA_TYPE,
                                             default_value=sp_processor.EM_TYPE_DEFAULT_VALUE)
        # Fill in with categorical values instead of numerical ones
        # (use @CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE for conversion)
        original_obj['data'][sp_processor.EM_TYPE_ATTRIBUTE_NAME] = \
            [CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE[x] for x in labels_pred]

        ArffHelper.dump(original_obj, open(out_fname, 'w'))

    print('Prediction and file operations finished, check {} for outputs!'.format(args.output))

    return zip_equal(out_fnames, predictions['pred'])


def parse_args():
    # Will keep most of the arguments, but suppress others
    base_parser = blstm_model.parse_args(dry_run=True)
    # Inherit all arguments, but retain the possibility to add the same args, but suppress them
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False, conflict_handler='resolve')

    # List all arguments (as lists of all ways to address each) that are to be eradicated
    args_to_suppress = [
        ['--model-name', '--model'],  # will add a more intuitive --model-path argument below
        # no need for the following when training is completed already
        ['--initial-epoch'],
        ['--batch-size'],
        ['--run-once', '--once', '-o'],
        ['--run-once-video'],
        ['--ground-truth-folder', '--gt-folder'],  # no need for ground truth
        ['--final-run', '--final', '-f'],  # it's always a "final" run here
        ['--folder', '--output-folder'],   # will override
        ['--training-samples'],
        ['--sp-tool-folder']
    ]

    for arg_group in args_to_suppress:
        parser.add_argument(*arg_group, help=argparse.SUPPRESS)

    parser.add_argument('--input', '--in', required=True,
                        help='Path to input data. Can be either a single .arff file, or a whole directory. '
                             'In the latter case, this directory will be scanned for .arff files, and all of them will '
                             'be used as inputs, generating corresponding labelled files.')

    # rewrite the help
    parser.add_argument('--output', '--output-folder', '--out', dest='output', default=None,
                        help='Write prediction results as ARFF file(s) here. Will mimic the structure of the --input '
                             'folder, or just create a single file, if --input itself points to an .arff file. '
                             'Can be a path to the desired output .arff file, in case --input is also just one file. '
                             'If not provided, will create a temporary folder and write the outputs there.')

    parser.add_argument('--model-path', '--model', default=None,
                        help='Path to a particular model (an .h5 file), which is to be used, or a folder containing '
                             'all 18 models that are trained in the Leave-One-Video-Out cross-validation procedure '
                             'on GazeCom. If this argument is '
                             'provided, it overrides all the architecture- and model-defining parameters. The '
                             'provided .h5 file will be loaded instead. \n\nIf --model-path is not provided, will '
                             'generate a model descriptor from architecture parameters and so on, and look for it '
                             'in the respective subfolder of ``data/models/''. Will then (or if --model-path contains '
                             'a path to a folder, and not to an .h5 file) take the model that was '
                             'trained on all data except for `bridge_1`, since this video has no "true" smooth '
                             'pursuit, so we will this way maximise the amount of this relatively rare class in the '
                             'used training set.')

    args = parser.parse_args()

    if args.model_path is None:
        model_descriptor = blstm_model.get_full_model_descriptor(args)
        args.model_path = 'data/models/LOO_{descr}/'.format(descr=model_descriptor)

    # If it is a path to a directory, find the model trained for the ``bridge_1'' clip.
    # Otherwise, we just assume that the path points to a model file.
    if os.path.isdir(args.model_path):
        all_model_candidates = sorted(glob.glob('{}/*_without_bridge_1*.h5'.format(args.model_path)))
        if len(all_model_candidates) == 0:
            raise ValueError('No model in the "{dir}" folder has ``without_bride_1\'\' in its name. Either pass '
                             'a path to an exact .h5 model file in --model-path, or make sure you have the right model '
                             'in the aforementioned folder.'.format(dir=args.model_path))
        elif len(all_model_candidates) > 1:
            raise ValueError('More than one model with ``without_bride_1\'\' in its name has been found in the "{dir}" '
                             'folder: {candidates}. Either pass a path to an exact .h5 model file in --model-path, '
                             'or make sure you have only one model trained without the clip ``bridge_1\'\' in the '
                             'aforementioned folder.'.format(dir=args.model_path,
                                                             candidates=all_model_candidates))
        args.model_path = all_model_candidates[0]  # since there has to be just one

    return args


def find_all_subfolder_prefixes_and_input_files(args):
    """
    Extract a matching set of paths to .arff files and additional folders between the --input folder and the files
    themselves (so that we will be able to replicate the structure later on)
    :param args: command line arguments
    :return: a list of tuples, where the first element is the sub-folder prefix and the second one is the full path
             to each .arff file
    """
    if os.path.isfile(args.input):
        return [('', args.input)]
    assert os.path.isdir(args.input), '--input is neither a file nor a folder'

    res = []
    for dirpath, dirnames, filenames in os.walk(args.input):
        filenames = [x for x in filenames if x.lower().endswith('.arff')]
        if filenames:
            dirpath_suffix = dirpath[len(args.input):].strip('/')
            res += [(dirpath_suffix, os.path.join(dirpath, fname)) for fname in filenames]
    return res


def get_corresponding_output_paths(subfolders_and_full_input_filenames, args):
    """
    Create a list that will contain output paths for all the @subfolders_and_full_input_filenames
    (the output of find_all_subfolder_prefixes_and_input_files() function) in the output folder.
    :param subfolders_and_full_input_filenames: subfolder prefixes,
           returned by find_all_subfolder_prefixes_and_input_files()
    :param args: command line arguments
    :return:
    """
    if args.output is None:
        args.output = tempfile.mkdtemp(prefix='blstm_model_output_')
        print('No --output was provided, creating a folder in', args.output, file=sys.stderr)

    if args.output.lower().endswith('.arff'):
        assert len(subfolders_and_full_input_filenames) == 1, 'If --output is just one file, cannot have more than ' \
                                                              'one input file! Consider providing a folder as the ' \
                                                              '--output.'
        return [args.output]

    res = []
    for subfolder, full_name in subfolders_and_full_input_filenames:
        res.append(os.path.join(args.output, subfolder, os.path.split(full_name)[-1]))
    return res


def get_features_columns(arff_obj, args):
    """
    Extracting features from the .arff file (reading the file, getting the relevant columns
    :param arff_obj: a loaded .arff file
    :param args: command line arguments
    :return:
    """
    keys_to_keep = blstm_model.get_arff_attributes_to_keep(args)

    keys_to_convert_to_degrees = ['x', 'y'] + [k for k in keys_to_keep if 'speed_' in k or 'acceleration_' in k]
    keys_to_convert_to_degrees = sorted(set(keys_to_convert_to_degrees).intersection(keys_to_keep))
    # Conversion is carried out by dividing by pixels-per-degree value (PPD)
    if get_features_columns.run_count == 0:
        if len(keys_to_convert_to_degrees) > 0:
            print('Will divide by PPD the following features', keys_to_convert_to_degrees)
    get_features_columns.run_count += 1

    # normalize coordinates in @o by dividing by @ppd_f -- the pixels-per-degree value of the @arff_obj
    ppd_f = sp_util.calculate_ppd(arff_obj)
    for k in keys_to_convert_to_degrees:
        arff_obj['data'][k] /= ppd_f

    # add to respective data sets (only the features to be used and the true labels)
    return np.hstack([np.reshape(arff_obj['data'][key], (-1, 1)) for key in keys_to_keep]).astype(np.float64)


get_features_columns.run_count = 0

if __name__ == '__main__':
    run(parse_args())
