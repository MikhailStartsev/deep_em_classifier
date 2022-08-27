#!/usr/bin/env python
# coding: utf-8
from datetime import datetime

import glob
import pickle
import json

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, TimeDistributed, Flatten, Activation, Dropout, Bidirectional
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import History, TensorBoard, Callback
import keras.initializers as KI
from keras.layers import BatchNormalization

from keras import backend as K
import numpy as np
import os
from argparse import ArgumentParser
import math
from copy import deepcopy
import itertools
import warnings

from sp_tool import util as sp_util
from sp_tool.arff_helper import ArffHelper
from sp_tool.evaluate import CORRESPONDENCE_TO_HAND_LABELLING_VALUES
from sp_tool import recording_processor as sp_processor

# If need to set a limit on how much GPU memory the model is allowed to use

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))


def zip_equal(*args):
    """
    Iterate the zip-ed combination of @args, making sure they have the same length
    :param args: iterables to zip
    :return: yields what a usual zip would
    """
    fallback = object()
    for combination in itertools.zip_longest(*args, fillvalue=fallback):
        if any((c is fallback for c in combination)):
            raise ValueError('zip_equals arguments have different length')
        yield combination


def categorical_f1_score_for_class(y_true, y_pred, class_i, dtype=None):
    """
    A generic function for computing sample-level F1 score for a class @class_i (i.e. for the classification problem
    "is @class_i" vs "is not @class_i").
    :param y_true: one-hot encoded true labels for a set of samples
    :param y_pred: predicted probabilities for all classes for the same set of samples
    :param class_i: which class to consider for the F1 score computation
    :param dtype: an optional intermediate data type parameter; unless some relevant exception is raised (type mismatch,
                  for example), no need to pass anything; in some cases 'float64' had to be passed, instead of the
                  default 'float32'
    :return: one floating-point value, the F1 score for the binary @class_i vs not-@class_i problem
    """
    pred_labels = K.argmax(y_pred, axis=-1)
    tp = K.sum(y_true[:, :, class_i] * K.cast(K.equal(pred_labels, class_i), 'float32' if not dtype else dtype))
    all_p_detections = K.sum(K.cast(K.equal(pred_labels, class_i), 'float32' if not dtype else dtype))
    all_p_true = K.sum(y_true[:, :, class_i])

    precision = tp / (all_p_detections + K.epsilon())
    recall = tp / (all_p_true + K.epsilon())
    f_score = 2 * precision * recall / (precision + recall + K.epsilon())

    return f_score


# A set of F1-score functions for the three major eye movement types that we use to monitor model training on both
# the training and the validation set. Almost the same signature as the "master"
# function categorical_f1_score_for_class() above.

def f1_FIX(y_true, y_pred, dtype=None):
    return categorical_f1_score_for_class(y_true, y_pred, 1, dtype)


def f1_SACC(y_true, y_pred, dtype=None):
    return categorical_f1_score_for_class(y_true, y_pred, 2, dtype)


def f1_SP(y_true, y_pred, dtype=None):
    return categorical_f1_score_for_class(y_true, y_pred, 3, dtype)


def create_model(num_classes, batch_size, train_data_shape, dropout_rate=0.3,
                 padding_mode='valid',
                 num_conv_layers=3, conv_filter_counts=(32, 16, 8),
                 num_dense_layers=1, dense_units_count=(32,),
                 num_blstm_layers=1, blstm_unit_counts=(16,),
                 unroll_blstm=False,
                 stateful=False,
                 no_bidirectional=True):
    """
    Create a 1D CNN-BLSTM model that contains 3 blocks of layers: Conv1D block, Dense block, and BLSTM block.
    Each of these is configurable via the parameters of this function; only the convolutional block cannot be entirely
    skipped.

    Each layer in the Conv1D block has the filter size of 3, and is followed by BatchNormalization and ReLU activation.
    Every layer in this block, starting from the second one, is preceded by Dropout.

    In the Dense block, every layer uses a TimeDistributed wrapper and is preceded by Dropout, followed by ReLU
    activation.

    All [B]LSTM layer(s) use tanh activation.

    After the BLSTM block, the model contains a single (time-distributed) Dense layer with softmax activation that
    has the @num_classes units.


    :param num_classes: number of classes to be classified
    :param batch_size: batch size
    :param train_data_shape: shape of the training data array (will infer sequence length -- @train_data_shape[1] -- and
                             number of features -- train_data_shape[2] -- here, @train_data_shape[0] is ignored, and
                             @batch_size is used instead).
    :param dropout_rate: each convolutional (except for the first one) and dense layer is preceded by Dropout with this
                         rate.
    :param padding_mode: convolution padding mode; can be 'valid' (default), 'same', or 'causal'; the latter can be
                         useful, if a modification into a realtime-like model is desired:
                         https://keras.io/layers/convolutional/#conv1d
    :param num_conv_layers: number of convolutional layers in the Conv1D block
    :param conv_filter_counts: number of filters in each respective Conv1D layer; has to be of length of at least
                               @num_conv_layers - will use the first @num_conv_layers elements
    :param num_dense_layers: number of dense layers in the Dense block
    :param dense_units_count: number of units in each respective Dense layer; has to be of length of at least
                              @num_dense_layers - will use the first @num_dense_layers elements
    :param num_blstm_layers: number of dense layers in the BLSTM block
    :param blstm_unit_counts: number of units in each respective [B]LSTM layer; has to be of length of at least
                              @num_blstm_layers - will use the first @num_blstm_layers elements
    :param unroll_blstm: whether to unroll the [B]LSTM(s), see https://keras.io/layers/recurrent/#lstm
    :param stateful: whether to make the [B]LSTM(s) stateful; not used yet
    :param no_bidirectional: if True, will use traditional LSTMs, not the Bidirectional wrapper,
                             see https://keras.io/layers/wrappers/#bidirectional
    :return: a keras.models.Sequential() model
    """
    assert num_conv_layers <= len(conv_filter_counts)
    if len(conv_filter_counts) != num_conv_layers:
        warnings.warn('@num_conv_layers={} is shorter than @conv_filter_counts={}, so the last {} elements of the '
                      'latter will be ignored. Might be incorrectly passed arguments!'.format(
                        num_conv_layers, conv_filter_counts, len(conv_filter_counts) - num_conv_layers))

    assert num_dense_layers <= len(dense_units_count)
    if len(dense_units_count) != num_dense_layers:
        warnings.warn('@num_dense_layers={} is shorter than @dense_unit_counts={}, so the last {} elements of the '
                      'latter will be ignored. Might be incorrectly passed arguments!'.format(
                       num_dense_layers, dense_units_count, len(dense_units_count) - num_dense_layers))

    assert num_blstm_layers <= len(blstm_unit_counts)
    if len(blstm_unit_counts) != num_blstm_layers:
        warnings.warn('@num_blstm_layers={} is shorter than @blstm_unit_counts={}, so the last {} elements of the '
                      'latter will be ignored. Might be incorrectly passed arguments!'.format(
                        num_conv_layers, conv_filter_counts, len(blstm_unit_counts) - num_blstm_layers))

    model = Sequential()

    for conv_layer_id in range(num_conv_layers):
        if conv_layer_id != 0:
            model.add(Dropout(dropout_rate))

        conv_layer_args = {
            'filters': conv_filter_counts[conv_layer_id],
            'kernel_size': 3,
            'padding': padding_mode,
            'kernel_initializer': KI.RandomNormal(),
            'bias_initializer': KI.Ones()
        }
        # special args for the first layer
        if conv_layer_id == 0:
            conv_layer_args['batch_input_shape'] = (batch_size, train_data_shape[1], train_data_shape[2])

        model.add(Conv1D(**conv_layer_args))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

    model.add(TimeDistributed(Flatten()))

    for dense_layer_id in range(num_dense_layers):
        model.add(Dropout(dropout_rate))
        model.add(TimeDistributed(Dense(dense_units_count[dense_layer_id], activation='relu',
                                        kernel_initializer=KI.RandomNormal(),
                                        bias_initializer=KI.Ones())))

    for blstm_layer_id in range(num_blstm_layers):
        if not no_bidirectional:
            model.add(Bidirectional(LSTM(blstm_unit_counts[blstm_layer_id],
                                         return_sequences=True, stateful=stateful,
                                         unroll=unroll_blstm)))
        else:
            model.add(LSTM(blstm_unit_counts[blstm_layer_id],
                           return_sequences=True, stateful=stateful,
                           unroll=unroll_blstm))

    model.add(TimeDistributed(Dense(num_classes, activation='softmax',
                                    kernel_initializer=KI.RandomNormal(),
                                    bias_initializer=KI.Ones())))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy',
                                                f1_SP,
                                                f1_FIX,
                                                f1_SACC])
    model.summary()
    return model


def extract_windows(X, y, window_length,
                    padding_features=0,
                    downsample=1, temporal_padding=False):
    """
    Extract fixed-sized (@window_length) windows from arbitrary-length sequences (in X and y),
    padding them, if necessary (mirror-padding is used).

    :param X: input data; list of arrays, each shaped like (NUM_SAMPLES, NUM_FEATURES);
              each list item corresponds to one eye tracking recording (one observer & one stimulus clip)
    :param y: corresponding labels; list of arrays, each shaped like (NUM_SAMPLES,);
              each list element corresponds to sample-level eye movement class labels in the respective sequence;
              list elements in X and y are assumed to be matching.
    :param window_length: the length of resulting windows; this is the input "context size" in the paper, in samples.
    :param padding_features: how many extra samples to take in the feature (X) space on each side
                             (resulting X will have sequence length longer than resulting Y, by 2 * padding_features,
                             while Y will have sample length of @window_length);
                             this is necessary due to the use of valid padding in convolution operations in the model;
                             if all convolutions are of size 3, and if they all use valid padding, @padding_features
                             should be set to the number of convolution layers.
    :param downsample: take each @downsample'th window; if equal to @window_length, no overlap between windows;
                       by default, all possible windows with the shift of 1 sample between them will be created,
                       resulting in NUM_SAMPLES-1 overlap; if overlap of K samples is desired, should set
                       @downsample=(NUM_SAMPLES-K)
    :param temporal_padding: whether to pad the entire sequences, so that the first window is centered around the
                             first sample of the real sequence (i.e. the sequence of recorded eye tracking samples);
                             not used
    :return: two lists of numpy arrays:
                (1) a list of windows corresponding to input data (features),
                (2) a list of windows corresponding to labels we will predict.
                These can be used as input to network training procedures, for example.
    """
    res_X = []
    res_Y = []
    # iterate through each file in this subset of videos
    for x_item, y_item in zip_equal(X, y):
        # pad for all windows
        padding_size_x = padding_features
        padding_size_y = 0
        if temporal_padding:
            padding_size_x += window_length // 2
            padding_size_y += window_length // 2

        padded_x = np.pad(x_item, ((padding_size_x, padding_size_x), (0, 0)), 'reflect')
        padded_y = np.pad(y_item, ((padding_size_y, padding_size_y), (0, 0)), 'reflect')

        #
        # Extract all valid windows in @padded_x, with given downsampling and size.
        # @res_X will have windows of size @window_length + 2*@padding_features
        window_length_x = window_length + 2 * padding_features
        res_X += [padded_x[i:i + window_length_x, :] for i in
                  range(0, padded_x.shape[0] - window_length_x + 1, downsample)]
        # @res_Y will have windows of size @window_length, central to the ones in @res_X
        res_Y += [padded_y[i:i + window_length, :] for i in
                  range(0, padded_y.shape[0] - window_length + 1, downsample)]
    return res_X, res_Y


def evaluate_test(model, X, y=None,
                  keys_to_subtract_start_indices=(),
                  correct_for_unknown_class=True,
                  padding_features=0,
                  temporal_padding=False,
                  split_by_items=False):
    """
    Predict (with a trained @model) labels for full sequences in @X and compare those to @y. The function returns both
    the predicted and true labels ("raw" results), and the results of several metrics
    (accuracy, sample-level F1 scores for classes fixation, saccade, pursuit, and noise) - "aggregated" results.
    If @y is not provided, only the predicted per-class probabilities are returned

    The context window width for prediction will be inferred from the @model.

    If predicting on data without labels, set @correct_for_unknown_class to False (see below)! If @y is not passed, it
    will not be used either way, but if a "fake" @y is provided (e.g. filled with UNKNOWN labels), make sure to change
    this option.

    :param model: a trained deep model (has to have a .predict_proba() method)
    :param X: input data; list of arrays, each shaped like (NUM_SAMPLES, NUM_FEATURES);
              each list item corresponds to one eye tracking recording (one observer & one stimulus clip)
    :param y: corresponding labels; needed only to evaluate the model predictions;
              the list of arrays, each shaped like (NUM_SAMPLES,);
              each list element corresponds to sample-level eye movement class labels in the respective sequence;
              list elements in X and y are assumed to be matching.
    :param keys_to_subtract_start_indices: indices of the keys, the starting value of which should be subtracted from
                                           all samples (inside of each window that is passed to the model)
    :param correct_for_unknown_class: if True, will assign "correct" probabilities (1 for class 0, 0 for other classes)
                                      to the samples that have an UNKNOWN label, since we don't want them influencing
                                      the evaluation scores.
    :param padding_features: how many extra samples to take in the feature (X) space on each side
                             (resulting X will have sequence length longer than resulting Y, by 2 * padding_features,
                             while Y will have sample length of @window_length);
                             this is necessary due to the use of valid padding in convolution operations in the model;
                             if all convolutions are of size 3, and if they all use valid padding, @padding_features
                             should be set to the number of convolution layers.
    :param temporal_padding: whether to pad the entire sequences, so that the first window is centered around the
                             first sample of the real sequence (i.e. the sequence of recorded eye tracking samples);
                             False by default
    :param split_by_items: whether to split "raw" results by individual sequences in @X and @y
                          (necessary when want to create output .arff files that match all input .arff files)
    :return: two dictionaries:
                 (1) "raw_results", with keys "pred" (per-class probabilities predicted by the network, for every
                     sample in @X) and "true" (one-hot encoded class labels, for every sample in @X -- taken from @y,
                     if @y was provided).
                     If @split_by_items is set to True, these will be lists of numpy arrays that correspond to
                     *sequences* (not samples!) in @X. Otherwise, these are themselves numpy arrays.
                 (2) "results", with keys "accuracy", "F1-FIX", "F1-SACC", "F1-SP", and "F1-NOISE" (if @y is provided)
    """
    res_Y = [] if y is not None else None
    res_X = []
    if y is None:
        # if no true labels are provided, fake the sequence
        y = [None] * len(X)

    window_length = model.output_shape[1]  # output_shape is (batch_size, NUM_SAMPLES, NUM_CLASSES)
    downsample = window_length  # no overlap here

    # Keep track of where each individual recording (1 observer, 1 stimulus clip) starts and ends -- will need this
    # to produce final automatically labelled .arff files
    items_start = []
    items_end = []

    for x_item, y_item in zip_equal(X, y):
        items_start.append(len(res_X))

        # how much padding is needed additionally, for full-window size
        target_length = int(math.ceil(x_item.shape[0] / float(window_length)) * window_length)
        padding_size_x = [0, target_length - x_item.shape[0]]
        padding_size_y = [0, target_length - y_item.shape[0]] if y_item is not None else [0, 0]

        # pad features (@x_item) for all windows
        padding_size_x = [elem + padding_features for elem in padding_size_x]
        # no @y_item-padding required

        if temporal_padding:
            padding_size_x = [elem + window_length // 2 for elem in padding_size_x]
            padding_size_y = [elem + window_length // 2 for elem in padding_size_y]

        padded_x = np.pad(x_item, (padding_size_x, (0, 0)), 'reflect')  # x is padded with reflections to limit artifacts
        if y_item is not None:
            padded_y = np.pad(y_item, (padding_size_y, (0, 0)), 'constant')  # y is zero-padded to ignore those labels
            # set to UNKNOWN class
            has_no_label_mask = padded_y.sum(axis=1) == 0
            padded_y[has_no_label_mask, 0] = 1

        #
        # Extract all valid windows in @padded_x, with given downsampling and size.
        # @res_X will have windows of size @window_length + 2*@padding_features
        window_length_x = window_length + 2 * padding_features
        res_X += [padded_x[i:i + window_length_x, :] for i in
                  range(0, padded_x.shape[0] - window_length_x + 1, downsample)]
        if y_item is not None:
            # @res_Y will have windows of size @window_length, central to the ones in @res_X
            res_Y += [padded_y[i:i + window_length, :] for i in
                      range(0, padded_y.shape[0] - window_length + 1, downsample)]

        items_end.append(len(res_X))

    res_X = np.array(res_X)
    if res_Y is not None:
        res_Y = np.array(res_Y)

    # Subtract the first value of each feature (for which it is necessary) inside each window
    for col_ind in keys_to_subtract_start_indices:
        res_X[:, :, col_ind] -= res_X[:, 0, col_ind].reshape(-1, 1)

    # augment to fit batch size
    batch_size = model.get_config()['layers'][0]['config']['batch_input_shape'][0]
    original_len = res_X.shape[0]
    target_len = int(np.ceil(float(original_len) / batch_size)) * batch_size
    res_X = np.pad(res_X,
                   pad_width=((0, target_len - res_X.shape[0]), (0, 0), (0, 0)),
                   mode='constant')
    # take only the needed predictions
    res_proba = model.predict(res_X, batch_size=batch_size)[:original_len]

    results = {}

    # The unknown labels in @res_Y should play no role, set respective probabilities in @res_proba:
    # If @y was provided, set "correct" probabilities for the UNKNOWN class, since we cannot properly evaluate against
    # an undefined label
    if correct_for_unknown_class and res_Y is not None:
        unknown_class_mask = res_Y[:, :, 0] == 1
        res_proba[unknown_class_mask] = 0.0
        res_proba[unknown_class_mask, 0] = 1.0
    res = np.argmax(res_proba, axis=-1)

    raw_results = {'true': res_Y, 'pred': res_proba}

    # cannot evaluate unless the labels @y were provided
    if res_Y is not None:
        results['accuracy'] = (np.mean(res[np.logical_not(unknown_class_mask)] ==
                                       np.argmax(res_Y[np.logical_not(unknown_class_mask)], axis=-1)))

        for stat_i, stat_name in zip(list(range(1, 5)), ['FIX', 'SACC', 'SP', 'NOISE']):
            results['F1-{}'.format(stat_name)] = K.eval(categorical_f1_score_for_class(res_Y, res_proba, stat_i,
                                                                                       'float64'))

    if split_by_items:
        # split into individual sequences
        split_res_true = [] if res_Y is not None else None
        split_res_pred = []
        for individual_start, individual_end in zip_equal(items_start, items_end):
            if res_Y is not None:
                split_res_true.append(raw_results['true'][individual_start:individual_end])
            split_res_pred.append(raw_results['pred'][individual_start:individual_end])
        raw_results = {'true': split_res_true, 'pred': split_res_pred}

    return raw_results, results


def get_architecture_descriptor(args):
    """
    Generate the descriptor of the model
    :param args: command line arguments
    :return: descriptor string
    """
    # (convert lists to tuples to avoid [] in the descriptor, which later confuse glob.glob())
    return '{numC}x{padC}C@{unitsC}_{numD}xD@{unitsD}_{numB}x{typeB}@{unitsB}'.format(
        numC=args.num_conv, padC=args.conv_padding_mode[0], unitsC=tuple(args.conv_units)[:args.num_conv],
        numD=args.num_dense, unitsD=tuple(args.dense_units)[:args.num_dense],
        numB=args.num_blstm, typeB=('B' if not args.no_bidirectional else 'L'),
        unitsB=tuple(args.blstm_units)[:args.num_blstm])


def get_feature_descriptor(args):
    """
    Generate the descriptor of the feature set that is used
    :param args: command line arguments
    :return: descriptor string
    """
    feature_postfix = []
    features_to_name = args.features[:]  # copy list
    naming_priority = ['movement', 'speed', 'acc', 'direction']
    naming_priority += sorted(set(args.features).difference(naming_priority))  # add the rest in alphabetical order
    for n in naming_priority:
        if n == 'movement':
            # to add "movement" to the descriptor, check that all 3 parts of these features are present
            if 'speed' in features_to_name and \
                            'acc' in features_to_name and \
                            'direction' in features_to_name:
                feature_postfix.append('movement')

                features_to_name.remove('speed')
                features_to_name.remove('acc')
                features_to_name.remove('direction')
            else:
                continue
        if n in features_to_name:
            feature_postfix.append(n)
            features_to_name.remove(n)
    feature_postfix = '_'.join(feature_postfix)

    # if we want to limit the number of temporal scales of the features, mark it in the signature
    if args.num_feature_scales < 5:
        feature_postfix += '_{}_temp_scales'.format(args.num_feature_scales)

    return feature_postfix


def get_full_model_descriptor(args):
    """
    Get a full descriptor of the model, which includes the architecture descriptor, feature descriptor, and info
     about context window size and overlap.
    :param args:
    :return:
    """
    return '{mdl}_{feat}_WINDOW_{win}{overlap}/'.format(mdl=get_architecture_descriptor(args),
                                                        feat=get_feature_descriptor(args),
                                                        win=args.window_size,
                                                        overlap='_overlap_{}'.format(args.overlap)
                                                                if args.overlap > 0 else '')


def get_arff_attributes_to_keep(args):
    keys_to_keep = []
    if 'xy' in args.features:
        keys_to_keep += ['x', 'y']

    if 'speed' in args.features:
        keys_to_keep += ['speed_{}'.format(i) for i in (1, 2, 4, 8, 16)[:args.num_feature_scales]]
    if 'direction' in args.features:
        keys_to_keep += ['direction_{}'.format(i) for i in (1, 2, 4, 8, 16)[:args.num_feature_scales]]
    if 'acc' in args.features:
        keys_to_keep += ['acceleration_{}'.format(i) for i in (1, 2, 4, 8, 16)[:args.num_feature_scales]]

    return keys_to_keep


def run(args):
    """
    Run model training/testing, depending on @args. See description of parse_args() for more information!
    :param args: terminal arguments to the program. See parse_args() help, or run with --help.
    :return: if args.dry_run is set to True, returns a model (created or loaded)
    """

    # For every numeric label we will need a categorical (human-readable) value to easier interpret results.
    CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE = {v: k for k, v in
                                                       CORRESPONDENCE_TO_HAND_LABELLING_VALUES.items()}
    print(CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE)
    num_classes = 5  # 0 = UNKNOWN, 1 = FIX, 2 = SP, 3 = SACC, 4 = NOISE

    # Paths, where to store models and generated outputs (.arff files with sample-level labels).
    # These are just the "root" folders for this data, subfolders will be automatically created
    # for each model configuration.
    #
    # Now these are stored locally, but if distributed training is desired, make sure that especially MODELS_DIR
    # points to a shared location that is accessible for reading and writing to all training nodes!
    # Pass the appropriate --model-root-path argument!
    MODELS_DIR = 'data/models/' if args.model_root_path is None else args.model_root_path
    OUT_DIR = 'data/outputs/'

    # by default, do training; --final-run will initiate testing mode
    TRAINING_MODE = True
    if args.final_run:
        TRAINING_MODE = False

    # NB: Some training parameters are set for the GazeCom data set! If you wish to *train* this model on
    # another data set, you will need to adjust these:
    #   - CLEAN_TIME_LIMIT -- during training, all samples (for all clips) with a timestamp exceeding this threshold
    #                         will be disregarded. Set LOAD_CLEAN_DATA to False instead of "=TRAINING_MODE", if this is
    #                         not desired.
    #   - certain .arff column names ("handlabeller_final" for the ground truth labels, feature names - e.g. "speed_1")
    #       would have to be adjusted, if respective data is located in other columns
    #   - keys_to_subtract_start -- if you add more features that need to be zeroed in the beginning of each window
    #                               that is given to the network, you need to add them to this set; by default, only
    #                               features "x" and "y" are treated this way.
    #   - num_classes above, if more classes are labelled in the data set (5 by default, 0 = UNKNOWN, the rest are
    #                 actual labels (1 = FIXATION, 2 = SACCADE, 3 = SMOOTH PURSUIT, 4 = NOISE);
    #                 also, CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE will have to be updated then!
    #   - time_scale -- in GazeCom, time is given in microseconds, so it has to be multiplied by 1e-6 to convert to
    #                   seconds; either convert your data set's time to microseconds as well, or change the @time_scale

    # If you only want to test a pre-trained model on a new data set, you have to convert the data set to .arff file
    # format with the fields 'time', 'x', 'y', and maybe 'handlabeller_final', if manual annotations are available.
    # These fields will be preserved in final outputs. Also you have to provide fields that correspond to all features
    # that the model utilises (e.g. for speed and direction, feature names are speed_1, speed_2, speed_4, speed_8,
    # and speed_16 (speed in degrees of visual angle per second, extracted at corresponding scales, in samples),
    # same for direction (in radians relative to the horizontal vector from left to right). See the `feature_extraction`
    # folder and its scripts for implementation details, as well as the corresponding paper.

    CLEAN_TIME_LIMIT = 21 * 1e6  # in microseconds; 21 seconds

    # During training, some data purification is performed ("clean" data is loaded):
    #   - samples after CLEAN_TIME_LIMIT microseconds are ignored
    #   - files SSK_*.arff are re-sampled by taking every second gaze sample
    #       (in GazeCom, these contain 500Hz recordings, compared to 250Hz of the rest of the data set)
    # Set LOAD_CLEAN_DATA to False, if this is not desired.
    LOAD_CLEAN_DATA = TRAINING_MODE
    if LOAD_CLEAN_DATA:
        print('Loading clean data')
    else:
        print('Loading raw data')

    print('Feature description:', get_feature_descriptor(args))
    print('Architecture description:', get_architecture_descriptor(args))

    # Where to find feature files. "{}" in the template is where a clip name will be inserted.
    # The "root" folder is by default 'data/inputs/GazeCom_all_features'
    files_template = args.feature_files_folder + '/{}/*.arff'

    # When extracting window of fixed size with fixed overlap, this formula defines the downsampling factor
    # (i.e. will take every @downsample'th window of @args.window_size in length, which will result in overlap of
    # exactly @args.overlap samples between subsequent windows)
    downsample = args.window_size - args.overlap
    # All convolutions are of size 3, so each Conv1D layer (with valid padding) requires 1 sample padded on each side.
    padding_features = args.num_conv if args.conv_padding_mode == 'valid' else 0

    # Set paths to model files and output files' directory, which include model parameters,
    # such as feature signatures, window size, overlap size.
    MODEL_PATH = '{dir}/LOO_{descr}/'.format(dir=MODELS_DIR,
                                             descr=get_full_model_descriptor(args))
    OUT_PATH = '{dir}/output_LOO_{descr}/'.format(dir=OUT_DIR,
                                                  descr=get_full_model_descriptor(args))
    # The record .model_name in @args overrides the MODEL_PATH set above.
    if args.model_name is not None:
        MODEL_PATH = '{}/{}/'.format(MODELS_DIR, args.model_name)

    print('Selected model path:', MODEL_PATH)
    if not args.dry_run:
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)

    # Load all pre-computed video parameters just to get video names (for cross-video validation, LOVO)
    # Only need the 'video_names' key in the dictionary in this json file (in case it needs to be adjusted for a
    # different data set).
    gc = json.load(open('data/inputs/GazeCom_video_parameters.json'))
    all_video_names = gc['video_names']

    # To clear the used memory, we will normally run this script as many times as there are different clips in GazeCom.
    # Not to load and pre-process raw input sequences this often, we dump them as an .h5 file, or load it, if it
    # already exists.
    raw_data_set_fname = 'data/cached/GazeCom_data_{feat}{is_clean}.h5'.format(feat=get_feature_descriptor(args),
                                                                               is_clean='' if LOAD_CLEAN_DATA
                                                                                           else '_not_clean')

    # During a final run, we want to create output files that have fewer fields: just the ones describing the raw data
    # (no features) and the ground truth label.
    if args.final_run:
        source_fnames = []
        source_objs = []
        source_keys_to_keep = ['time', 'x', 'y', 'confidence', 'handlabeller_final']

    # Depending on the command line arguments, certain keys are used as features (@keys_to_keep here)
    keys_to_keep = get_arff_attributes_to_keep(args)

    print('Using the following features:', keys_to_keep)

    data_X = []
    data_Y = []
    data_Y_one_hot = []
    # If no file with pre-processed features and labels exists yet, create it.
    if not os.path.exists(raw_data_set_fname):
        if not os.path.exists(os.path.split(raw_data_set_fname)[0]):
            os.makedirs(os.path.split(raw_data_set_fname)[0])

        # Will convert to degrees the following keys: x, y, and all speed and acceleration features.
        keys_to_convert_to_degrees = ['x', 'y'] + [k for k in keys_to_keep if 'speed_' in k or 'acceleration_' in k]
        keys_to_convert_to_degrees = sorted(set(keys_to_convert_to_degrees).intersection(keys_to_keep))
        # Conversion is carried out by dividing by pixels-per-degree value (PPD)
        print('Will divide by PPD the following keys', keys_to_convert_to_degrees)

        time_scale = 1e-6  # time is originally in microseconds; scale to seconds

        total_files = 0
        for video_name in gc['video_names']:
            print('For {} using files from {}'.format(video_name, files_template.format(video_name)))
            fnames = sorted(glob.glob(files_template.format(video_name)))
            total_files += len(fnames)

            data_X.append([])
            data_Y.append([])
            data_Y_one_hot.append([])
            if args.final_run:
                source_fnames.append(list(fnames))
                source_objs.append([])

            for f in fnames:
                o = ArffHelper.load(open(f))
                if LOAD_CLEAN_DATA and 'SSK_' in f:
                    # the one observer with 500HZ instead of 250Hz
                    o['data'] = o['data'][::2]
                if LOAD_CLEAN_DATA:
                    o['data'] = o['data'][o['data']['time'] <= CLEAN_TIME_LIMIT]

                if args.final_run:  # make a copy of the data before any further modifications! record to @source_objs
                    source_obj = deepcopy(o)
                    # only the essential keys in array
                    source_obj['data'] = source_obj['data'][source_keys_to_keep]
                    # in attributes too
                    attribute_names = [n for n, _ in source_obj['attributes']]
                    source_obj['attributes'] = [source_obj['attributes'][attribute_names.index(attr)]
                                                for attr in source_keys_to_keep]
                    source_objs[-1].append(source_obj)

                # normalize coordinates in @o by dividing by @ppd_f -- the pixels-per-degree value of this file @f
                ppd_f = sp_util.calculate_ppd(o)
                for k in keys_to_convert_to_degrees:
                    o['data'][k] /= ppd_f

                # add to respective data sets (only the features to be used and the true labels)
                data_X[-1].append(np.hstack([np.reshape(o['data'][key], (-1, 1)) for key in keys_to_keep]).astype(np.float64))
                assert data_X[-1][-1].dtype == np.float64
                if 'time' in keys_to_keep:
                    data_X[-1][-1][:, keys_to_keep.index('time')] *= time_scale
                data_Y[-1].append(o['data']['handlabeller_final'])  # "true" labels
                data_Y_one_hot[-1].append(np.eye(num_classes)[data_Y[-1][-1]])  # convert numeric labels to one-hot form

        if total_files > 0:
            print('Loaded a total of {} files'.format(total_files))
        else:
            raise ValueError('No input files found! Check that the directory "{}" exists '
                             'and is accessible for reading, or provide a different value for the '
                             '--feature-files-folder argument. Make sure you extracted the respective archive, '
                             'if the data was provided in this way. This folder must contain '
                             '18 subfolders with names corresponding to '
                             'GazeCom clip names.'.format(args.feature_files_folder))

        # As mentioned above, preserve the pre-processed data to not re-do this again, at least on the same system.
        # This creates files that are dependent of the features that are preserved and whether @LOAD_CLEAN_DATA is set.
        if not args.final_run:
            pickle.dump({'data_X': data_X, 'data_Y': data_Y, 'data_Y_one_hot': data_Y_one_hot},
                        open(raw_data_set_fname, 'w'))
        else:
            pickle.dump({'data_X': data_X, 'data_Y': data_Y, 'data_Y_one_hot': data_Y_one_hot,
                         'source_fnames': source_fnames, 'source_objs': source_objs},
                        open(raw_data_set_fname, 'w'))
        print('Written to', raw_data_set_fname)
    else:
        # If the raw file already exists, just load it
        print('Loading from', raw_data_set_fname)
        loaded_data = pickle.load(open(raw_data_set_fname))
        data_X, data_Y, data_Y_one_hot = loaded_data['data_X'], loaded_data['data_Y'], loaded_data['data_Y_one_hot']
        if args.final_run:
            source_fnames, source_objs = loaded_data['source_fnames'], loaded_data['source_objs']

    # Each record of the @windows_X and @windows_Y lists corresponds to one "fold" of the used cross-validation
    # procedure
    windows_X = []
    windows_Y = []

    # Will subtract the initial value from the following keys (only the changes in these keys should matter),
    # not to overfit, for example, for spatial location of the eye movement: x and y coordinates.
    keys_to_subtract_start = sorted({'x', 'y'}.intersection(keys_to_keep))
    print('Will subtract the starting values of the following features:', keys_to_subtract_start)
    keys_to_subtract_start_indices = [i for i, key in enumerate(keys_to_keep) if key in keys_to_subtract_start]

    for subset_index in range(len(data_X)):
        x, y = extract_windows(data_X[subset_index], data_Y_one_hot[subset_index],
                               window_length=args.window_size,
                               downsample=downsample,
                               padding_features=padding_features)
        windows_X.append(np.array(x))
        windows_Y.append(np.array(y))

        # Subtract the first value of each feature inside each window
        for col_ind in keys_to_subtract_start_indices:
            windows_X[-1][:, :, col_ind] -= windows_X[-1][:, 0, col_ind].reshape(-1, 1)

    # TensorBoard logs directory to supervise model training
    log_root_dir = 'data/tensorboard_logs/{}'
    logname = 'logs_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M"),
                                  os.path.split(MODEL_PATH.rstrip('/'))[1])  # add model description to filename

    callbacks_list = [History(), TensorBoard(batch_size=args.batch_size, log_dir=log_root_dir.format(logname),
                                             write_images=False)]  # , histogram_freq=100, write_grads=True)]

    # All results are stored as lists - one element for one left-out group (one fold of the cross-validation,
    # in this case)
    results = {
        'accuracy': [],
        'F1-SP': [],
        'F1-FIX': [],
        'F1-SACC': [],
        'F1-NOISE': []
    }
    raw_results = {'true': [], 'pred': []}
    training_histories = []  # histories for all folds

    # LOVO-CV (Leave-One-Video-Out; Leave-n-Observers-Out overestimates model performance, see paper)
    num_training_runs = 0  # count the actual training runs, in case we need to stop after just one
                           # (if @args.run_once == True)
    # iterate through stimulus video clips, leaving each one out in turn
    for i, video_name in enumerate(all_video_names):
        if args.run_once and (args.run_once_video is not None and video_name != args.run_once_video):
            # --run-once is enabled and the --run-once-video argument has been passed. If this video name does not
            # match the @video_name, skip it
            continue
        # Check if final (trained) model already exists.
        # Thanks to creating an empty file in the "else" clause below, can run several training procedures at the same
        # time (preferably on different machines or docker instances, with MODEL_PATH set to some location that is
        # accessible to all the machines/etc.) - the other training procedures will skip training the model for this
        # particular fold of the cross-validation procedure.
        model_fname = MODEL_PATH + '/Conv_sample_windows_epochs_{}_without_{}.h5'.format(args.num_epochs, video_name)
        if os.path.exists(model_fname):
            ALREADY_TRAINED = True
            print('Skipped training, file in', model_fname, 'exists')
            if not args.final_run:
                # if no need to generate output .arff files, just skip this cross-validation fold
                continue
        else:
            # have not begun training yet
            ALREADY_TRAINED = False
            print('Creating an empty file in {}'.format(model_fname))
            os.system('touch "{}"'.format(model_fname))

        if not ALREADY_TRAINED:
            # need to train the model

            r = np.random.RandomState(0)
            # here, we ignore the "left out" video clip @i, but aggregate over all others
            train_set_len = sum([len(windows_X[j]) for j in range(len(windows_X)) if j != i])
            print('Total amount of windows:', train_set_len)
            # will shuffle all samples according to this permutation, and will keep only the necessary amount
            # (by default, 50,000 for training)
            perm = r.permutation(train_set_len)[:args.num_training_samples]
            train_X = []
            train_Y = []
            indices_range_low = 0
            for j in range(len(windows_X)):
                if j == i:
                    continue  # skip the test set
                indices_range_high = indices_range_low + len(windows_X[j])
                # permutation indices [indices_range_low; indices_range_high) are referring
                # to the windows in @windows_X[j]
                local_indices = perm[(perm >= indices_range_low) * (perm < indices_range_high)]
                # re-map in "global" indices (inside @perm) onto "local" indices (inside @windows_X[j])
                local_indices -= indices_range_low
                train_X.append(windows_X[j][local_indices])
                train_Y.append(windows_Y[j][local_indices])

                indices_range_low = indices_range_high

            # assemble the entire training set
            train_X = np.concatenate(train_X)
            train_Y = np.concatenate(train_Y)

            # if fine-tuning, load a pre-trained model; if not - create a model from scratch
            if args.initial_epoch:
                # We have to pass our metrics (f1_SP and so on) as custom_objects here and below, since it won't load
                # otherwise
                model = keras.models.load_model(MODEL_PATH + '/Conv_sample_windows_epochs_{}_without_{}.h5'.format(args.initial_epoch, video_name),
                                                custom_objects={'f1_SP': f1_SP, 'f1_SACC': f1_SACC, 'f1_FIX': f1_FIX})
                print('Loaded model from', MODEL_PATH + '/Conv_sample_windows_epochs_{}_without_{}.h5'.format(args.initial_epoch, video_name))
            else:
                model = create_model(num_classes=num_classes, batch_size=args.batch_size,
                                     train_data_shape=train_X.shape,
                                     dropout_rate=0.3,
                                     padding_mode=args.conv_padding_mode,
                                     num_conv_layers=args.num_conv, conv_filter_counts=args.conv_units,
                                     num_dense_layers=args.num_dense, dense_units_count=args.dense_units,
                                     num_blstm_layers=args.num_blstm, blstm_unit_counts=args.blstm_units,
                                     unroll_blstm=False,
                                     no_bidirectional=args.no_bidirectional)
        else:
            model = keras.models.load_model(model_fname, custom_objects={'f1_SP': f1_SP,
                                                                         'f1_SACC': f1_SACC,
                                                                         'f1_FIX': f1_FIX})
            print('Skipped training, loaded model from', model_fname)

        if args.dry_run:
            return model

        # need to run training now?
        if TRAINING_MODE and not ALREADY_TRAINED:
            # Store model training history. Make sure (again?) that training is always performed with
            # @num_training_samples sequences
            assert train_X.shape[0] == train_Y.shape[0]
            assert train_X.shape[0] == args.num_training_samples, 'Not enough training samples (might need to ' \
                                                                  'increase --overlap, decrease --window, or ' \
                                                                  'decrease --num-training-samples in the command ' \
                                                                  'line arguments.'

            training_histories.append(model.fit(train_X, train_Y,
                                      epochs=args.num_epochs,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      callbacks=callbacks_list,
                                      validation_split=0.1,
                                      verbose=1,
                                      initial_epoch=args.initial_epoch))
            model.save(model_fname)
            # Completed a cross-validation fold with actual training
            num_training_runs += 1

        # Test model on the left-out cross-validation group (@i is the index corresponding to the test set on this run).
        # @raw will contain predicated class probabilities as well as true labels.
        # @processed will contain performance statistics of this fold
        raw, processed = evaluate_test(model, data_X[i], data_Y_one_hot[i],
                                       keys_to_subtract_start_indices=keys_to_subtract_start_indices,
                                       padding_features=padding_features,
                                       split_by_items=args.final_run)  # keep sequences of different observers separate,
                                                                       # if we still need to write output .arff files

        # store all results (raw and processed)
        for k in list(raw_results.keys()):
            if args.final_run:
                # On the "final" run, @raw[k] is split into several sequences each, need to concatenate here
                # (in order to maintain the same format of the .pickle files)
                raw_results[k].append(np.concatenate(raw[k], axis=0))
            else:
                raw_results[k].append(raw[k])
        for k in list(results.keys()):
            results[k].append(processed[k])

        print('Evaluating for video', video_name)
        for stat_name in ['FIX', 'SACC', 'SP', 'NOISE']:
            print('F1-{}'.format(stat_name), results['F1-{}'.format(stat_name)][-1])

        if args.final_run and args.output_folder is not None:
            if args.output_folder == 'auto':
                args.output_folder = OUT_PATH
            print('Creating the detection outputs in', args.output_folder)
            # Generate actual ARFF outputs
            # Iterate through file names, original objects (from input .arff's), ground truth labels,
            # and predicted labels:
            for source_fname, source_obj, labels_true, labels_pred in \
                    zip_equal(source_fnames[i], source_objs[i],
                              raw['true'], raw['pred']):
                full_folder, suffix = os.path.split(source_fname)
                folder_name = os.path.split(full_folder)[1]  # subfolder name = video name
                out_fname = '{}/{}/{}'.format(args.output_folder, folder_name, suffix)

                # create folders that might not exist yet
                if not os.path.exists(args.output_folder):
                    os.mkdir(args.output_folder)
                if not os.path.exists('{}/{}'.format(args.output_folder, folder_name)):
                    os.mkdir('{}/{}'.format(args.output_folder, folder_name))

                # get labels from probabilities for each label
                labels_true = np.argmax(labels_true, axis=-1)
                labels_pred = np.argmax(labels_pred, axis=-1)
                known_class_mask = labels_true != 0  # in case there were some unassigned labels in the ground truth
                labels_true = labels_true[known_class_mask]
                labels_pred = labels_pred[known_class_mask]

                # sanity check: "true" labels must match the "handlabeller_final" column of the input .arff files
                assert (labels_true == source_obj['data']['handlabeller_final']).all()

                # add a column containing predicted labels
                source_obj = ArffHelper.add_column(source_obj,
                                                   name=sp_processor.EM_TYPE_ATTRIBUTE_NAME,
                                                   dtype=sp_processor.EM_TYPE_ARFF_DATA_TYPE,
                                                   default_value=sp_processor.EM_TYPE_DEFAULT_VALUE)
                # fill in with categorical values instead of numerical ones
                # (use @CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE for conversion)
                source_obj['data'][sp_processor.EM_TYPE_ATTRIBUTE_NAME] = \
                    [CORRESPONDENCE_TO_HAND_LABELLING_VALUES_REVERSE[x] for x in labels_pred]

                ArffHelper.dump(source_obj, open(out_fname, 'w'))

        if args.run_once and num_training_runs >= 1:
            break

    # get statistics over all splits that are already processed
    if not ALREADY_TRAINED or args.final_run:

        raw_results['true'] = np.concatenate(raw_results['true'])
        raw_results['pred'] = np.concatenate(raw_results['pred'])

        mask = np.argmax(raw_results['true'], axis=-1) != 0
        print('Found', np.logical_not(mask).sum(), 'UNKNOWN samples in the raw ``true\'\' predictions ' \
                                                   '(including the artificially padded parts of the last windows ' \
                                                   'in each sequence, in order to match window width)')

        print(raw_results['true'].shape, raw_results['pred'].shape)

        unknown_class_mask = raw_results['true'][:, :, 0] == 1  # count "unknown"s in the one-hot-encoded true labels

        print('Overall classification scores per class:')
        for stat_i, stat_name in zip(list(range(1, 5)), ['FIX', 'SACC', 'SP', 'NOISE']):
            results['overall-F1-{}'.format(stat_name)] = K.eval(categorical_f1_score_for_class(raw_results['true'],
                                                                                               raw_results['pred'],
                                                                                               stat_i,
                                                                                               'float64'))
            print('F1-{}'.format(stat_name), results['overall-F1-{}'.format(stat_name)])

        results['overall-acc'] = np.mean(np.argmax(raw_results['true'][np.logical_not(unknown_class_mask)], axis=-1) ==
                                         np.argmax(raw_results['pred'][np.logical_not(unknown_class_mask)], axis=-1))

        # how many samples did the network leave "UNKNOWN"
        results['overall-UNKNOWN-samples'] = (np.argmax(raw_results['pred'], axis=-1) == 0).sum()
        print('Sample left UNKNOWN:', results['overall-UNKNOWN-samples'], '(including the UNKNOWN samples matching ' \
                                                                          'the window-padded ``true\'\' labels from ' \
                                                                          'above)')

        # Run the full evaluation.
        # Need a downloaded and installed sp_tool package for this! See http://michaeldorr.de/smoothpursuit/sp_tool.zip
        # See README for more information on how to do this.
        if args.final_run and args.output_folder:
            print('Running sp_tool eval --> {out_dir}/eval.json'.format(out_dir=args.output_folder))
            cmd = 'python {sp_tool_dir}/examples/evaluate_on_gazecom.py ' \
                  '--in "{out_dir}" ' \
                  '--hand "{gt_dir}" ' \
                  '--pretty > ' \
                  '"{out_dir}/eval.json"'.format(sp_tool_dir=args.sp_tool_folder,
                                                 out_dir=args.output_folder,
                                                 gt_dir=args.ground_truth_folder)
            print('Running command:\n', cmd)

            if os.path.exists('{}/examples/'.format(args.sp_tool_folder)):
                cmd_res = os.system(cmd)
                if cmd_res != 0:
                    print('Something failed during the sp_tool evaluation run. Check the command above and run it ' \
                          'manually, if necessary! Make sure you also *installed* the sp_tool framework, not just ' \
                          'downloaded it (see sp_tool/README for details). Also, make sure both the output folder ' \
                          '"{}" and the ground truth folder "{}" exist (e.g. were extracted from the respective ' \
                          'archives).'.format(args.output_folder, args.ground_truth_folder))

            else:
                print('\n\nCould not run final evaluation! sp_tool folder could not be found in', args.sp_tool_folder)
                print('Pass the --sp-tool-folder argument that points to the correct location (relative or absolute) ' \
                      'of the sp_tool folder, or download the full deep_eye_movement_classification.zip archive from' \
                      'http://michaeldorr.de/smoothpursuit again.')


def parse_args(dry_run=False):
    """
    Parse command line arguments, or just create the parser (if @dry_run == True)
    :param dry_run: if True, will return the argument parser itself, and not the parsed arguments
    :return: either parser arguments' Namespace object (if @dry_run == False),
             or the argument parser itself (if @dry_run == True)
    """
    parser = ArgumentParser('Sequence-to-sequence eye movement classification')
    parser.add_argument('--final-run', '--final', '-f', dest='final_run', action='store_true',
                        help='Final run with testing only, but on full data (not "clean" data).')
    parser.add_argument('--folder', '--output-folder', dest='output_folder', default=None,
                        help='Only for --final-run: write prediction results as ARFF files here.'
                             'Can be set to "auto" to select automatically.')
    parser.add_argument('--model-root-path', default='data/models',
                        help='The path which will contain all trained models. If you are running the testing, set this '
                             'argument to the same value as was used during training, so that the models can be '
                             'automatically detected.')

    parser.add_argument('--feature-files-folder', '--feature-folder', '--feat-folder',
                        default='data/inputs/GazeCom_all_features/',
                        help='Folder containing the files with already extracted features.')
    parser.add_argument('--ground-truth-folder', '--gt-folder', default='data/inputs/GazeCom_ground_truth/',
                        help='Folder containing the ground truth.')

    parser.add_argument('--run-once', '--once', '-o', dest='run_once', action='store_true',
                        help='Run one step of the LOO-CV run-through and exit (helps with memory consumption,'
                             'then run manually 18 times for GazeCom, for 18 videos).')
    parser.add_argument('--run-once-video', default=None,
                        help='Run one step of the LOO-CV run-through on the video with *this* name and exit. '
                             'Used for partial testing of the models.')

    parser.add_argument('--batch-size', dest='batch_size', default=5000, type=int,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', '--epochs', dest='num_epochs', default=1000, type=int,
                        help='Number of epochs')
    parser.add_argument('--initial-epoch', dest='initial_epoch', default=0, type=int,
                        help='Start training from this epoch')
    parser.add_argument('--training-samples', dest='num_training_samples', default=50000, type=int,
                        help='Number of training samples. The default value is appropriate for windows of 65 samples, '
                             'no overlap between the windows. If window size is increased, need to set --overlap to '
                             'something greater than zero, ideally - maintain a similar number of windows. For example,'
                             ' if we use windows of size 257, we set overlap to (257-65) = 192, so that there would be '
                             'as many windows as with a window size of 65, but without overlap.\n\n'
                             'If you decide to increase the number of training samples, you will most likely have to '
                             'adjust --window-size and --overlap values!')
    parser.add_argument('--window-size', '--window', dest='window_size', default=65, type=int,
                        help='Window size for classifying')
    parser.add_argument('--window-overlap', '--overlap', dest='overlap', default=0, type=int,
                        help='Windows overlap for training data generation')

    parser.add_argument('--model-name', '--model', dest='model_name', default=None,
                        help='Model name. This allows for naming your models as you wish, BUT '
                             'it will override the model and feature descriptors that are included in the '
                             'automatically generated model names, so use with caution!')

    parser.add_argument('--features', '--feat', choices=['movement',  # = speed + direction + acceleration
                                                         'speed', 'acc', 'direction',
                                                         'xy'],
                        nargs='+', default=['speed', 'direction'],
                        help='All of the features that are to be used, can be listed without separators, e.g. '
                             '"--features speed direction". "acc" stands for acceleration; "movement" is a combination '
                             'of all movement features -- speed, direction, acceleration.')

    parser.add_argument('--num-feature-scales', type=int, default=5,
                        help='Number of temporal scales for speed/direction/acceleration features (max is 5, min is 1).'
                             ' Recommended to leave default.'
                             ' The actual scales are 1, 2, 4, 8, and 16 samples, and the first @num_feature_scales '
                             'from this list will be used.')

    parser.add_argument('--num-conv', default=3, type=int,
                        help='Number of convolutional layers before dense layers and BLSTM')
    parser.add_argument('--num-dense', default=1, type=int,
                        help='Number of dense layers before BLSTM')
    parser.add_argument('--num-blstm', default=1, type=int,
                        help='Number of BLSTM layers')

    parser.add_argument('--conv-padding-mode', default='valid', choices=['valid', 'same'],
                        help='Conv1D padding type (applied to all convolutional layers)')

    parser.add_argument('--conv-units', nargs='+', default=[32, 16, 8], type=int,
                        help='Number of filters in respective 1D convolutional layers. If not enough is provided '
                             'for all the layers, the last layer\'s number of filters will be re-used. '
                             'Should pass as, for example, "--conv-units 32 16 8 4".')
    parser.add_argument('--dense-units', nargs='+', default=[32], type=int,
                        help='Number of units in the dense layer (before BLSTM). If not enough is provided '
                             'for all the layers, the last layer\'s number of units will be re-used.'
                             'Should pass as, for example, "--dense-units 32 32".')
    parser.add_argument('--blstm-units', nargs='+', default=[16], type=int,
                        help='Number of units in BLSTM layers. If not enough is provided '
                             'for all the layers, the last layer\'s number of units will be re-used.'
                             'Should pass as, for example, "--blstm-units 16 16 16".')
    parser.add_argument('--no-bidirectional', '--no-bi', action='store_true',
                        help='Use conventional LSTMs, no bi-directional wrappers.')

    parser.add_argument('--dry-run', action='store_true',
                        help='Do not train or test anything, just create a model and show the architecture and '
                             'number of trainable parameters.')

    parser.add_argument('--sp-tool-folder', default='../sp_tool/',
                        help='Folder containing the sp_tool framework. Can be downloaded and installed from '
                             'http://michaeldorr.de/smoothpursuit/sp_tool.zip as a stand-alone package, or as '
                             'part of the deep_eye_movement_classification.zip archive via '
                             'http://michaeldorr.de/smoothpursuit.')

    if dry_run:
        return parser

    args = parser.parse_args()
    if 'movement' in args.features:
        args.features.remove('movement')
        args.features += ['speed', 'acc', 'direction']
    args.features = list(sorted(set(args.features)))

    if not 1 <= args.num_feature_scales <= 5:
        raise ValueError('--num-feature-scales can be between 1 and 5')

    if len(args.conv_units) < args.num_conv:
        args.conv_units += [args.conv_units[-1]] * (args.num_conv - len(args.conv_units))
        warnings.warn('Not enough --conv-units passed, repeating the last one. Resulting filter '
                      'counts: {}'.format(args.conv_units))
    if len(args.dense_units) < args.num_dense:
        args.dense_units += [args.dense_units[-1]] * (args.num_dense - len(args.dense_units))
        warnings.warn('Not enough --dense-units passed, repeating the last one. Resulting filter '
                      'counts: {}'.format(args.dense_units))
    if len(args.blstm_units) < args.num_blstm:
        args.blstm_units += [args.blstm_units[-1]] * (args.num_blstm - len(args.blstm_units))
        warnings.warn('Not enough --blstm-units passed, repeating the last one. Resulting unit '
                      'counts: {}'.format(args.blstm_units))

    return args


def __main__():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    __main__()
