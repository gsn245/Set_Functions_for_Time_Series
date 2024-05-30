"""Utility functions for training and evaluation."""
import math
import random

from collections.abc import Sequence

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import seft.models
from tensorflow.data.experimental import AUTOTUNE
import tensorboard.plugins.hparams.api as hp
from tensorboard.plugins.hparams import api_pb2
from .normalization import Normalizer
import tqdm
import os

#from seft.medical_ts_datasets_radv import medical_ts_datasets
import medical_ts_datasets

get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types
make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator


def build_hyperparameter_metrics(evaluation_metrics):
    metrics = []
    # Add training metrics
    metrics.append(hp.Metric(
        'batch_loss', display_name='loss (train)'))
    metrics.append(hp.Metric(
        'batch_acc', display_name='accuracy (train)'))

    # Add validation metrics
    metrics.append(hp.Metric(
        'epoch_val_loss', display_name='loss (validation)'))
    metrics.append(hp.Metric(
        'epoch_val_acc', display_name='accuracy (validation)'))
    metrics.append(hp.Metric(
        'best_val_loss', display_name='best loss (validation)'))
    metrics.append(hp.Metric(
        'best_val_acc', display_name='best accuracy (validation)'))

    # Add evaluation metrics
    metrics.extend(
        [
            hp.Metric(
                'epoch_val_' + metric, display_name=metric + ' (validation)')
            for metric in evaluation_metrics
        ]
    )
    return metrics


def init_hyperparam_space(logdir, hparams, metrics):
    # Add dataset and model as hyperparameters
    hparams = [
        hp.HParam('dataset', hp.Discrete(medical_ts_datasets.builders)),
        hp.HParam('model', hp.Discrete(seft.models.__all__))
    ] + list(hparams)
    sess = tf.compat.v1.keras.backend.get_session()
    with tf.compat.v2.summary.create_file_writer(logdir).as_default() as w:
        # add none operation to graph
        sess.run(w.init())
        sess.run(hp.hparams_config(hparams=hparams, metrics=metrics))
        sess.run(w.flush())


def get_padding_values(input_dataset_types, label_padding=-100):
    """Get a tensor of padding values fitting input_dataset_types.

    Here we pad everything with 0. and the labels with `label_padding`. This
    allows us to be able to recognize them later during the evaluation, even
    when the values have already been padded into batches.

    Args:
        tensor_shapes: Nested structure of tensor shapes.

    Returns:
        Nested structure of padding values where all are 0 except teh one
        corresponding to tensor_shapes[1], which is padded according to the
        `label_padding` value.

    """
    def map_to_zero(dtypes):
        if isinstance(dtypes, Sequence):
            return tuple((map_to_zero(d) for d in dtypes))
        return tf.cast(0., dtypes)

    def map_to_label_padding(dtypes):
        if isinstance(dtypes, Sequence):
            return tuple((map_to_zero(d) for d in dtypes))
        return tf.cast(label_padding, dtypes)

    if len(input_dataset_types) == 2:
        data_type, label_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type)
        )

    if len(input_dataset_types) == 3:
        data_type, label_type, sample_weight_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type),
            map_to_zero(sample_weight_type)
        )


def positive_instances(*args):
    if len(args) == 2:
        data, label = args
    if len(args) == 3:
        data, label, sample_weights = args

    return tf.math.equal(tf.reduce_max(label), 1)

def negative_instances(*args):
    if len(args) == 2:
        data, label = args
    if len(args) == 3:
        data, label, sample_weights = args
    return tf.math.equal(tf.reduce_max(label), 0)

def normalize_and_serialize_all_splits_and_datasets(dataset_name="physionet2012"):

    for i in range(1, 6):
        train_dataset, dataset_info = tfds.load(
            dataset_name,
            split="train",
            as_supervised=True,
            with_info=True,
            builder_kwargs={'split': i},
            data_dir="./datasets/"
        )
        validation_dataset, _ = tfds.load(
            dataset_name,
            split="validation",
            as_supervised=True,
            with_info=True,
            builder_kwargs={'split': i},
            data_dir="./datasets/"
        )
        test_dataset, _ = tfds.load(
            dataset_name,
            split="test",
            as_supervised=True,
            with_info=True,
            builder_kwargs={'split': i},
            data_dir="./datasets/"
        )

        print(f"Loaded datasets for split {i}")

        for j, dataset in enumerate([train_dataset, validation_dataset, test_dataset]):

            set_id = ["train", "validation", "test"][j]

            print(f"Processing split {i} and set {set_id}")

            np_dataset = tfds.as_numpy(dataset)

            normalizer = Normalizer(dataset_name, i)
            print(f"Normalizer for split {i} created")
            demo_mean = normalizer._demo_means
            demo_std = normalizer._demo_stds
            ts_mean = normalizer._ts_means
            ts_std = normalizer._ts_stds


            all_labels = []
            all_times = []
            all_values = []
            all_measurements = []
            all_static = []

            print(f"Total length of dataset: {dataset_info.splits[set_id].num_examples}")

            for row in tqdm.tqdm(np_dataset, total=dataset_info.splits[set_id].num_examples):
                all_labels.append(row[1])
                all_times.append(row[0][1])
                # Because shapes are not the same for every person depending on number of readings,
                # we pre-normalize (and fill NAs with 0s)
                all_values.append(
                    np.where(
                        row[0][3],
                        (row[0][2] - ts_mean)/ts_std,
                        np.zeros_like(row[0][3])
                    )
                )
                all_measurements.append(row[0][3])
                all_static.append(row[0][0])

            normalized_static = ((np.array(all_static)) - demo_mean) / demo_std
            normalized_values = np.array(all_values)
            normalized_times = np.array(all_times)
            normalized_measurements = np.array(all_measurements)
            normalized_labels = np.array(all_labels)

            pdict = [{
                "ts_values": normalized_values[i],
                "ts_indicators": normalized_measurements[i],
                "ts_times": normalized_times[i],
                "static": normalized_static[i],
                "labels": normalized_labels[i]}
            for i in range(len(all_labels))]

            np.save(f"./{set_id}_{dataset_name}_{i}.npy", arr=pdict, allow_pickle=True)

def build_training_iterator(dataset_name, epochs, batch_size, prepro_fn, split,
                            balance=False, class_balance=None):

    #normalize_and_serialize_all_splits_and_datasets(dataset_name)
    #quit()
    
    # Comment out this block if using own dataset 
    
    dataset, dataset_info = tfds.load(
        dataset_name,
        split="train",
        as_supervised=True,
        with_info=True,
        builder_kwargs={'split': split},
        data_dir="./datasets/"
    )
    

    # dataset_np = tfds.as_numpy(dataset)
    # for ex in dataset_np:
    #     print(ex)
    #     quit()

    
    """
    # FOR HAR IMPLEMENTATION data = np.load("../Patient_Journey_Classification/HARdata/split_" + str(split) + "/train_" + "HAR" + "_" + str(split) + ".npy", allow_pickle=True) #array of dicts

    examples = []
    labels = []
    n_samples = 0

    all_statics = []
    all_times = []
    all_values = []
    all_sensor_masks = []
    all_lengths = []
    for ex in data: 
        # create examples (list (individuals) of tuples (data types = static, times, values, mask, length of values) of arrays (data values))
        static = ex["static"][0] #tf.cast(ex["static"], tf.float32)
        times = ex["ts_times"] #tf.cast(ex["ts_times"], tf.float32)
        values = ex["ts_values"] #tf.cast(ex["ts_values"], tf.float32)
        mask = ex["ts_indicators"]
        length = mask.sum(-1) # true if >0 else false
        length = (length > 0).sum() # np.array([(length > 0).sum()])
        all_statics.append(static)
        all_times.append(times)
        all_values.append(values)
        all_sensor_masks.append(mask)
        all_lengths.append(length)
        # example_tuple = (static, times, values, mask, length)  #(static, times, values, tf.cast(mask, tf.float32), tf.cast(length, tf.float32)) 
        # examples.append(example_tuple)
        # create labels (list of labels)
        labels.append(np.int32(ex["labels"])) #labels.append(tf.cast(int(ex["labels"]), tf.int8))      
        n_samples =+ 1

    examples = (all_statics, all_times, all_values, all_sensor_masks, all_lengths)
    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
    # if that doesn't work, try from generator function?

    # Conver to tensorflow dataset

    # Generate dataset and dataset_info on your own
    # By np loading dataset and converting to tf.data.Dataset()

    """
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    #test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels)) 
    #dataset, dataset_ifo = 

    n_samples = dataset_info.splits['train'].num_examples

    steps_per_epoch = int(math.floor(n_samples / batch_size))
    if prepro_fn is not None: #make sure it's none for my dataset
        dataset = dataset.map(prepro_fn, num_parallel_calls=AUTOTUNE)

    if balance:
        majority_class = max(
            range(len(class_balance)), key=lambda i: class_balance[i])
        minority_class = min(
            range(len(class_balance)), key=lambda i: class_balance[i])

        n_majority = class_balance[majority_class] * n_samples
        n_minority = class_balance[minority_class] * n_samples
        # Generate two separate datasets using filter
        pos_data = (dataset
                    .filter(positive_instances)
                    .shuffle(
                        int(class_balance[1] * n_samples),
                        reshuffle_each_iteration=True)
                    .repeat()
                    )
        neg_data = (dataset
                    .filter(negative_instances)
                    .shuffle(
                        int(class_balance[0] * n_samples),
                        reshuffle_each_iteration=True)
                    .repeat()
                    )
        # And sample from them
        dataset = tf.data.experimental.sample_from_datasets(
            [pos_data, neg_data], weights=[0.5, 0.5])
        # One epoch should at least contain all negative examples or max
        # each instance of the minority class 3 times
        steps_per_epoch = min(
            math.ceil(2 * n_majority / batch_size),
            math.ceil(3 * 2 * n_minority / batch_size)
        )
    else:
        # Shuffle repeat and batch
        dataset = dataset.shuffle(n_samples, reshuffle_each_iteration=True)
        dataset = dataset.repeat(epochs)

    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=True
    )
    return batched_dataset.prefetch(AUTOTUNE), steps_per_epoch


def build_validation_iterator(dataset_name, batch_size, prepro_fn, split):
    """Build a validation iterator for a tensorflow datasets dataset.

    Args:
        dataset_name: Name of the tensoflow datasets dataset. To be used with
            tfds.load().
        epochs: Number of epochs to run
        batch_size: Batch size
        prepro_fn: Optional preprocessing function that should be applied to
            prior to batching.

    Returns:
        A tensorflow dataset which iterates through the validation dataset
           epoch times.

    """
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.VALIDATION,
        as_supervised=True,
        with_info=True,
        builder_kwargs={'split': split},
        data_dir="./datasets/"
    )
    n_samples = dataset_info.splits['validation'].num_examples
    steps_per_epoch = int(math.ceil(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(prepro_fn, num_parallel_calls=AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=False
    )
    return batched_dataset, steps_per_epoch


def build_test_iterator(dataset_name, batch_size, prepro_fn, split):
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.TEST,
        as_supervised=True,
        with_info=True,
        builder_kwargs={'split': split},
        data_dir="./datasets/"
    )
    n_samples = dataset_info.splits['test'].num_examples
    steps = int(math.floor(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(prepro_fn, num_parallel_calls=AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=False
    )
    return batched_dataset, steps


class HParamWithDefault(hp.HParam):
    """Subclass of tensorboard HParam, with additional default parameter."""

    def __init__(self, name, domain=None, display_name=None, description=None,
                 default=None):
        super().__init__(name, domain, display_name, description)
        self._default = default

    @property
    def default(self):
        return self._default


class LogRealInterval(hp.RealInterval):
    """Domain of real values on the log scale."""

    def __repr__(self):
        return "LogRealInterval(%r, %r)" % (self._min_value, self._max_value)

    def sample_uniform(self, rng=random):
        pre_exp = rng.uniform(
            math.log10(self._min_value), math.log10(self._max_value))
        return round(10 ** pre_exp, 5)
