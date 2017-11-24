# coding=utf-8
# Copyright 2017 The Nizza Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TODO
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from nizza import registry

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Flags which need to be specified for each nizza run (decoding and training)
flags.DEFINE_string("model_dir", "", "Directory containing the checkpoints.")
flags.DEFINE_string("model", "", "Model name.")
flags.DEFINE_string("hparams_set", "", "Predefined hyper-parameter set.")
flags.DEFINE_string("hparams", "", "Additional hyper-parameters.")


def get_run_config():
  """Constructs the RunConfig using command line arguments.

  Returns
    tf.contrib.learn.RunConfig
  """
  run_config = tf.contrib.learn.RunConfig()
  run_config = run_config.replace(model_dir=FLAGS.model_dir)
  return run_config


def get_hparams():
  """Gets the hyperparameters from command line arguments.

  Returns:
    An HParams instance.

  Throws:
    ValueError if FLAGS.hparams_set could not be found
    in the registry.
  """
  hparams = registry.get_registered_hparams_set(FLAGS.hparams_set)
  hparams.parse(FLAGS.hparams)
  return hparams


def build_input_fn(file_pattern, 
                   batch_size=1, 
                   shuffle=False, 
                   repeat_count=1):
  """This function can be used to build input functions for
  tf.Estimators.

  Wrap with lambda expression before passing it to tf.Estimator:

    ...
    traininput_fn=lambda: build_input_fn(...)
    ...

  Args:
    file_pattern (string): Path to the TFRecord database file. Can
        contain wildcards
    batch_size (int): Batch size
    shuflle (bool): Whether to shuffle the dataset.
    repeat_count (int): Number of epochs, or None for infinite repeat.

  Returns:
    dict of batched features.
  """
  data_fields = {
      "inputs": tf.VarLenFeature(tf.int64),
      "targets": tf.VarLenFeature(tf.int64)
  }
  data_items_to_decoders = {
      field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields}

  def decode_record(record):
    """Serialized Example to dict of <feature name, Tensor>."""
    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    decode_items = list(data_items_to_decoders)
    decoded = decoder.decode(record, items=decode_items)
    return dict(zip(decode_items, decoded))

  data_files = tf.contrib.slim.parallel_reader.get_data_files(file_pattern)
  if shuffle:
    random.shuffle(data_files)
  dataset = tf.data.TFRecordDataset(data_files)
  dataset = dataset.map(decode_record)
  dataset = dataset.repeat(repeat_count)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=FLAGS.batch_size * 100)
  dataset = dataset.batch(FLAGS.batch_size)
  iterator = dataset.make_one_shot_iterator()
  batch_features = iterator.get_next()
  return batch_features


