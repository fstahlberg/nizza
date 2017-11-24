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

