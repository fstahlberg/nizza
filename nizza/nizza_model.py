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

"""Base class for Nizza models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class NizzaModel(tf.estimator.Estimator):

  def __init__(self, params, config=None):
    """Constructs a new model instance.

    Args:
      params (HParams): Hyper-parameters for this model.
      config (RunConfig): Run time configuration.
    """
    super(NizzaModel, self).__init__(
        model_fn=self.nizza_model_fn, 
        params=params, 
        config=config)

  def nizza_model_fn(self, features, mode, params):
    print("modelfn")
    print(features)
    print(mode)
    print(params)
    loss = tf.constant(2)
    train_op = tf.no_op()
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

