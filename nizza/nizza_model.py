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

from nizza.utils import common_utils

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
    logits = self.compute_logits(features, mode, params)
    loss = self.compute_loss(features, mode, params, logits)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

  def embed(self, features, feature_name, params):
      with tf.variable_scope("%s_embed" % feature_name):
        embed_matrix = tf.get_variable("embedding_matrix",
              [getattr(params, "%s_vocab_size" % feature_name), 
               getattr(params, "%s_embed_size" % feature_name)])
        return tf.nn.embedding_lookup(embed_matrix, features[feature_name])
    

  def compute_logits(self, features, mode, params):
    raise NotImplementedError("Model does not implement logit computation.")

  def compute_loss(self, features, mode, params, logits):
    loss_num, loss_den = common_utils.padded_cross_entropy(
        logits, features["targets"], params.label_smoothing)
    loss = loss_num / tf.maximum(1.0, loss_den)
    return loss
