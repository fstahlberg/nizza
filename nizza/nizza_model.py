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
    """This is the model_fn for nizza models. Subclasses should not
    override this function directly, but rather control its behavior
    by implementing `precompute()` and `compute_loss()`

    Args:
      features (dict): Dictionary of tensors holding the raw data.
      mode (int): Running mode (train, eval, predict)
      params (HParams): Hyper-parameters for this model.

    Returns:
      EstimatorSpec as expected by the tf.estimator framework.
    """
    precomputed = self.precompute(features, mode, params)
    loss = self.compute_loss(features, mode, params, precomputed)
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
    """This is a helper function for alignment models for embeddings.
    This function returns an embedding for features[name] of size
    params.name_embed_size assuming a vocab size of params.name_vocab_size.
    features[feature_name] has to be an integer tensor of shape
    [batch_size, max_sequence_length].

    Args:
      features (dict): Dictionary of tensors
      feature_name (string): Name of the feature
      params (HParams): Hyper-parameters for this model.

    Returns:
      A [batch_size, max_sequence_length, embed_size] float32 of embeddings.
    """
    with tf.variable_scope("%s_embed" % feature_name):
      embed_matrix = tf.get_variable("embedding_matrix",
            [getattr(params, "%s_vocab_size" % feature_name), 
             getattr(params, "%s_embed_size" % feature_name)])
      return tf.nn.embedding_lookup(embed_matrix, features[feature_name])

  def precompute(self, features, mode, params):
    """Implemnenting this function can bundle the computation of 
    variables which are used for both decoding (alignment/translation) and
    training. The return value of that function is passed to compute_loss()
    etc. as the `precomputed` argumemnt.

    Args:
      features (dict): Dictionary of tensors holding the raw data.
      mode (int): Running mode (train, eval, predict)
      params (HParams): Hyper-parameters for this model.

    Returns:
      object.
    """
    return None

  def compute_loss(self, features, mode, params, precomputed):
    """Computes the training loss for the alignment model. For example,
    you could implement cross-entropy loss as

      loss_num, loss_den = common_utils.padded_cross_entropy(
          precomputed, features["targets"], params.label_smoothing)
      loss = loss_num / tf.maximum(1.0, loss_den)
      return loss

    assuming that precomputed holds the logits for the targets.

    Args:
      features (dict): Dictionary of tensors holding the raw data.
      mode (int): Running mode (train, eval, predict)
      params (HParams): Hyper-parameters for this model.
      precomputed: Return value of `precomputed()`

    Returns:
      A single float32 scalar which is the loss over the batch.
    """
    raise NotImplementedError("Model does not implement loss.")
