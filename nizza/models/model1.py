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

from nizza.nizza_model import NizzaModel
from nizza.utils import common_utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def register_hparams_sets():
  base = hparam.HParams(
    hidden_units=[512, 512, 512],
    inputs_embed_size=512,
    activation_fn=tf.nn.relu,
    logit_fn=tf.exp, # tf.exp, tf.sigmoid
    label_smoothing=0.1,
    dropout=None
  )
  all_hparams = {}
  all_hparams["model1_default"] = base
  return all_hparams


def register_models():
  return {"model1": Model1}


class Model1(NizzaModel):
  def compute_logits(self, features, mode, params):
    net = self.embed(features, 'inputs', params)
    for layer_id, num_hidden_units in enumerate(params.hidden_units):
      with tf.variable_scope(
          "hiddenlayer_%d" % layer_id,
          values=(net,)) as hidden_layer_scope:
        net = tf.contrib.layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=params.activation_fn)
        if params.dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = layers.dropout(net, keep_prob=(1.0 - params.dropout))
      common_utils.add_hidden_layer_summary(net, hidden_layer_scope.name)

    with tf.variable_scope(
        "logits",
        values=(net,)) as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          net,
          params.targets_vocab_size,
          activation_fn=None)
    common_utils.add_hidden_layer_summary(logits, logits_scope.name)
    return logits

  def compute_loss(self, features, mode, params, logits):
    inputs = features["inputs"]
    targets = features["targets"]
    probs_num = params.logit_fn(logits)
    probs_denom = tf.reduce_sum(probs_num, axis=-1)
    inputs_weights = common_utils.weights_nonzero(inputs) 
    factors = tf.expand_dims(inputs_weights / probs_denom, -1)
    log_probs_sum = tf.log(tf.reduce_sum(factors * probs_num, axis=-2))
    # log_probs_sum has shape [batch_size, target_vocab_size]
    outer_summands = common_utils.gather_2d(log_probs_sum, 
                                            tf.cast(targets, tf.int32))
    targets_weights = common_utils.weights_nonzero(targets) 
    # targets, targets_weights, outer_summands have [batch_size, target_length]
    return -tf.reduce_sum(targets_weights * outer_summands)

