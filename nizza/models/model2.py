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
"""Implementation of the neural alignment model 2. This module contains
two variants of model 2. One follows the original IBM 2 model. The second
one does not condition the distortion model on the target sentence length.
Forward probabilities of the second one can be directly used in
translation where the first one may be more appropriate for alignment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from nizza.nizza_model import NizzaModel
from nizza.models.model1 import Model1
from nizza.utils import common_utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def _print_shape_py(t, msg):
  print("%s shape: %s" % (msg, t.shape))
  return sum(t.shape)


def print_shape(t, msg="", dtype=tf.float32):
  """Print shape of the tensor for debugging."""
  add = tf.py_func(_print_shape_py, [t, msg], tf.int64)
  shp = t.get_shape()
  ret = t + tf.cast(tf.reduce_max(add) - tf.reduce_max(add), dtype=dtype)
  ret.set_shape(shp)
  return ret


def _print_data_py(t, msg):
  print("%s shape: %s" % (msg, t.shape))
  print("%s data: %s" % (msg, t))
  return sum(t.shape)


def print_data(t, msg="", dtype=tf.float32):
  """Print shape and content of the tensor for debugging."""
  add = tf.py_func(_print_data_py, [t, msg], tf.int64)
  shp = t.get_shape()
  ret = t + tf.cast(tf.reduce_max(add) - tf.reduce_max(add), dtype=dtype)
  ret.set_shape(shp)
  return ret


def register_hparams_sets():
  base = hparam.HParams(
    lex_hidden_units=[512, 512, 512],
    dist_hidden_units=[128],
    inputs_embed_size=512,
    pos_embed_size=128,
    activation_fn=tf.nn.relu,
    max_timescale=250.0,
    logit_fn=tf.sigmoid, # tf.exp, tf.sigmoid
    dropout=None
  )
  all_hparams = {}
  all_hparams["model2_default"] = base
  return all_hparams


def register_models():
  return {"model2": Model2}


class Model2(Model1):

  def precompute(self, features, mode, params):
    """We precompute the lexical translation logits for each src token and the
    IxJ table of distortion logits.
    """
    lex_logits = self.compute_lex_logits(features, 'inputs', params)
    I = tf.cast(common_utils.get_sentence_length(features["inputs"]), tf.int32)
    J = tf.cast(common_utils.get_sentence_length(features["targets"]), tf.int32)
    dist_logits = self.compute_distance_scores(I, J, params)
    return lex_logits, dist_logits, I, J

  def compute_positional_embeddings(
        self, max_pos, params, n_channels, max_timescale=1.0e4):
    """Compute the positional embeddings which serve as input to DistNet.

    Args:
      max_pos: A scalar with the maximal position
      params (HParams): hyper-parameters for that model
      n_channels (int): A Python int with the required embedding dimensionality
      max_timescale: a Python float with the maximum period

    Returns:
      A [max_pos+1, embed_size] float32 tensor with positional embeddings.
    """
    position = tf.to_float(tf.range(max_pos+1))
    num_timescales = n_channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(n_channels, 2)]])
    signal = tf.reshape(signal, [max_pos+1, n_channels])
    return signal

  def compute_distance_scores(self, I, J, params):
    """Compute the unnormalized DistNet scores.

    Args:
      I: A [batch_size] int32 tensor with source sentence lengths
      J: A [batch_size] int32 tensor with target sentence lengths
      params: hyper-parameters for that model

    Returns:
      A [batch_size, max_src_len, max_trg_len] float32 tensor where the
      entry at [b, i, j] stores DistNet(i, j, I[b], J[b])
    """
    max_i = tf.reduce_max(I)
    max_j = tf.reduce_max(J)
    batch_size = tf.shape(I)[0]
    expand_I = common_utils.expand_to_shape(I, ["x", max_i, max_j])
    expand_J = common_utils.expand_to_shape(J, ["x", max_i, max_j])
    expand_i = common_utils.expand_to_shape(tf.range(max_i), [batch_size, "x", max_j])
    expand_j = common_utils.expand_to_shape(tf.range(max_j), [batch_size, max_i, "x"])
    int_inputs = tf.stack([expand_I, expand_J, expand_i, expand_j], axis=-1)
    max_pos = tf.reduce_max(tf.concat([I, J], 0)) + 1
    pos_embeds = self.compute_positional_embeddings(
        max_pos, params, params.pos_embed_size,params.max_timescale)
    embedded = tf.gather(pos_embeds, int_inputs)
    net = tf.reshape(embedded, [batch_size, max_i, max_j, params.pos_embed_size * 4])
    for layer_id, num_hidden_units in enumerate(params.dist_hidden_units):
      with tf.variable_scope(
          "dist_hiddenlayer_%d" % layer_id,
          values=(net,)) as hidden_layer_scope:
        net = tf.contrib.layers.fully_connected(
            net,
            num_hidden_units,
            activation_fn=params.activation_fn)
        if params.dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = layers.dropout(net, keep_prob=(1.0 - params.dropout))
      common_utils.add_hidden_layer_summary(net, hidden_layer_scope.name)
    with tf.variable_scope(
        "dist_logits",
        values=(net,)) as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          net,
          1,
          activation_fn=params.logit_fn)
    logits = tf.squeeze(logits, axis=-1)
    common_utils.add_hidden_layer_summary(logits, logits_scope.name)
    return logits

  def compute_loss(self, features, mode, params, precomputed):
    lex_probs_num, dist_logits, I, J = precomputed
    # lex_probs_num[b, i, t] stores p(t|e_i) for example b
    # dist_logits[b, i, j] stores DistNet(i, j, I[b], J[b])
    inputs = tf.cast(features["inputs"], tf.int32)
    targets = tf.cast(features["targets"], tf.int32)
    inputs_weights = common_utils.weights_nonzero(inputs) 
    targets_weights = common_utils.weights_nonzero(targets) 
    dist_logits_zeroed = dist_logits * tf.expand_dims(inputs_weights, axis=2)
    dist_partition = tf.log(tf.reduce_sum(dist_logits_zeroed, axis=1))
    dist_partition_sum = tf.reduce_sum(dist_partition * targets_weights)
   
    shp = tf.shape(inputs)
    batch_size = shp[0]
    max_src_len = shp[1]
    max_trg_len = tf.shape(targets)[1]
    lex_probs_denom = tf.reduce_sum(lex_probs_num, axis=-1)
    factors = tf.expand_dims(inputs_weights / lex_probs_denom, -1)
    # lex_probs_num is [batch_size, src_len, trg_vocab_size]
    # factors is [batch_size, src_len, 1]
    targets_repeated = tf.tile(tf.expand_dims(targets, 1), tf.convert_to_tensor([1, max_src_len, 1]))
    # targets_repeated is [batch_size, src_len, trg_len]
    lex_probs_flat = tf.reshape(lex_probs_num, [batch_size*max_src_len, params.targets_vocab_size])
    targets_flat = tf.reshape(targets_repeated, [batch_size*max_src_len, max_trg_len])
    lex_scores_flat = common_utils.gather_2d(lex_probs_flat, targets_flat)
    lex_scores = tf.reshape(lex_scores_flat, [batch_size, max_src_len, max_trg_len])
    # lex_scores is [batch_size, src_len, trg_len] and contains p(f_j|e_i)
    lexdist_sum = tf.reduce_sum(factors * lex_scores * dist_logits, axis=1)
    lexdist_loss = tf.reduce_sum(targets_weights * tf.log(lexdist_sum))
    return -lexdist_loss + dist_partition_sum
