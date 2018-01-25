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

import copy
import math

from nizza.nizza_model import NizzaModel
from nizza.models.model1 import Model1
from nizza.utils import common_utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def register_hparams_sets():
  base = hparam.HParams(
    lex_hidden_units=[512, 512],
    lex_layer_types=["ffn", "ffn"],
    dist_hidden_units=[128],
    inputs_embed_size=512,
    pos_embed_size=128,
    activation_fn=tf.nn.relu,
    max_timescale=250.0,
    dropout=None
  )
  all_hparams = {}
  all_hparams["model2_default"] = base
  all_hparams["model2s_default"] = base
  # RNN setups
  params = copy.deepcopy(base)
  params.lex_hidden_units = [512]
  params.lex_layer_types = ["bilstm"]
  all_hparams["model2_rnn"] = params
  all_hparams["model2s_rnn"] = params
  return all_hparams


def register_models():
  return {"model2": Model2, "model2s": Model2s}


class BaseModel2(Model1):

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

  def compute_dist_net(self, net, params):
    """Computes the distortion model from positional embeddings.

    Args:
      net: A [batch_size, max_src_len, max_trg_len, pos_embed_size]
        tensor with the concatenated positional embeddings of all
        inputs.
      params: Hyper-parameters

    Returns:
      A [batch_size, max_src_len, max_trg_len] float32 tensor where the
      entry at [b, i, j] stores DistNet(i, j, ...)
    """
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
          activation_fn=None)
    logits = tf.squeeze(logits, axis=-1)
    common_utils.add_hidden_layer_summary(logits, logits_scope.name)
    return logits

  def compute_loss(self, features, mode, params, precomputed):
    lex_logits, dist_logits = precomputed
    inputs = tf.cast(features["inputs"], tf.int32)
    targets = tf.cast(features["targets"], tf.int32)
    # src_weights, trg_weights, and src_bias are used for masking out
    # the pad symbols for loss computation *_weights is 0.0 at pad
    # symbols and 1.0 everywhere else. src_bias is scaled between
    # -1.0e9 and 1.0e9
    src_weights = common_utils.weights_nonzero(inputs) # mask padding
    trg_weights = common_utils.weights_nonzero(targets) # mask padding
    src_bias = (src_weights - 0.5) * 2.0e9
    batch_size = tf.shape(inputs)[0]
    # lex_logits has shape [batch_size, max_src_len, trg_vocab_size]
    lex_partition = tf.reduce_logsumexp(lex_logits, axis=-1)
    dist_partition = tf.reduce_logsumexp(dist_logits, axis=1)
    # lex_partition has shape [batch_size, max_src_len]
    # dist_partition has shape [batch_size, max_trg_len]
    max_src_len = tf.shape(inputs)[1]
    targets_expand = tf.tile(tf.expand_dims(targets, 1), [1, max_src_len, 1])
    # targets_expand has shape [batch_size, max_src_len, max_trg_len]
    lex_src_trg_scores_flat = common_utils.gather_2d(
        tf.reshape(lex_logits, [batch_size * max_src_len,- 1]),
        tf.reshape(targets_expand, [batch_size * max_src_len, -1]))
    lex_src_trg_scores = tf.reshape(lex_src_trg_scores_flat, 
                                    [batch_size, max_src_len, -1])
    # src_trg_scores has shape [batch_size, max_src_len, max_trg_len]
    src_trg_scores = lex_src_trg_scores - tf.expand_dims(lex_partition, 2) + dist_logits
    src_trg_scores_masked = tf.minimum(src_trg_scores, 
                                       tf.expand_dims(src_bias, 2))
    trg_scores = tf.reduce_logsumexp(src_trg_scores_masked, axis=1) - dist_partition
    return -tf.reduce_sum(trg_weights * trg_scores)


  def predict_next_word(self, features, params, precomputed):
    lex_logits, dist_logits = precomputed
    src_weights = common_utils.weights_nonzero(features["inputs"]) # mask padding
    src_bias = (src_weights - 0.5) * 2.0e9
    j = tf.shape(features['targets'])[1] - 1
    this_dist_logits = dist_logits[:, :, j]
    lex_partition = tf.reduce_logsumexp(lex_logits, axis=-1, keep_dims=True)
    dist_partition = tf.reduce_logsumexp(this_dist_logits, axis=1)
    # lex_partition has shape [batch_size, max_src_len, 1]
    # dist_partition has shape [batch_size]
    src_scores = lex_logits - lex_partition + tf.expand_dims(this_dist_logits, 2)
    # src_scores has shape [batch_size, max_src_len, trg_vocab_size]
    src_scores_masked = tf.minimum(src_scores, 
                                   tf.expand_dims(src_bias, 2))
    return tf.reduce_logsumexp(src_scores_masked, axis=1) - tf.expand_dims(dist_partition, 1)


class Model2(BaseModel2):

  def precompute(self, features, mode, params):
    """We precompute the lexical translation logits for each src token and the
    IxJ table of distortion logits.
    """
    lex_logits = self.compute_lex_logits(features, 'inputs', params)
    I = tf.cast(common_utils.get_sentence_length(features["inputs"]), tf.int32)
    J = tf.cast(common_utils.get_sentence_length(features["targets"]), tf.int32)
    dist_logits = self.compute_distance_scores(I, J, params)
    return lex_logits, dist_logits

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
    return self.compute_dist_net(net, params)


class Model2s(BaseModel2):

  def precompute(self, features, mode, params):
    """We precompute the lexical translation logits for each src token and the
    IxJ table of distortion logits.
    """
    lex_logits = self.compute_lex_logits(features, 'inputs', params)
    I = tf.cast(common_utils.get_sentence_length(features["inputs"]), tf.int32)
    max_j = tf.shape(features['targets'])[1]
    dist_logits = self.compute_distance_scores(I, max_j, params)
    return lex_logits, dist_logits

  def compute_distance_scores(self, I, max_j, params):
    """Compute the unnormalized DistNet scores.

    Args:
      I: A [batch_size] int32 tensor with source sentence lengths
      max_j: A int32 TF scalarwith the maximum target sentence lengths
      params: hyper-parameters for that model

    Returns:
      A [batch_size, max_src_len, max_trg_len] float32 tensor where the
      entry at [b, i, j] stores DistNet(i, j, I[b])
    """
    max_i = tf.reduce_max(I)
    batch_size = tf.shape(I)[0]
    expand_I = common_utils.expand_to_shape(I, ["x", max_i, max_j])
    expand_i = common_utils.expand_to_shape(tf.range(max_i), [batch_size, "x", max_j])
    expand_j = common_utils.expand_to_shape(tf.range(max_j), [batch_size, max_i, "x"])
    int_inputs = tf.stack([expand_I, expand_i, expand_j], axis=-1)
    max_pos = tf.maximum(max_i, max_j) + 1
    pos_embeds = self.compute_positional_embeddings(
        max_pos, params, params.pos_embed_size,params.max_timescale)
    embedded = tf.gather(pos_embeds, int_inputs)
    net = tf.reshape(embedded, [batch_size, max_i, max_j, params.pos_embed_size * 3])
    return self.compute_dist_net(net, params)

