# coding=utf-8
# Copyright 2018 The Nizza Authors.
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
"""Implementation of the neural alignment model 1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nizza.nizza_model import NizzaModel
from nizza.utils import common_utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import copy


def register_hparams_sets():
  base = hparam.HParams(
    lex_hidden_units=[512, 512],
    lex_layer_types=["ffn", "ffn"], # ffn, lstm, bilstm
    inputs_embed_size=512,
    activation_fn=tf.nn.relu,
    dropout=None
  )
  all_hparams = {}
  all_hparams["model1_default"] = base
  # RNN setup
  params = copy.deepcopy(base)
  params.lex_hidden_units = [512]
  params.lex_layer_types = ["bilstm"]
  all_hparams["model1_rnn"] = params
  return all_hparams


def register_models():
  return {"model1": Model1}


class Model1(NizzaModel):
  """Neural version of IBM 1 model. See paper for more details."""

  def precompute(self, features, mode, params):
    """We precompute the lexical translation logits for each src token."""
    return self.compute_lex_logits(features, 'inputs', params)

  def compute_lex_logits(self, features, feature_name, params):
    """Model 1 style lexical translation scores, modelled by a feedforward
    neural network. The source words must be stored in raw form in
    features[feature_name] of shape [batch_size, max_src_sequence_length].

    Args:
      features (dict): Dictionary of tensors
      feature_name (string): Name of the feature
      params (HParams): Hyper-parameters for this model.
      
    Returns:
      A [batch_size, max_src_equence_length, trg_vocab_size] float32 tensor
      with unnormalized translation scores of the target words given the
      source words.
    """
    not_padding = tf.not_equal(features['inputs'], common_utils.PAD_ID)
    sequence_length = tf.reduce_sum(tf.cast(not_padding, tf.int32), axis=1)
    net = self.embed(features, 'inputs', params)
    for layer_id, (num_hidden_units, layer_type) in enumerate(
          zip(params.lex_hidden_units, params.lex_layer_types)):
      with tf.variable_scope(
          "lex_hiddenlayer_%d" % layer_id,
          values=(net,)) as hidden_layer_scope:
        if layer_type == "ffn":
          net = tf.contrib.layers.fully_connected(
              net,
              num_hidden_units,
              activation_fn=params.activation_fn)
        elif layer_type == "lstm":
          cell = tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size)
          net, _ = tf.nn.dynamic_rnn(
            cell,
            net,
            sequence_length=sequence_length,
            dtype=tf.float32)
        elif layer_type == "bilstm":
          with tf.variable_scope("fwd"):
            fwd_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_units / 2)
          with tf.variable_scope("bwd"):
            bwd_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_units / 2)
          bi_net, _ = tf.nn.bidirectional_dynamic_rnn(
            fwd_cell, bwd_cell, net,
            sequence_length=sequence_length,
            dtype=tf.float32)
          net = tf.concat(bi_net, -1)
        else:
          raise AttributeError("Unknown layer type '%s'" % layer_type)
        if params.dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = layers.dropout(net, keep_prob=(1.0 - params.dropout))
      common_utils.add_hidden_layer_summary(net, hidden_layer_scope.name)
    with tf.variable_scope(
        "lex_logits",
        values=(net,)) as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          net,
          params.targets_vocab_size,
          activation_fn=None)
    common_utils.add_hidden_layer_summary(logits, logits_scope.name)
    return logits

  def compute_loss(self, features, mode, params, precomputed):
    """See paper on how to compute the loss for Model 1."""
    lex_logits = precomputed
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
    partition = tf.reduce_logsumexp(lex_logits, axis=-1)
    # partition has shape [batch_size, max_src_len]
    max_src_len = tf.shape(inputs)[1]
    targets_expand = tf.tile(tf.expand_dims(targets, 1), [1, max_src_len, 1])
    # targets_expand has shape [batch_size, max_src_len, max_trg_len]
    src_trg_scores_flat = common_utils.gather_2d(
        tf.reshape(lex_logits, [batch_size * max_src_len,- 1]),
        tf.reshape(targets_expand, [batch_size * max_src_len, -1]))
    src_trg_scores = tf.reshape(src_trg_scores_flat, 
                                [batch_size, max_src_len, -1])
    # src_trg_scores has shape [batch_size, max_src_len, max_trg_len]
    src_trg_scores_norm = src_trg_scores - tf.expand_dims(partition, 2)
    src_trg_scores_masked = tf.minimum(src_trg_scores_norm, 
                                       tf.expand_dims(src_bias, 2))
    trg_scores = tf.reduce_logsumexp(src_trg_scores_masked, axis=1)
    # trg_weights and trg_scores have shape [batch_size, max_trg_len]
    return -tf.reduce_sum(trg_weights * trg_scores)

  def predict_next_word(self, features, params, precomputed):
    """Returns the sum over all lexical translation scores."""
    lex_logits = precomputed
    inputs = tf.cast(features["inputs"], tf.int32)
    src_weights = common_utils.weights_nonzero(inputs) # mask padding
    src_bias = (src_weights - 0.5) * 2.0e9
    # lex_logits has shape [batch_size, max_src_len, trg_vocab_size]
    partition = tf.reduce_logsumexp(lex_logits, axis=-1, keep_dims=True)
    # partition has shape [batch_size, max_src_len, 1]
    src_scores = lex_logits - partition
    src_scores_masked = tf.minimum(src_scores, tf.expand_dims(src_bias, 2))
    return tf.reduce_logsumexp(src_scores_masked, axis=1)

