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

"""Looks up model names and hparams set names."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nizza.models import model1, model2

# Singletons (GoF) storing the registry entries
_hparams_set_registry = {}
_model_registry = {}
_registry_built = False


def build_registry():
  """Registers all known hparams sets and models."""
  # TODO(fstahlberg): Decentralize registry and use annotations
  global _registry_built, _hparams_set_registry, _model_registry
  if _registry_built:
    return
  # hparams sets
  _hparams_set_registry.update(model1.register_hparams_sets())
  _hparams_set_registry.update(model2.register_hparams_sets())
  # models
  _model_registry.update(model1.register_models())
  _model_registry.update(model2.register_models())
  _registry_build = True


def print_registry():
  """Print registry."""
  build_registry()
  print("Registered models:")
  for n in _model_registry:
    print("  %s" % n)
  print("\nRegistered hparams sets:")
  for n in _hparams_set_registry:
    print("  %s" % n)


def get_registered_hparams_set(name):
  """Looks up a predefined hparams set by the name. Builds the registry
  if this hasn't been done yet.

  Args:
    name (string): Name of the hparams set. See build_registry().

  Returns:
    An tf.contrib.training.HParams instance.

  Throws:
    An ValueError if the name could not be found.
  """
  build_registry()
  if not name:
    raise ValueError("Please provide a name for the hparams set!")
  try:
    return _hparams_set_registry[name]
  except KeyError:
    raise ValueError("'%s' has not been registered as hparams set!" % name)


def get_registered_model(name, params, run_config):
  """Looks up a model and constructs it using the provided hparams. 
  Builds the registry if this hasn't been done yet.

  Args:
    name (string): Name of the model. See build_registry().
    params(HParams): Hyperparameters.
    run_config(RunConfig): Runtime configuration

  Returns:
    A NizzaModel instance.

  Throws:
    An ValueError if the name could not be found.
  """
  build_registry()
  if not name:
    raise ValueError("Please provide a model name!")
  try:
    return _model_registry[name](params, run_config)
  except KeyError:
    raise ValueError("'%s' has not been registered as model!" % name)

