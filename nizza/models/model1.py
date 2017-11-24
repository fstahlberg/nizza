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

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def register_hparams_sets():
  return {"model1_default":  hparam.HParams(hidden_units=[10, 20, 10])}


def register_models():
  return {"model1": Model1}


class Model1(NizzaModel):
  pass
