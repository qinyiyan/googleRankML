# Copyright 2024 RecML authors <recommendations-ml@google.com>.
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
"""Mean metrics."""

import jax
import keras
from recml.core.metrics import reduction_metrics


def accuracy(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    **_,
) -> reduction_metrics.Mean:
  """Computes accuracy from observations.

  Args:
    y_true: The true labels of shape [D1, ..., D_N]
    y_pred: The predicted logits of shape [D1, ..., D_N, num_classes].
    weights: Optional weights of shape broadcastable to [D1, ... D_N].
    **_: Unused kwargs.

  Returns:
    A metric accumulation of the accuracy.
  """
  assert keras.backend.backend() == 'jax'
  acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
  return reduction_metrics.mean(acc, weights)


def top_k_accuracy(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    k: int,
    **_,
) -> reduction_metrics.Mean:
  """Computes top-k accuracy from observations.

  Args:
    y_true: The true labels of shape [D1, ..., D_N]
    y_pred: The predicted logits of shape [D1, ..., D_N, num_classes].
    weights: Optional weights of shape broadcastable to [D1, ... D_N].
    k: The number of top classes to consider. Must be less than num_classes.
    **_: Unused kwargs.

  Returns:
    A metric accumulation of the top-k accuracy.
  """
  assert keras.backend.backend() == 'jax'
  acc = keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)
  return reduction_metrics.mean(acc, weights)


def binary_accuracy(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    *,
    threshold: float = 0.5,
    **_,
) -> reduction_metrics.Mean:
  """Computes binary accuracy from observations.

  Args:
    y_true: The true labels of shape [D1, ..., D_N]
    y_pred: The binary predictions of shape [D1, ..., D_N].
    weights: Optional weights of shape broadcastable to [D1, ... D_N].
    threshold: The threshold to use for binary classification.
    **_: Unused kwargs.

  Returns:
    A metric accumulation of the binary accuracy.
  """
  assert keras.backend.backend() == 'jax'
  bin_acc = keras.metrics.binary_accuracy(y_true, y_pred, threshold=threshold)
  return reduction_metrics.mean(bin_acc, weights)


def mean_squared_error(
    y_true: jax.Array,
    y_pred: jax.Array,
    weights: jax.Array | None = None,
    **_,
) -> reduction_metrics.Mean:
  """Computes mean squared error from observations.

  Args:
    y_true: The true labels of shape [D1, ..., D_N].
    y_pred: The predictions of shape [D1, ..., D_N].
    weights: Optional weights of shape broadcastable to [D1, ... D_N].
    **_: Unused kwargs.

  Returns:
    A metric accumulation of the mean squared error.
  """
  assert keras.backend.backend() == 'jax'
  mse = keras.metrics.mean_squared_error(y_true, y_pred)
  return reduction_metrics.mean(mse, weights)
