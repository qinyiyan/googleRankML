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
"""Tests for mean metrics."""

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from recml.core.metrics import mean_metrics


class MeanMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([1, 0, 1, 0]),
          'y_pred': np.array([[0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.6, 0.4]]),
          'weights': None,
          'expected_output': 1.0,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[1, 0, 1, 0]]),
          'y_pred': np.array(
              [[[0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]]]
          ),
          'weights': np.array([[1, 2, 3, 4]]),
          'expected_output': 0.4,
      },
  )
  def test_accuracy(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      weights: np.ndarray | None,
      expected_output: np.ndarray,
  ):
    accuracy = mean_metrics.accuracy(y_true, y_pred, weights)
    np.testing.assert_allclose(expected_output, accuracy.compute())
    np.testing.assert_allclose(expected_output, accuracy.localize().compute())

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([[1], [4], [2], [3], [3], [1], [0], [5]]),
          'y_pred': np.array([
              [[0.1, 0.7, 0.5, 0.3, 0.2, 0.0]],  # [1, 2, 3, 4, 0, 5]
              [[0.2, 0.8, 0.0, 0.1, 0.4, 0.3]],  # [1, 4, 5, 0, 3, 2]
              [[0.1, 0.2, 0.4, 0.8, 0.0, 0.3]],  # [3, 2, 5, 1, 0, 4]
              [[1.0, 0.9, 0.1, 0.3, 0.2, 0.0]],  # [0, 1, 3, 4, 2, 5]
              [[0.1, 0.7, 0.5, 0.3, 0.2, 0.0]],  # [1, 2, 3, 4, 0, 5]
              [[0.2, 0.8, 0.0, 0.1, 0.4, 0.3]],  # [1, 4, 5, 0, 3, 2]
              [[0.1, 0.2, 0.4, 0.8, 0.0, 0.3]],  # [3, 2, 5, 1, 0, 4]
              [[1.0, 0.9, 0.1, 0.3, 0.2, 0.0]],  # [0, 1, 3, 4, 2, 5]
          ]),
          'weights': None,
          'ks': [1, 2, 3, 4, 5, 6],
          'expected_outputs': [0.25, 0.5, 0.75, 0.75, 0.875, 1.0],
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([[0.1, 0.7], [0.2, 0.4], [0.1, 0.3], [0.2, 0.1]]),
          'weights': np.array([0.2, 0.6, 0.1, 0.1]),
          'ks': [1, 2],
          'expected_outputs': np.array([0.8, 1.0]),
      },
  )
  def test_top_k_accuracies(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      weights: np.ndarray | None,
      ks: Sequence[int],
      expected_outputs: Sequence[float],
  ):
    for k, expected_output in zip(ks, expected_outputs):
      accuracy = mean_metrics.top_k_accuracy(y_true, y_pred, weights, k=k)
      np.testing.assert_allclose(expected_output, accuracy.compute())
      np.testing.assert_allclose(expected_output, accuracy.localize().compute())

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([1, 0, 1, 0]),
          'y_pred': np.array([0.4, 0.6, 0.8, 0.2]),
          'weights': None,
          'threshold': 0.5,
          'expected_output': 0.5,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[1, 0, 1, 0]]),
          'y_pred': np.array([[0.8, 0.6, 0.7, 0.6]]),
          'weights': np.array([[1, 2, 3, 4]]),
          'threshold': 0.75,
          'expected_output': 0.7,
      },
  )
  def test_binary_accuracy(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      weights: np.ndarray | None,
      threshold: float,
      expected_output: np.ndarray,
  ):
    accuracy = mean_metrics.binary_accuracy(
        y_true, y_pred, weights, threshold=threshold
    )
    np.testing.assert_allclose(expected_output, accuracy.compute())
    np.testing.assert_allclose(expected_output, accuracy.localize().compute())

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0.3, 0.5, 0.7, 0.9]),
          'y_pred': np.array([0.4, 0.6, 0.8, 0.2]),
          'weights': None,
          'expected_output': 0.13,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[0.3, 0.6, 0.2, 0.6]]),
          'y_pred': np.array([[0.8, 0.6, 0.7, 0.6]]),
          'weights': np.array([0.5]),
          'expected_output': 0.125,
      },
  )
  def test_mean_squared_error(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      weights: np.ndarray | None,
      expected_output: np.ndarray,
  ):
    mse = mean_metrics.mean_squared_error(y_true, y_pred, weights)
    np.testing.assert_allclose(expected_output, mse.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_output, mse.localize().compute(), rtol=1e-3
    )


if __name__ == '__main__':
  absltest.main()
