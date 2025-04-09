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
"""Tests for reduction metrics."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from recml.core.metrics import reduction_metrics


def mse(y_true, y_pred):
  return (y_true - y_pred) ** 2


class ReductionMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'scalar_weighted_sum',
          'metric': reduction_metrics.sum,
          'args': [0.5, 0.5],
          'kwargs': {},
          'expected_output': 0.25,
      },
      {
          'testcase_name': 'unweighted_sum',
          'metric': reduction_metrics.sum,
          'args': [np.array([1, 3, 5, 7])],
          'kwargs': {},
          'expected_output': 16.0,
      },
      {
          'testcase_name': 'weighted_sum',
          'metric': reduction_metrics.sum,
          'args': [np.array([1, 3, 5, 7]), np.array([1, 1, 0, 0])],
          'kwargs': {},
          'expected_output': 4.0,
      },
      {
          'testcase_name': 'weighted_sum_2d',
          'metric': reduction_metrics.sum,
          'args': [np.array([[1, 3], [5, 7]])],
          'kwargs': {'weights': np.array([[1, 1], [1, 0]])},
          'expected_output': 9.0,
      },
      {
          'testcase_name': 'weighted_sum_2d_broadcast',
          'metric': reduction_metrics.sum,
          'args': [np.array([[1, 3], [5, 7]]), np.array([[1, 0]])],
          'kwargs': {},
          'expected_output': 6.0,
      },
      {
          'testcase_name': 'weighted_sum_3d_broadcast',
          'metric': reduction_metrics.sum,
          'args': [
              np.array([
                  [[0.3, 0.7, 0.4, 0.6], [0.5, 0.75, 0.25, 1.5]],
                  [[0.6, 0.3, 0.1, 1.0], [0.3, 0.7, 0.75, 0.25]],
              ])
          ],
          'kwargs': {'weights': np.array([[1, 1], [1, 0]])},
          'expected_output': 7.0,
      },
      {
          'testcase_name': 'unweighted_sum_from_fun',
          'metric': reduction_metrics.Sum.from_fun(mse).from_model_output,
          'args': [
              np.array([
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
              ]),
              np.array([
                  [0, 0, 1, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1],
              ]),
          ],
          'kwargs': {},
          'expected_output': 10.0,
      },
      {
          'testcase_name': 'scalar_weighted_mean',
          'metric': reduction_metrics.mean,
          'args': [0.5, 0.5],
          'kwargs': {},
          'expected_output': 0.5,
      },
      {
          'testcase_name': 'unweighted_mean',
          'metric': reduction_metrics.mean,
          'args': [np.array([1, 3, 5, 7])],
          'kwargs': {},
          'expected_output': 4.0,
      },
      {
          'testcase_name': 'weighted_mean',
          'metric': reduction_metrics.mean,
          'args': [np.array([1, 3, 5, 7]), np.array([1, 1, 0, 0])],
          'kwargs': {},
          'expected_output': 2.0,
      },
      {
          'testcase_name': 'weighted_mean_neg_weights',
          'metric': reduction_metrics.mean,
          'args': [np.array([1, 3, 5, 7]), np.array([-1, -1, 0, 0])],
          'kwargs': {},
          'expected_output': 2.0,
      },
      {
          'testcase_name': 'weighted_mean_2d',
          'metric': reduction_metrics.mean,
          'args': [np.array([[1, 3], [5, 7]])],
          'kwargs': {'weights': np.array([[1, 1], [1, 0]])},
          'expected_output': 3.0,
      },
      {
          'testcase_name': 'weighted_mean_2d_broadcast',
          'metric': reduction_metrics.mean,
          'args': [np.array([[1, 3], [5, 7]]), np.array([[1, 0]])],
          'kwargs': {},
          'expected_output': 3.0,
      },
      {
          'testcase_name': 'weighted_mean_3d_broadcast',
          'metric': reduction_metrics.mean,
          'args': [
              np.array([
                  [[0.3, 0.7, 0.4, 0.6], [0.5, 0.75, 0.25, 1.5]],
                  [[0.6, 0.3, 0.1, 1.0], [0.3, 0.7, 0.75, 0.25]],
              ])
          ],
          'kwargs': {'weights': np.array([[1, 1], [1, 0]])},
          'expected_output': 7 / 12,
      },
      {
          'testcase_name': 'unweighted_mean_from_fun',
          'metric': reduction_metrics.Mean.from_fun(mse).from_model_output,
          'args': [
              np.array([
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
              ]),
              np.array([
                  [0, 0, 1, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1],
              ]),
          ],
          'kwargs': {},
          'expected_output': 0.5,
      },
      {
          'testcase_name': 'weighted_mean_from_fun',
          'metric': reduction_metrics.Mean.from_fun(mse).from_model_output,
          'args': [
              np.array([
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
              ]),
              np.array([
                  [0, 0, 1, 1, 0],
                  [1, 1, 1, 1, 1],
                  [0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1],
              ]),
          ],
          'kwargs': {'weights': np.array([1.0, 1.5, 2.0, 2.5])},
          'expected_output': 0.542857,
      },
  )
  def test_reduction_metric(
      self,
      metric: Callable[..., reduction_metrics.ReductionMetric],
      args: Sequence[Any],
      kwargs: Mapping[str, Any],
      expected_output: float | np.ndarray,
  ):
    instance = metric(*args, **kwargs)
    np.testing.assert_allclose(expected_output, instance.compute(), 1e-3)
    np.testing.assert_allclose(
        expected_output, instance.localize().compute(), 1e-3
    )


if __name__ == '__main__':
  absltest.main()
