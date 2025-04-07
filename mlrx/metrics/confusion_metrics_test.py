# Copyright 2024 RecML authors <no-reply@google.com>.
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
"""Tests for confusion metrics."""

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from mlrx.metrics import confusion_metrics


class ConfusionMetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'predictions': np.array([
              [0, 0, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1],
          ]),
          'labels': np.array([
              [0, 1, 0, 1, 0],
              [0, 0, 1, 1, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
          ]),
          'weights': None,
          'thresholds': [0.5],
          'expected_tp': np.array([7.0]),
          'expected_tn': np.array([3.0]),
          'expected_fp': np.array([7.0]),
          'expected_fn': np.array([3.0]),
      },
      {
          'testcase_name': 'weighted',
          'predictions': np.array([
              [0, 0, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1],
          ]),
          'labels': np.array([
              [0, 1, 0, 1, 0],
              [0, 0, 1, 1, 1],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
          ]),
          'weights': np.array([1.0, 1.5, 2.0, 2.5]),
          'thresholds': [0.5],
          'expected_tp': np.array([12.0]),
          'expected_tn': np.array([4.0]),
          'expected_fp': np.array([14.0]),
          'expected_fn': np.array([5.0]),
      },
      {
          'testcase_name': 'unweighted_thresholds',
          'predictions': np.array([
              [0.9, 0.2, 0.8, 0.1],
              [0.2, 0.9, 0.7, 0.6],
              [0.1, 0.2, 0.4, 0.3],
              [0.0, 1.0, 0.7, 0.3],
          ]),
          'labels': np.array([
              [0, 1, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 1, 1, 1],
          ]),
          'weights': None,
          'thresholds': [0.15, 0.5, 0.85],
          'expected_tp': np.array([6.0, 3.0, 1.0]),
          'expected_tn': np.array([2.0, 5.0, 7.0]),
          'expected_fp': np.array([7.0, 4.0, 2.0]),
          'expected_fn': np.array([1.0, 4.0, 6.0]),
      },
      {
          'testcase_name': 'weighted_thresholds',
          'predictions': np.array([
              [0.9, 0.2, 0.8, 0.1],
              [0.2, 0.9, 0.7, 0.6],
              [0.1, 0.2, 0.4, 0.3],
              [0.0, 1.0, 0.7, 0.3],
          ]),
          'labels': np.array([
              [0, 1, 1, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 1, 1, 1],
          ]),
          'weights': np.array([
              [1.0, 2.0, 3.0, 5.0],
              [7.0, 11.0, 13.0, 17.0],
              [19.0, 23.0, 29.0, 31.0],
              [5.0, 15.0, 10.0, 0],
          ]),
          'thresholds': [0.15, 0.5, 0.85],
          'expected_tp': np.array([37.0, 28.0, 15.0]),
          'expected_tn': np.array([24.0, 107.0, 137.0]),
          'expected_fp': np.array([125.0, 42.0, 12.0]),
          'expected_fn': np.array([5.0, 14.0, 27.0]),
      },
  )
  def test_estimate_confusion_matrix(
      self,
      predictions: jax.Array,
      labels: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_tp: jax.Array,
      expected_tn: jax.Array,
      expected_fp: jax.Array,
      expected_fn: jax.Array,
  ):
    tp, tn, fp, fn = confusion_metrics.estimate_confusion_matrix(
        predictions=predictions,
        labels=labels,
        weights=weights,
        thresholds=thresholds,
    )
    np.testing.assert_allclose(expected_tp, tp)
    np.testing.assert_allclose(expected_tn, tn)
    np.testing.assert_allclose(expected_fp, fp)
    np.testing.assert_allclose(expected_fn, fn)

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 1, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.5,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[0, 1, 1, 0], [1, 0, 0, 1]]),
          'y_pred': np.array([[1, 0, 1, 0], [1, 0, 1, 0]]),
          'weights': np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
          'thresholds': [0.5],
          'expected_outputs': (3.0 + 4.0) / ((1.0 + 3.0) + (4.0 + 2.0)),
      },
      {
          'testcase_name': 'div_by_zero',
          'y_true': np.array([0, 0, 0, 0]),
          'y_pred': np.array([0, 0, 0, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.0,
      },
      {
          'testcase_name': 'unweighted_thresholds',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 0.6, 0]),
          'weights': None,
          'thresholds': [0.5, 0.7],
          'expected_outputs': [0.5, 0.0],
      },
      {
          'testcase_name': 'weighted_thresholds',
          'y_true': np.array([[0, 1], [1, 0]]),
          'y_pred': np.array([[1, 0], [0.6, 0]], dtype='float32'),
          'weights': np.array([[4, 0], [3, 1]], dtype='float32'),
          'thresholds': [0.5, 1.0],
          'expected_outputs': [(0 + 3.0) / ((0 + 3.0) + (4.0 + 0.0)), 0.0],
      },
  )
  def test_precision(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_outputs: float | Sequence[float],
  ):
    precision = confusion_metrics.precision(
        y_true, y_pred, weights, threshold=thresholds
    )
    np.testing.assert_allclose(expected_outputs, precision.compute())
    np.testing.assert_allclose(expected_outputs, precision.localize().compute())

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 1, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.5,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([[0, 1, 1, 0], [1, 0, 0, 1]]),
          'y_pred': np.array([[1, 0, 1, 0], [0, 1, 0, 1]]),
          'weights': np.array([[1, 2, 3, 4], [4, 3, 2, 1]]),
          'thresholds': [0.5],
          'expected_outputs': (3.0 + 1.0) / ((2.0 + 3.0) + (4.0 + 1.0)),
      },
      {
          'testcase_name': 'div_by_zero',
          'y_true': np.array([0, 0, 0, 0]),
          'y_pred': np.array([0, 0, 0, 0]),
          'weights': None,
          'thresholds': [0.5],
          'expected_outputs': 0.0,
      },
      {
          'testcase_name': 'unweighted_thresholds',
          'y_true': np.array([0, 1, 1, 0]),
          'y_pred': np.array([1, 0, 0.6, 0]),
          'weights': None,
          'thresholds': [0.5, 0.7],
          'expected_outputs': [0.5, 0.0],
      },
      {
          'testcase_name': 'weighted_thresholds',
          'y_true': np.array([[0, 1], [1, 0]]),
          'y_pred': np.array([[1, 0], [0.6, 0]], dtype='float32'),
          'weights': np.array([[1, 4], [3, 2]], dtype='float32'),
          'thresholds': [0.5, 1.0],
          'expected_outputs': [(0 + 3.0) / ((0 + 3.0) + (4.0 + 0.0)), 0.0],
      },
  )
  def test_recall(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_outputs: float | Sequence[float],
  ):
    recall = confusion_metrics.recall(
        y_true, y_pred, weights, threshold=thresholds
    )
    np.testing.assert_allclose(expected_outputs, recall.compute())
    np.testing.assert_allclose(expected_outputs, recall.localize().compute())

  @parameterized.named_parameters({
      'testcase_name': 'unweighted',
      'y_true': np.array([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0],
          [1, 0, 0],
          [0, 0, 1],
      ]),
      'y_pred': np.array([
          [0.9, 0.1, 0],
          [0.2, 0.6, 0.2],
          [0, 0, 1],
          [0.4, 0.3, 0.3],
          [0, 0.9, 0.1],
          [0, 0, 1],
      ]),
      'weights': None,
      'thresholds': [0.5],
      'expected_outputs': 0.727,
  })
  def test_f1_score(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      thresholds: Sequence[float],
      expected_outputs: float,
  ):
    f1 = confusion_metrics.f1_score(
        y_true, y_pred, weights, threshold=thresholds
    )
    np.testing.assert_allclose(expected_outputs, f1.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_outputs, f1.localize().compute(), rtol=1e-3
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9], dtype='float32'),
          'weights': np.array([1, 2, 3, 4]),
          'num_thresholds': 3,
          'from_logits': False,
          # auc = (slope / Total Pos) * [dTP - intercept * log(Pb/Pa)]
          # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
          # P = tp + fp = [10, 4, 0]
          # dTP = [7-4, 4-0] = [3, 4]
          # dP = [10-4, 4-0] = [6, 4]
          # slope = dTP/dP = [0.5, 1]
          # intercept = (TPa+(slope*Pa) = [(4 - 0.5*4), (0 - 1*0)] = [2, 0]
          # (Pb/Pa) = (Pb/Pa) if Pb > 0 AND Pa > 0 else 1 = [10/4,4/0] = [2.5,1]
          # auc * TotalPos = [(0.5 * (3 + 2 * log(2.5))), (1 * (4 + 0))]
          #                = [2.416, 4]
          # auc = [2.416, 4]/(tp[1:]+fn[1:])
          'expected_outputs': 2.416 / 7 + 4 / 7,
      },
      {
          'testcase_name': 'weighted_negative',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9], dtype='float32'),
          'weights': np.array([-1, -2, -3, -4]),
          'num_thresholds': 3,
          'from_logits': False,
          # Divisor in auc formula is max(tp[1:]+fn[1:], 0), which is all zeros
          # because the all values in tp and fn are negative, divide_no_nan will
          # produce all zeros.
          'expected_outputs': 0.0,
      },
  )
  def test_aucpr(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      num_thresholds: int,
      from_logits: bool,
      expected_outputs: float,
  ):
    aucpr = confusion_metrics.aucpr(
        y_true,
        y_pred,
        weights,
        from_logits=from_logits,
        num_thresholds=num_thresholds,
    )
    np.testing.assert_allclose(expected_outputs, aucpr.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_outputs, aucpr.localize().compute(), rtol=1e-3
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted_all_correct',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0, 1, 1]),
          'weights': None,
          'num_thresholds': 3,
          'from_logits': False,
          'expected_outputs': 1.0,
      },
      {
          'testcase_name': 'unweighted',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9]),
          'weights': None,
          'num_thresholds': 3,
          'from_logits': False,
          # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
          # recall = [2/2, 1/(1+1), 0] = [1, 0.5, 0]
          # fp_rate = [2/2, 0, 0] = [1, 0, 0]
          # heights = [(1 + 0.5)/2, (0.5 + 0)/2] = [0.75, 0.25]
          # widths = [(1 - 0), (0 - 0)] = [1, 0]
          # aucroc = 0.75 * 1 + 0.25 * 0
          'expected_outputs': 0.75,
      },
      {
          'testcase_name': 'unweighted_from_logits',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': -np.log(1.0 / (np.array([0, 0.5, 0.3, 0.9]) + 1e-12) - 1.0),
          'weights': None,
          'num_thresholds': 3,
          'from_logits': True,
          'expected_outputs': 0.75,  # Same as unweighted case.
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 0, 1, 1]),
          'y_pred': np.array([0, 0.5, 0.3, 0.9]),
          'weights': np.array([1, 2, 3, 4]),
          'num_thresholds': 3,
          'from_logits': False,
          # tp = [7, 4, 0], fp = [3, 0, 0], fn = [0, 3, 7], tn = [0, 3, 3]
          # recall = [7/7, 4/(4+3), 0] = [1, 0.571, 0]
          # fp_rate = [3/3, 0, 0] = [1, 0, 0]
          # heights = [(1 + 0.571)/2, (0.571 + 0)/2] = [0.7855, 0.2855]
          # widths = [(1 - 0), (0 - 0)] = [1, 0]
          # aucroc = 0.7855 * 1 + 0.2855 * 0
          'expected_outputs': 0.7855,
      },
  )
  def test_aucroc(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      num_thresholds: int,
      from_logits: bool,
      expected_outputs: float,
  ):
    aucroc = confusion_metrics.aucroc(
        y_true,
        y_pred,
        weights,
        from_logits=from_logits,
        num_thresholds=num_thresholds,
    )
    print(aucroc)
    np.testing.assert_allclose(expected_outputs, aucroc.compute(), rtol=1e-3)
    np.testing.assert_allclose(
        expected_outputs, aucroc.localize().compute(), rtol=1e-3
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'unweighted_high_recall',
          'y_true': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
          'y_pred': np.array(
              [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
          ),
          'weights': None,
          'recall': 0.8,
          'expected_outputs': 2.0 / 3,
      },
      {
          'testcase_name': 'unweighted_low_recall',
          'y_true': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
          'y_pred': np.array(
              [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
          ),
          'weights': None,
          'recall': 0.6,
          'expected_outputs': 0.75,
      },
      {
          'testcase_name': 'weighted',
          'y_true': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
          'y_pred': np.array(
              [0.0, 0.1, 0.2, 0.5, 0.6, 0.2, 0.5, 0.6, 0.8, 0.9]
          ),
          'weights': np.array([2, 1, 2, 1, 2, 1, 2, 2, 1, 2]),
          'recall': 7.0 / 8,
          'expected_outputs': 0.7,
      },
  )
  def test_prediction_at_recall(
      self,
      y_true: jax.Array,
      y_pred: jax.Array,
      weights: jax.Array | None,
      recall: float,
      expected_outputs: float | Sequence[float],
  ):
    precision_at_recall = confusion_metrics.precision_at_recall(
        y_true, y_pred, weights=weights, recall=recall
    )
    np.testing.assert_allclose(
        expected_outputs, precision_at_recall.compute(), rtol=1e-3
    )
    np.testing.assert_allclose(
        expected_outputs, precision_at_recall.localize().compute(), rtol=1e-3
    )


if __name__ == '__main__':
  absltest.main()
