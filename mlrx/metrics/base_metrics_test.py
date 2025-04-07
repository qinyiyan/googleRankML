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
"""Tests for base metrics."""

from absl.testing import absltest
import jax.numpy as jnp
from mlrx.metrics import base_metrics


class BaseMetricsTest(absltest.TestCase):

  def test_scalar(self):
    m1 = base_metrics.scalar(1.0)
    m2 = base_metrics.scalar(2.0)
    m3 = m1.merge(m2)
    self.assertEqual(m3.compute(), 2.0)

    self.assertRaises(ValueError, base_metrics.scalar, jnp.array([1.0, 2.0]))


if __name__ == "__main__":
  absltest.main()
