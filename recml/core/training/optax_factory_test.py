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
"""Test for optax optimizer factories."""

from absl.testing import absltest
import jax
import numpy as np
import optax
from recml.core.training import optax_factory


class OptaxFactoryTest(absltest.TestCase):

  def assertOptimizersEqual(
      self,
      a: optax.GradientTransformation,
      b: optax.GradientTransformation,
      steps: int = 10,
  ):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    params = {
        "x": jax.random.uniform(k1, (128, 128)),
        "y": jax.random.uniform(k2, (128, 128)),
        "z": jax.random.uniform(k3, (128, 128)),
    }
    grads = jax.tree.map(lambda p: jax.random.uniform(k4, p.shape), params)

    opt_state_a = a.init(params)
    opt_state_b = b.init(params)

    for _ in range(steps):
      updates_a, opt_state_a = a.update(grads, opt_state_a, params)
      updates_b, opt_state_b = b.update(grads, opt_state_b, params)
      for k in params:
        np.testing.assert_allclose(updates_a[k], updates_b[k])

  def test_optimizer_factory(self):
    optimizer_a = optax_factory.OptimizerFactory(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=5,
            decay_steps=10,
            end_value=0,
        ),
        scaling=optax.scale_by_rms(),
        weight_decay=1e-4,
        weight_decay_mask=r"^(?!.*(?:x|y)$).*",
        grad_clip_norm=1.0,
    ).make()
    optimizer_b = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_rms(),
        optax.add_decayed_weights(
            1e-4, mask=optax_factory._regex_mask(r"^(?!.*(?:x|y)$).*")
        ),
        optax.scale_by_learning_rate(
            optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-3,
                warmup_steps=5,
                decay_steps=10,
                end_value=0,
            )
        ),
    )
    optimizer_c = optax_factory.OptimizerFactory(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=5,
            decay_steps=10,
            end_value=0,
        ),
        scaling=optax.scale_by_rms(),
        weight_decay=1e-4,
        weight_decay_mask=r"^(?!.*(?:z)$).*",
        grad_clip_norm=1.0,
    ).make()
    self.assertOptimizersEqual(optimizer_a, optimizer_b, steps=10)
    self.assertRaises(
        AssertionError,
        self.assertOptimizersEqual,
        optimizer_a,
        optimizer_c,
        steps=10,
    )

  def test_adam_factory(self):
    optimizer_a = optax_factory.AdamFactory(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=5,
            decay_steps=10,
            end_value=0,
        ),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
    ).make()
    optimizer_b = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-3,
                warmup_steps=5,
                decay_steps=10,
                end_value=0,
            ),
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=1e-4,
            mask=optax_factory._default_weight_decay_mask,
        ),
    )
    self.assertOptimizersEqual(optimizer_a, optimizer_b, steps=10)

  def test_adagrad_factory(self):
    optimizer_a = optax_factory.AdagradFactory(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=5,
            decay_steps=10,
            end_value=0,
        ),
        initial_accumulator_value=0.1,
        eps=1e-7,
        grad_clip_norm=1.0,
    ).make()
    optimizer_b = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adagrad(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=1e-3,
                warmup_steps=5,
                decay_steps=10,
                end_value=0,
            ),
            initial_accumulator_value=0.1,
            eps=1e-7,
        ),
    )
    self.assertOptimizersEqual(optimizer_a, optimizer_b, steps=10)


if __name__ == "__main__":
  absltest.main()
