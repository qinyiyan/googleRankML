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
"""Tests for the HSTU implementation."""

from absl.testing import absltest
from absl.testing import parameterized
import keras
from keras.src import testing
import numpy as np
from recml.layers.keras import hstu


class HSTUTest(testing.TestCase):

  def test_hstu(self):
    item_ids = keras.ops.array([[1, 2, 3], [4, 5, 0]], "int32")
    padding_mask = keras.ops.array([[1, 1, 1], [1, 1, 0]], "int32")
    init_kws = {
        "vocab_size": 500,
        "max_positions": 20,
        "model_dim": 32,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
    }

    tvars = (
        (500 * 32)  # Item embedding
        + (20 * 32)  # Position embedding
        + 3  # 3 decoder blocks
        * (
            (32 * 32 * 4 + 32 * 32)  # UQKV + output
            + (2 * 32) * 2  # Input + attention Layer norms.
        )
        + (2 * 32)  # Final norm
        + (500 * 32)  # Output embedding
    )
    seed_generators = 1 + 3  # 1 seed generator for each dropout layer.
    model = hstu.HSTU(**init_kws)
    model.build(keras.ops.shape(item_ids))
    self.assertEqual(model.count_params(), tvars)

    self.run_layer_test(
        hstu.HSTU,
        init_kwargs=init_kws,
        input_data=item_ids,
        call_kwargs={"padding_mask": padding_mask},
        expected_output_shape=(2, 3, 500),
        expected_output_dtype="float32",
        expected_num_seed_generators=seed_generators,
        run_training_check=False,
    )


if __name__ == "__main__":
  absltest.main()
