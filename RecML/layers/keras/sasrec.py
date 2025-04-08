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
"""Models baselined."""

from collections.abc import Mapping, Sequence
from typing import Any

import keras
import keras_hub
from recml.layers.keras import utils

Tensor = Any


@keras.saving.register_keras_serializable("recml")
class SASRec(keras.layers.Layer):
  """SASRec architecture as in [1].

  Implements the SASRec model architecture as described in 'Self-Attentive
  Sequential Recommendation' [1].

  [1] https://arxiv.org/abs/1808.09781
  """

  def __init__(
      self,
      *,
      vocab_size: int,
      max_positions: int,
      model_dim: int,
      mlp_dim: int,
      num_heads: int,
      num_layers: int,
      dropout: float = 0.0,
      norm_eps: float = 1e-6,
      scale_by_sqrt_dim: bool = False,
      add_head: bool = True,
      **kwargs,
  ):
    """Initializes the instance.

    Args:
      vocab_size: The size of the item vocabulary.
      max_positions: The maximum number of positions in a sequence.
      model_dim: The width of the embeddings in the model.
      mlp_dim: The width of the MLP in each transformer block.
      num_heads: The number of attention heads in each transformer block.
      num_layers: The number of transformer blocks in the model.
      dropout: The dropout rate. Defaults to 0.
      norm_eps: The epsilon for RMS normalization.
      scale_by_sqrt_dim: Whether to scale the item embeddings by
        sqrt(model_dim). Defaults to False.
      add_head: Whether to decode the sequence embeddings to logits.
      **kwargs: Passed through to the super class.
    """
    super().__init__(**kwargs)

    self.item_embedding = keras_hub.layers.ReversibleEmbedding(
        input_dim=vocab_size,
        output_dim=model_dim,
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
        dtype=self.dtype_policy,
        reverse_dtype=self.compute_dtype,
        name="item_embedding",
    )

    self.position_embedding = keras_hub.layers.PositionEmbedding(
        sequence_length=max_positions,
        initializer=keras.initializers.RandomNormal(stddev=0.02),
        dtype=self.dtype_policy,
        name="position_embedding",
    )

    self.embeddings_dropout = keras.layers.Dropout(
        dropout, name="embedding_dropout"
    )

    self.decoder_blocks = [
        keras_hub.layers.TransformerDecoder(
            intermediate_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=utils.gelu_approximate,
            layer_norm_epsilon=norm_eps,
            normalize_first=True,
            dtype=self.dtype_policy,
            name=f"decoder_block_{i}",
        )
        for i in range(num_layers)
    ]
    self.final_norm = keras.layers.LayerNormalization(
        epsilon=norm_eps, name="final_norm"
    )

    self._vocab_size = vocab_size
    self._model_dim = model_dim
    self._scale_by_sqrt_dim = scale_by_sqrt_dim
    self._add_head = add_head
    self._config = {
        "vocab_size": vocab_size,
        "max_positions": max_positions,
        "model_dim": model_dim,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "norm_eps": norm_eps,
        "scale_by_sqrt_dim": scale_by_sqrt_dim,
        "add_head": add_head,
    }

  def build(self, inputs_shape: Sequence[int]):
    self.item_embedding.build(inputs_shape)
    self.position_embedding.build((*inputs_shape, self._model_dim))

    for decoder_block in self.decoder_blocks:
      decoder_block.build((*inputs_shape, self._model_dim))

    self.final_norm.build((*inputs_shape, self._model_dim))

  def call(
      self,
      inputs: Tensor,
      padding_mask: Tensor | None = None,
      attention_mask: Tensor | None = None,
      mask_positions: Tensor | None = None,
      training: bool = False,
  ) -> Tensor:
    embeddings = self.item_embedding(inputs)
    if self._scale_by_sqrt_dim:
      embeddings *= keras.ops.cast(
          self._model_dim**0.5, keras.ops.dtype(embeddings)
      )
    embeddings += self.position_embedding(embeddings)

    embeddings = self.final_norm(embeddings)
    embeddings = self.embeddings_dropout(embeddings, training=training)

    for decoder_block in self.decoder_blocks:
      embeddings = decoder_block(
          embeddings,
          decoder_padding_mask=padding_mask,
          decoder_attention_mask=attention_mask,
          training=training,
      )

    embeddings = self.final_norm(embeddings)

    if not self._add_head:
      return embeddings

    return self.item_embedding(embeddings, reverse=True)

  def compute_output_shape(self, inputs_shape: Sequence[int]) -> Sequence[int]:
    output_dim = self._vocab_size if self._add_head else self._model_dim
    return (*inputs_shape, output_dim)

  def get_config(self) -> Mapping[str, Any]:
    return {**super().get_config(), **self._config}
