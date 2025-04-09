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
class BERT4Rec(keras.layers.Layer):
  """BERT4Rec architecture as in [1].

  Implements the BERT4Rec model architecture as described in 'BERT4Rec:
  Sequential Recommendation with Bidirectional Encoder Representations from
  Transformer' [1].

  [1] https://arxiv.org/abs/1904.06690
  """

  def __init__(
      self,
      *,
      vocab_size: int,
      max_positions: int,
      num_types: int | None = None,
      model_dim: int,
      mlp_dim: int,
      num_heads: int,
      num_layers: int,
      dropout: float = 0.0,
      norm_eps: float = 1e-12,
      add_head: bool = True,
      **kwargs,
  ):
    """Initializes the instance.

    Args:
      vocab_size: The size of the item vocabulary.
      max_positions: The maximum number of positions in a sequence.
      num_types: The number of types. If None, no type embedding is used.
        Defaults to None.
      model_dim: The width of the embeddings in the model.
      mlp_dim: The width of the MLP in each transformer block.
      num_heads: The number of attention heads in each transformer block.
      num_layers: The number of transformer blocks in the model.
      dropout: The dropout rate. Defaults to 0.
      norm_eps: The epsilon for layer normalization.
      add_head: Whether to add a masked language modeling head.
      **kwargs: Passed through to the super class.
    """

    super().__init__(**kwargs)

    self.item_embedding = keras_hub.layers.ReversibleEmbedding(
        input_dim=vocab_size,
        output_dim=model_dim,
        embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        dtype=self.dtype_policy,
        reverse_dtype=self.compute_dtype,
        name="item_embedding",
    )
    if num_types is not None:
      self.type_embedding = keras.layers.Embedding(
          input_dim=num_types,
          output_dim=model_dim,
          embeddings_initializer=keras.initializers.TruncatedNormal(
              stddev=0.02
          ),
          dtype=self.dtype_policy,
          name="type_embedding",
      )
    else:
      self.type_embedding = None

    self.position_embedding = keras_hub.layers.PositionEmbedding(
        sequence_length=max_positions,
        initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        dtype=self.dtype_policy,
        name="position_embedding",
    )

    self.embeddings_norm = keras.layers.LayerNormalization(
        epsilon=1e-12, name="embedding_norm"
    )
    self.embeddings_dropout = keras.layers.Dropout(
        dropout, name="embedding_dropout"
    )

    self.encoder_blocks = [
        keras_hub.layers.TransformerEncoder(
            intermediate_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            activation=utils.gelu_approximate,
            layer_norm_epsilon=norm_eps,
            normalize_first=False,
            dtype=self.dtype_policy,
            name=f"encoder_block_{i}",
        )
        for i in range(num_layers)
    ]
    if add_head:
      self.head = keras_hub.layers.MaskedLMHead(
          vocabulary_size=vocab_size,
          token_embedding=self.item_embedding,
          intermediate_activation=utils.gelu_approximate,
          kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
          dtype=self.dtype_policy,
          name="mlm_head",
      )
    else:
      self.head = None

    self._vocab_size = vocab_size
    self._model_dim = model_dim
    self._config = {
        "vocab_size": vocab_size,
        "max_positions": max_positions,
        "num_types": num_types,
        "model_dim": model_dim,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "norm_eps": norm_eps,
        "add_head": add_head,
    }

  def build(self, inputs_shape: Sequence[int]):
    self.item_embedding.build(inputs_shape)
    if self.type_embedding is not None:
      self.type_embedding.build(inputs_shape)

    self.position_embedding.build((*inputs_shape, self._model_dim))
    self.embeddings_norm.build((*inputs_shape, self._model_dim))

    for encoder_block in self.encoder_blocks:
      encoder_block.build((*inputs_shape, self._model_dim))

    if self.head is not None:
      self.head.build((*inputs_shape, self._model_dim))

  def call(
      self,
      inputs: Tensor,
      type_ids: Tensor | None = None,
      padding_mask: Tensor | None = None,
      attention_mask: Tensor | None = None,
      mask_positions: Tensor | None = None,
      training: bool = False,
  ) -> Tensor:
    embeddings = self.item_embedding(inputs)
    if self.type_embedding is not None:
      if type_ids is None:
        raise ValueError(
            "`type_ids` cannot be None when `num_types` is not None."
        )
      embeddings += self.type_embedding(type_ids)
    embeddings += self.position_embedding(embeddings)

    embeddings = self.embeddings_norm(embeddings)
    embeddings = self.embeddings_dropout(embeddings, training=training)

    for encoder_block in self.encoder_blocks:
      embeddings = encoder_block(
          embeddings,
          padding_mask=padding_mask,
          attention_mask=attention_mask,
          training=training,
      )

    if self.head is None:
      return embeddings

    return self.head(embeddings, mask_positions)

  def compute_output_shape(
      self,
      inputs_shape: Sequence[int],
      mask_positions_shape: Tensor | None = None,
  ) -> Sequence[int | None]:
    if self.head is not None:
      if mask_positions_shape is None:
        raise ValueError(
            "`mask_positions_shape` cannot be None when `add_head` is True."
        )
      return (*inputs_shape[:-1], mask_positions_shape[-1], self._vocab_size)
    return (*inputs_shape, self._model_dim)

  def get_config(self) -> Mapping[str, Any]:
    return {**super().get_config(), **self._config}
