# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DINOv2 model."""

from typing import Dict, List, Optional

import torch
from torch import nn

from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from layers import Dinov2Embeddings, Dinov2Encoder, Dinov2PatchEmbeddings, Dinov2PreTrainedModel


class BaselineDInoV2HF(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config, num_classes: int = 200):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.num_prefix_tokens = 1
        self.h_fmap = int(config.image_size / config.patch_size)
        self.w_fmap = int(config.image_size / config.patch_size)
        self.num_classes = num_classes
        self.num_prefix_tokens = 1
        self._init_weights_head()

    def _init_weights_head(self):
        torch.nn.init.trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.)

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ):

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)

        sequence_output = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        sequence_output = self.layernorm(sequence_output)
        cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        cls_output = self.head(cls_output)

        return cls_output


