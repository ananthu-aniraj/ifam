from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import Dinov2Config

from layers import Dinov2PreTrainedModel, Dinov2Embeddings, Dinov2Encoder, Dinov2PatchEmbeddings


class DinoV2ModelAttnMask2Stage(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config, num_classes: int = 200, use_soft_masks: bool = False):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes, bias=False)
        self.num_prefix_tokens = 1
        self.h_fmap = int(config.image_size / config.patch_size)
        self.w_fmap = int(config.image_size / config.patch_size)
        self.use_soft_masks = use_soft_masks
        self.register_buffer("prefix_attn_mask", torch.ones(1, self.num_prefix_tokens))
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
            attn_mask_per_img: Optional[torch.Tensor] = None,
    ):

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)
        prefix_token_attn_mask = self.prefix_attn_mask.expand(embedding_output.shape[0], -1)  # [B, num_prefix_tokens]
        if attn_mask_per_img is not None:
            if self.use_soft_masks:
                attn_mask = (
                    (torch.cat([prefix_token_attn_mask, attn_mask_per_img], dim=-1)).unsqueeze(1).expand(-1,
                                                                                                         embedding_output.shape[
                                                                                                             1],
                                                                                                         -1).unsqueeze(
                        1).contiguous())  # [B, 1, N, N]
                attn_mask = torch.log(attn_mask.float() + 1e-10)

            else:
                attn_mask = (
                    (torch.cat([prefix_token_attn_mask, attn_mask_per_img], dim=-1)).unsqueeze(1).expand(-1,
                                                                                                         embedding_output.shape[
                                                                                                             1],
                                                                                                         -1).unsqueeze(
                        1).bool().contiguous())  # [B, 1, N, N]
        else:
            attn_mask = None

        sequence_output = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            attn_mask=attn_mask,
        )
        sequence_output = self.layernorm(sequence_output)
        cls_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        cls_output = self.head(self.fc_norm(cls_output))

        return cls_output
