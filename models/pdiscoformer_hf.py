from typing import Optional

import torch
from torch import nn
from transformers import Dinov2Config

from layers import Dinov2PreTrainedModel, Dinov2Embeddings, Dinov2Encoder


class DinoV2PDiscoOrigHF(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config, num_landmarks: int, num_classes: int, part_dropout: float,
                 softmax_temperature: float):
        super().__init__(config)
        self.config = config
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.noise_variance = 0.0
        self.num_prefix_tokens = 1
        self.gumbel_softmax = True
        self.modulation_orth = True
        self.classifier_type = "linear"
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_prefix_tokens = 1
        self.feature_dim = config.hidden_size
        self.h_fmap = int(config.image_size / config.patch_size)
        self.w_fmap = int(config.image_size / config.patch_size)
        self.unflatten = nn.Unflatten(1, (self.h_fmap, self.w_fmap))
        self.fc_landmarks = torch.nn.Conv2d(self.feature_dim, num_landmarks + 1, 1, bias=False)
        self.modulation = torch.nn.LayerNorm([self.feature_dim, self.num_landmarks + 1])
        self.dropout_full_landmarks = torch.nn.Dropout1d(part_dropout)
        self.fc_class_landmarks = torch.nn.Linear(in_features=self.feature_dim, out_features=num_classes,
                                                  bias=False)
        self.softmax_temperature = softmax_temperature

        self._init_weights_head()

    def _init_weights_head(self):
        # Initialize weights with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.fc_class_landmarks.weight, std=0.02)
        if self.fc_class_landmarks.bias is not None:
            torch.nn.init.zeros_(self.fc_class_landmarks.bias)

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

        x = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        x = self.layernorm(x)
        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps vit, a = convolution kernel
        batch_size = x.shape[0]
        x = x[:, self.num_prefix_tokens:, :]  # [B, num_patch_tokens, embed_dim]
        x = self.unflatten(x)  # [B, H, W, embed_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]

        ab = self.fc_landmarks(x)  # [B, num_landmarks + 1, H, W]
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1).contiguous()
        a_sq = self.fc_landmarks.weight.pow(2).sum(1, keepdim=True).expand(-1, batch_size, x.shape[-2],
                                                                           x.shape[-1]).contiguous()
        a_sq = a_sq.permute(1, 0, 2, 3).contiguous()

        dist = b_sq - 2 * ab + a_sq
        part_logits = -dist
        maps = torch.nn.functional.gumbel_softmax(part_logits, dim=1, tau=self.softmax_temperature,
                                                  hard=False)  # [B, num_landmarks + 1, H, W]

        # Use maps to get weighted average features per landmark
        all_features = (maps.unsqueeze(1) * x.unsqueeze(2)).contiguous()  # [B, embed_dim, num_landmarks + 1, H, W]

        all_features = all_features.mean(-1).mean(-1).contiguous()  # [B, embed_dim, num_landmarks + 1]

        all_features_mod = self.modulation(all_features)  # [B, embed_dim, num_landmarks + 1]
        # Classification based on the landmark features
        scores = self.fc_class_landmarks(
            self.dropout_full_landmarks(all_features_mod[..., :-1].permute(0, 2, 1).contiguous())).permute(0, 2,
                                                                                                           1).contiguous()  # [B, num_landmarks, num_classes]
        return all_features_mod, maps, scores, dist
