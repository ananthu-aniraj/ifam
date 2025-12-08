# Compostion of the VisionTransformer class from timm with extra features: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from layers import EfficientProbing


class BaselineViTEP(torch.nn.Module):
    """
    Modifications:
    - Use PDiscoBlock instead of Block
    - Use PDiscoAttention instead of Attention
    - Return the mean of k over heads from attention
    - Option to use only class tokens or only patch tokens or both (concat) for classification
    """

    def __init__(self, init_model: torch.nn.Module, num_classes: int, pooling_type: str) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token

        self.patch_embed = init_model.patch_embed

        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm

        self.fc_norm = init_model.fc_norm
        if pooling_type == 'ep':
            self.efficient_pool = EfficientProbing(dim=init_model.embed_dim, num_queries=32)  # EP_32
        if pooling_type == "cls" or pooling_type == "gap" or pooling_type == "ep":
            self.head = nn.Linear(init_model.embed_dim, num_classes)
        elif pooling_type == "gap_cls":
            self.head = nn.Linear(init_model.embed_dim * 2, num_classes)
        else:
            raise ValueError("pooling_type must be one of 'cls', 'gap', 'gap_cls', 'ep'")
        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])
        self._init_weights_head()
        self.return_transformer_qkv = False

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def _init_weights_head(self):
        trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        # Forward pass through transformer
        x = self.norm_pre(x)

        x = self.blocks(x)

        x = self.norm(x)
        if self.pooling_type == "ep":
            x = self.efficient_pool(x[:, self.num_prefix_tokens:, :])
        elif self.pooling_type == "cls":
            x = x[:, 0, :]
        elif self.pooling_type == "gap":
            x = x[:, self.num_prefix_tokens:, :].mean(dim=1)
        else:
            x = torch.cat([x[:, 0, :], x[:, self.num_prefix_tokens:, :].mean(dim=1)], dim=1)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        # Classification head
        x = self.fc_norm(x)
        x = self.head(x)
        return x