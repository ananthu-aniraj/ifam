import torch
import torch.nn as nn
from typing import Tuple
from timm.models.vision_transformer import Attention, Block
from layers import MaskedBlock, MaskedAttention, SPMaskedBlock, \
    SPMaskedAttention, SPBlock, SPAttention, SimPool, AttentionPoolLatentMod


class LateMaskViTStudent(torch.nn.Module):
    """
    Modifications:
    - Use MaskedBlock instead of Block
    - Use MaskedAttention instead of Attention
    - Option to use only class tokens or only patch tokens or both (concat) for classification
    """

    def __init__(self, init_model: torch.nn.Module,
                 pooling_type: str) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.embed_dim = init_model.embed_dim
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token
        self.num_attention_heads = init_model.blocks[0].attn.num_heads

        self.patch_embed = init_model.patch_embed

        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm
        if pooling_type == "sim_pool":
            self.sim_pool = SimPool(dim=init_model.embed_dim)
        elif pooling_type == "attention_pool_latent":
            self.sim_pool = AttentionPoolLatentMod(
                in_features=init_model.embed_dim, embed_dim=init_model.embed_dim, num_heads=self.num_attention_heads,
                pool_type="token", norm_layer=nn.LayerNorm)
        else:
            self.sim_pool = None

        self.fc_norm = init_model.fc_norm
        if isinstance(init_model.patch_embed.img_size, tuple):
            self.h_fmap = int(init_model.patch_embed.img_size[0] // init_model.patch_embed.patch_size[0])
            self.w_fmap = int(init_model.patch_embed.img_size[1] // init_model.patch_embed.patch_size[1])
        else:
            self.h_fmap = int(init_model.patch_embed.img_size // init_model.patch_embed.patch_size)
            self.w_fmap = int(init_model.patch_embed.img_size // init_model.patch_embed.patch_size)

        self.register_buffer("prefix_attn_mask", torch.ones(1, self.num_prefix_tokens))
        self.convert_blocks_and_attention()

    def convert_blocks_and_attention(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.__class__ = MaskedBlock
            elif isinstance(module, Attention):
                module.__class__ = MaskedAttention
            elif isinstance(module, SPBlock):
                module.__class__ = SPMaskedBlock
            elif isinstance(module, SPAttention):
                module.__class__ = SPMaskedAttention

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

    def forward(self, x: torch.Tensor, attn_mask_per_img: torch.Tensor) -> torch.Tensor | Tuple[
        torch.Tensor, torch.Tensor]:

        x = self.forward_features(x)
        x = self.forward_masks(x, attn_mask_per_img)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        # Forward pass through transformer
        x = self.norm_pre(x)

        # Process each block
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # [B, N, C]
        return x

    def forward_masks(self, x: torch.Tensor, attn_mask_per_img: torch.Tensor) -> torch.Tensor:
        attn_mask_simpool = attn_mask_per_img.unsqueeze(1).unsqueeze(1).bool().contiguous()  # [B, 1, 1, N]
        x = x[:, self.num_prefix_tokens:, :]  # remove class tokens -> (B, N, C)

        # Apply late mask
        x = x * attn_mask_per_img.unsqueeze(-1)  # [B, N, C]

        if self.pooling_type == "sim_pool" or self.pooling_type == "attention_pool_latent":
            x = self.sim_pool(x, attn_mask_simpool)
        elif self.pooling_type == "max_pool":
            x = x.amax(dim=1)
        elif self.pooling_type == "avg_pool":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Pooling type {self.pooling_type} not recognized")
        return x
