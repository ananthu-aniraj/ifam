import torch
import torch.nn as nn
from typing import Tuple
from timm.models.vision_transformer import Attention, Block
from layers import MaskedBlock, MaskedAttention, PatchPruning, SPMaskedBlock, \
    SPMaskedAttention, SPBlock, SPAttention, SimPool, AttentionPoolLatentMod


class LateMaskViT(torch.nn.Module):
    """
    Modifications:
    - Use MaskedBlock instead of Block
    - Use MaskedAttention instead of Attention
    - Option to use only class tokens or only patch tokens or both (concat) for classification
    """

    def __init__(self, init_model: torch.nn.Module, num_classes: int,
                 pooling_type: str, reinit_fc_norm: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.num_prefix_tokens = init_model.num_prefix_tokens
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
        self.patch_prune = PatchPruning(keep_ratio=1, num_prefix_tokens=init_model.num_prefix_tokens)
        if pooling_type == "sim_pool":
            self.sim_pool = SimPool(dim=init_model.embed_dim)
        elif pooling_type == "attention_pool_latent":
            self.sim_pool = AttentionPoolLatentMod(
                in_features=init_model.embed_dim, embed_dim=init_model.embed_dim, num_heads=self.num_attention_heads,
                pool_type="token", norm_layer=nn.LayerNorm)
        else:
            self.sim_pool = None

        if reinit_fc_norm:
            self.fc_norm = nn.LayerNorm(init_model.embed_dim)
        else:
            self.fc_norm = init_model.fc_norm

        self.head = nn.Linear(init_model.embed_dim, num_classes)
        if isinstance(init_model.patch_embed.img_size, tuple):
            self.h_fmap = int(init_model.patch_embed.img_size[0] // init_model.patch_embed.patch_size[0])
            self.w_fmap = int(init_model.patch_embed.img_size[1] // init_model.patch_embed.patch_size[1])
        else:
            self.h_fmap = int(init_model.patch_embed.img_size // init_model.patch_embed.patch_size)
            self.w_fmap = int(init_model.patch_embed.img_size // init_model.patch_embed.patch_size)

        self.register_buffer("prefix_attn_mask", torch.ones(1, self.num_prefix_tokens))
        self.convert_blocks_and_attention()
        self._init_weights_head()

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

    def _init_weights_head(self):
        torch.nn.init.trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.)

    def forward(self, x: torch.Tensor, attn_mask_per_img: torch.Tensor) -> torch.Tensor | Tuple[
        torch.Tensor, torch.Tensor]:

        x = self.forward_features(x, attn_mask_per_img)
        x = self.forward_head(x)
        return x

    def forward_features(self, x: torch.Tensor, attn_mask_per_img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        num_keep = attn_mask_per_img.count_nonzero(dim=-1).max().item()  # [B]

        # Forward pass through transformer
        x = self.norm_pre(x)

        # Process each block
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # [B, N, C]
        x, keep_indices = self.patch_prune(x, attn_mask_per_img, num_keep=num_keep)
        attn_mask_per_img = attn_mask_per_img.gather(dim=-1, index=keep_indices)  # [B, num_keep]
        attn_mask_simpool = attn_mask_per_img.unsqueeze(1).unsqueeze(1).bool().contiguous()  # [B, 1, 1, N]
        x = x[:, self.num_prefix_tokens:, :]  # remove class tokens -> (B, N, C)

        # Apply late mask
        x = x * attn_mask_per_img.unsqueeze(-1)  # [B, num_keep, C]

        if self.pooling_type == "sim_pool" or self.pooling_type == "attention_pool_latent":
            x = self.sim_pool(x, attn_mask_simpool)
        elif self.pooling_type == "max_pool":
            x = x.amax(dim=1)
        elif self.pooling_type == "avg_pool":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"Pooling type {self.pooling_type} not recognized")
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_norm(x)
        x = self.head(x)
        return x


def test_late_mask_vit():
    from timm import create_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height = 518
    dinov2_vits14 = create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        drop_path_rate=0,
        img_size=height,
    )
    model = LateMaskViT(dinov2_vits14, num_classes=1000, pooling_type="sim_pool", reinit_fc_norm=True)
    model.to(device)
    x = torch.rand(8, 3, 518, 518).to(device)
    mask = (torch.rand(8, 1369) > 0.3).float().to(device)
    model(x, mask)


if __name__ == "__main__":
    test_late_mask_vit()
