import torch
from typing import Tuple
from timm.models.vision_transformer import Attention, Block
from layers import MaskedBlock, MaskedAttention, PatchPruning, SPMaskedBlock, \
    SPMaskedAttention, SPBlock, SPAttention, SimPool, AttentionPoolLatentMod


class AttnMaskViTTeacher(torch.nn.Module):
    """
    Modifications:
    - Use MaskedBlock instead of Block
    - Use MaskedAttention instead of Attention
    """

    def __init__(self, init_model: torch.nn.Module) -> None:
        super().__init__()
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.embed_dim = init_model.embed_dim
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token
        self.num_attention_heads = init_model.blocks[0].attn.num_heads

        self.patch_embed = init_model.patch_embed

        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.patch_prune = PatchPruning(keep_ratio=1, num_prefix_tokens=init_model.num_prefix_tokens)
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm

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
        x = self.forward_features(x, attn_mask_per_img)
        return x

    def forward_features(self, x: torch.Tensor, attn_mask_per_img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        num_keep = attn_mask_per_img.count_nonzero(dim=-1).max().item()

        x, keep_indices = self.patch_prune(x, attn_mask_per_img, num_keep=num_keep)
        attn_mask_per_img = attn_mask_per_img.gather(dim=-1, index=keep_indices)  # [B, num_keep]
        prefix_token_attn_mask = self.prefix_attn_mask.expand(x.shape[0], -1)  # [B, num_prefix_tokens]
        attn_mask = (
            (torch.cat([prefix_token_attn_mask, attn_mask_per_img], dim=-1)).unsqueeze(1).expand(-1, x.shape[1],
                                                                                                 -1).unsqueeze(
                1).bool().contiguous())  # [B, 1, N, N]
        # Forward pass through transformer
        x = self.norm_pre(x)

        # Process each block
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        x = x[:, 0, :]
        return x
