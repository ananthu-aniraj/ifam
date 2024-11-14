# Compostion of the VisionTransformer class from timm with extra features: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import torch
import torch.nn as nn
from typing import Tuple, Union, Sequence, Any
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block, Attention
from layers import BlockWQKVReturn, AttentionWQKVReturn, SimPool, AttentionPoolLatentMod

from utils.misc_utils import compute_attention


class LinearProbeViT(torch.nn.Module):
    """
    Modifications:
    - Use PDiscoBlock instead of Block
    - Use PDiscoAttention instead of Attention
    - Return the mean of k over heads from attention
    - Option to use only class tokens or only patch tokens or both (concat) for classification
    """

    def __init__(self, init_model: torch.nn.Module, num_classes: int,
                 pooling_type: str, reinit_fc_norm: bool = False,
                 return_transformer_qkv: bool = False) -> None:
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
        self.part_embed = nn.Identity()
        self.patch_prune = nn.Identity()
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

        self.return_transformer_qkv = return_transformer_qkv
        self.convert_blocks_and_attention()
        self._init_weights_head()

    def convert_blocks_and_attention(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.__class__ = BlockWQKVReturn
            elif isinstance(module, Attention):
                module.__class__ = AttentionWQKVReturn

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

    def forward(self, x: torch.Tensor, part_loc: torch.Tensor = None) -> torch.Tensor | Tuple[
        torch.Tensor, torch.Tensor]:
        x = self.forward_features(x, part_loc)
        x = self.forward_head(x)
        return x

    def forward_features(self, x: torch.Tensor, part_loc: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        x = self.part_embed(x)
        x = self.patch_prune(x)

        # Forward pass through transformer
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # (B, total_tokens, C)

        # Scale part location to match the size of the feature map
        if part_loc is not None:
            token_loc = ((part_loc[:, 0] * self.h_fmap) * (part_loc[:, 1] * self.w_fmap)).long()  # (B,)
            token_loc = token_loc - 1
        else:
            token_loc = torch.randint(0, x.shape[1] - self.num_prefix_tokens - 1, (x.shape[0],))  # (B,)

        # Classification head
        if self.pooling_type == "cls_token":
            x = x[:, 0, :]
        else:
            x = x[:, self.num_prefix_tokens:, :]  # remove class tokens -> (B, N, C)
            if self.pooling_type == "sim_pool" or self.pooling_type == "attention_pool_latent":
                x = self.sim_pool(x)
            elif self.pooling_type == "max_pool":
                x = x.amax(dim=1)
            elif self.pooling_type == "specific_patch_probe":
                token_loc = token_loc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, 1, 1) -> (B, 1, C)
                x = x.gather(1, token_loc).squeeze(1)  # (B, N, C) -> (B, C)
            elif self.pooling_type == "avg_pool":
                x = x.mean(dim=1)
            else:
                raise ValueError(f"Pooling type {self.pooling_type} not recognized")
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_norm(x)
        x = self.head(x)
        return x

    def get_specific_intermediate_layer(
            self,
            x: torch.Tensor,
            n: int = 1,
            return_qkv: bool = False,
            return_att_weights: bool = False,
    ):
        num_blocks = len(self.blocks)
        attn_weights = []
        if n >= num_blocks:
            raise ValueError(f"n must be less than {num_blocks}")

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        if n == -1:
            if return_qkv:
                raise ValueError("take_indice cannot be -1 if return_transformer_qkv is True")
            else:
                return x

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)

                if return_att_weights:
                    attn_weight, _ = compute_attention(qkv)
                    attn_weights.append(attn_weight.detach())
            else:
                x = blk(x)
            if i == n:
                output = x.clone()
                if self.return_transformer_qkv and return_qkv:
                    qkv_output = qkv.clone()
                break
        if self.return_transformer_qkv and return_qkv and return_att_weights:
            return output, qkv_output, attn_weights
        elif self.return_transformer_qkv and return_qkv:
            return output, qkv_output
        elif self.return_transformer_qkv and return_att_weights:
            return output, attn_weights
        else:
            return output

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        if self.return_transformer_qkv:
            qkv_outputs = []
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)
            else:
                x = blk(x)
            if i in take_indices:
                outputs.append(x)
                if self.return_transformer_qkv:
                    qkv_outputs.append(qkv)
        if self.return_transformer_qkv:
            return outputs, qkv_outputs
        else:
            return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> tuple[tuple, Any]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        if self.return_transformer_qkv:
            outputs, qkv = self._intermediate_layers(x, n)
        else:
            outputs = self._intermediate_layers(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return_out = tuple(zip(outputs, prefix_tokens))
        else:
            return_out = tuple(outputs)

        if self.return_transformer_qkv:
            return return_out, qkv
        else:
            return return_out
