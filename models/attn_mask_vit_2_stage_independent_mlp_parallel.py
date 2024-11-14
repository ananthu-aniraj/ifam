import torch
import torch.nn as nn
from typing import Tuple, List

from timm.models.vision_transformer import Attention, Block

from timm.layers import AttentionPoolLatent

from layers import MaskedBlock, MaskedAttention, SPMaskedBlock, \
    SPMaskedAttention, SPBlock, SPAttention, IndependentMLPsVariableOutputPredictor, BlockWQKVReturn


class AttnMaskViTParallel2StageIndependentMLP(torch.nn.Module):
    """
    Modifications:
    - Use MaskedBlock instead of Block
    - Use MaskedAttention instead of Attention
    - Option to use only class tokens or only patch tokens or both (concat) for classification
    """

    def __init__(self, init_model: torch.nn.Module, num_att_per_part: List[int],
                 num_copies_cls: int = 1, dropout_rate: float = 0.1, use_part_attention: bool = False) -> None:
        super().__init__()
        self.num_att_per_part = num_att_per_part
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.feature_dim = init_model.embed_dim
        # self.cls_token = init_model.cls_token
        # self.reg_token = init_model.reg_token
        self.mlp_hidden_dim = init_model.blocks[0].mlp.fc1.out_features
        self.num_heads = init_model.blocks[0].attn.num_heads
        self.mlp_ratio = int(self.mlp_hidden_dim // self.feature_dim)
        self.use_part_attention = use_part_attention
        self.num_copies_cls = num_copies_cls
        self.orig_prefix_tokens = init_model.num_prefix_tokens
        if self.has_class_token:
            self.cls_token_c = nn.ParameterList([init_model.cls_token for _ in range(num_copies_cls)])
            self.num_prefix_tokens = num_copies_cls
        if self.num_reg_tokens > 0:
            self.reg_token_c = nn.ParameterList([init_model.reg_token for _ in range(num_copies_cls)])
            self.num_prefix_tokens += num_copies_cls * self.num_reg_tokens

        self.patch_embed = init_model.patch_embed
        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm
        self.modulation = torch.nn.LayerNorm([self.num_copies_cls, self.feature_dim])
        # if use_part_attention:
        #     self.part_attention = BlockWQKVReturn(dim=self.feature_dim, num_heads=self.num_heads,
        #                                           mlp_ratio=self.mlp_ratio, drop_path=dropout_rate, init_values=1e-5)
        # else:
        #     self.part_attention = nn.Identity()
        #
        # self.head = IndependentMLPsVariableOutputPredictor(part_dim=self.num_copies_cls - 1,
        #                                                    latent_dim=init_model.embed_dim, out_dims=num_att_per_part,
        #                                                    dropout=dropout_rate)
        self.cross_att = AttentionPoolLatent(in_features=self.feature_dim, out_features=self.feature_dim,
                                             embed_dim=self.feature_dim, num_heads=self.num_heads,
                                             latent_len=1, latent_dim=self.feature_dim,
                                             pool_type='token', norm_layer=nn.LayerNorm, drop=dropout_rate)
        self.head_norm = torch.nn.LayerNorm(self.feature_dim)
        self.head = nn.Linear(self.feature_dim, sum(num_att_per_part), bias=False)
        if isinstance(init_model.patch_embed.img_size, tuple):
            self.h_fmap = int(init_model.patch_embed.img_size[0] // init_model.patch_embed.patch_size[0])
            self.w_fmap = int(init_model.patch_embed.img_size[1] // init_model.patch_embed.patch_size[1])
        else:
            self.h_fmap = int(init_model.patch_embed.img_size // init_model.patch_embed.patch_size)
            self.w_fmap = int(init_model.patch_embed.img_size // init_model.patch_embed.patch_size)

        # Create block diagonal matrix for prefix token attention mask
        prefix_attn_masks = []

        # Create copies
        for i in range(num_copies_cls):
            prefix_attn_masks.append(
                torch.ones((self.orig_prefix_tokens, self.orig_prefix_tokens)))

        # Create block diagonal matrix
        prefix_attn_mask = torch.block_diag(*prefix_attn_masks).unsqueeze(
            0)  # [1, num_prefix_tokens, num_prefix_tokens]
        prefix_attn_mask[prefix_attn_mask == 0] = float("-inf")
        prefix_attn_mask[prefix_attn_mask == 1] = 0
        self.register_buffer("prefix_attn_mask", prefix_attn_mask)
        cls_token_indices = []
        for i in range(self.num_copies_cls):
            cls_token_indices.append(i * self.orig_prefix_tokens)
        cls_token_indices = torch.tensor(cls_token_indices)
        self.register_buffer("cls_token_indices", cls_token_indices)
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
        # if self.cls_token is not None:
        #     to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        # if self.reg_token is not None:
        #     to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.has_class_token:
            for i in range(self.num_copies_cls):
                to_cat.append(self.cls_token_c[i].expand(x.shape[0], -1, -1))
                if self.num_reg_tokens > 0:
                    to_cat.append(self.reg_token_c[i].expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            raise NotImplementedError("Not implemented yet")
        return self.pos_drop(x)

    def _init_weights_head(self):
        if self.head.__class__.__name__ == "IndependentMLPsVariableOutputPredictor" or \
                self.head.__class__.__name__ == "IndependentMLPs":
            self.head.reset_weights()
        else:
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        if self.use_part_attention:
            for m in self.part_attention.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, attn_mask_per_img_patches: torch.Tensor,
                attn_mask_per_img_prefix: torch.Tensor) -> torch.Tensor | Tuple[
        torch.Tensor, torch.Tensor]:

        x = self.forward_features(x, attn_mask_per_img_patches, attn_mask_per_img_prefix)

        # Classification head
        x = self.forward_head(x)
        return x

    def forward_features(self, x: torch.Tensor, attn_mask_per_img_patches: torch.Tensor,
                         attn_mask_per_img_prefix: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: [B, C, H, W]
            attn_mask_per_img_patches: [B, N_patches, N_patches]
            attn_mask_per_img_prefix: [B, N_patches, num_copy_cls]

        Returns:

        """
        x = self.patch_embed(x)

        prefix_token_attn_mask = self.prefix_attn_mask.expand(x.shape[0], -1,
                                                              -1)  # [B, num_prefix_tokens, num_prefix_tokens]
        cls_token_indices = self.cls_token_indices.unsqueeze(0).expand(x.shape[0], -1)  # [B, num_copy_cls]
        prefix_token_patch_mask = prefix_token_attn_mask @ attn_mask_per_img_prefix.permute(0, 2,
                                                                                            1)  # [B, num_prefix_tokens, N_patches]

        prefix_token_full_mask = torch.cat([prefix_token_attn_mask, prefix_token_patch_mask],
                                           dim=-1).contiguous()  # [B, num_prefix_tokens, N]

        patches_full_mask = torch.cat([prefix_token_patch_mask.transpose(1, 2), attn_mask_per_img_patches],
                                      dim=-1).contiguous()  # [B, N_patches, N]
        attn_mask_per_img = torch.cat([prefix_token_full_mask, patches_full_mask],
                                      dim=1).bool().contiguous()  # [B, N, N]
        attn_mask_per_img = attn_mask_per_img.unsqueeze(1).contiguous()  # [B, 1, N, N]

        # Position Embedding
        x = self._pos_embed(x)

        # Forward pass through transformer
        x = self.norm_pre(x)

        # Process each block
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask_per_img)

        x = self.norm(x)  # [B, N, C]
        x = x.gather(1, index=cls_token_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]).clone())
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x_mod = self.modulation(x)[:, :-1, :]  # [B, num_copy_cls-1, C]
        x_mod = self.cross_att(x_mod)  # [B, num_copy_cls-1, C]
        x_final = self.head(self.head_norm(x_mod))  # [B, num_copy_cls-1, num_classes]
        return x_final


def test_attn_mask_vit():
    from timm import create_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height = 518
    dinov2_vits14 = create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        drop_path_rate=0,
        img_size=height,
    )
    num_att_per_part = [10] * 8  # 8 landmarks
    model = AttnMaskViTParallel2StageIndependentMLP(dinov2_vits14, num_att_per_part=num_att_per_part,
                                                    num_copies_cls=9, use_part_attention=True)
    model.to(device)
    x = torch.rand(8, 3, 518, 518).to(device)
    maps = torch.randn(8, 1369, 9).to(device)
    # argmax_map = torch.argmax(map, dim=1)  # [B, 1369]
    # mask = F.one_hot(argmax_map).float().to(device)  # [B, 1369, 9]
    soft_maps = torch.softmax(maps, dim=-1)  # [B, 1369, 9]
    mask_per_patch = soft_maps @ soft_maps.permute(0, 2, 1)  # [B, 1369, 1369]
    mask = torch.repeat_interleave(soft_maps, model.orig_prefix_tokens, dim=-1).contiguous()  # [B, 1369, 45]
    mask_per_patch = torch.log(mask_per_patch)  # 1 gets mapped to 0 in log space and 0 gets mapped to -inf
    mask = torch.log(mask)  # 1 gets mapped to 0 in log space and 0 gets mapped to -inf
    model(x, attn_mask_per_img_patches=mask_per_patch, attn_mask_per_img_prefix=mask)


if __name__ == "__main__":
    test_attn_mask_vit()
