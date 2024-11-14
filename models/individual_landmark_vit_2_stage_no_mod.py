# Compostion of the VisionTransformer class from timm with extra features: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Union, Sequence

from timm.models.vision_transformer import Block, Attention
from utils.misc_utils import compute_attention, gumbel_softmax_straight_through_custom

from layers import BlockWQKVReturn, AttentionWQKVReturn


class IndividualLandmarkViTNoMod(torch.nn.Module):

    def __init__(self, init_model: torch.nn.Module, num_landmarks: int = 8,
                 return_transformer_qkv: bool = False, gumbel_softmax: bool = True,
                 softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 part_dropout: float = 0.3) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token

        self.feature_dim = init_model.embed_dim
        self.patch_embed = init_model.patch_embed
        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm
        self.return_transformer_qkv = return_transformer_qkv
        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])

        self.unflatten = nn.Unflatten(1, (self.h_fmap, self.w_fmap))
        self.fc_landmarks = torch.nn.Conv2d(self.feature_dim, num_landmarks + 1, 1, bias=False)
        self.part_dropout = part_dropout
        self.gumbel_softmax = gumbel_softmax
        self.softmax_temperature = softmax_temperature
        self.gumbel_softmax_hard = gumbel_softmax_hard

        self.convert_blocks_and_attention()

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

    def forward(self, x: Tensor, hard_softmax: bool = False) -> Tensor:

        if self.gumbel_softmax_hard:
            hard_softmax = True

        # Patch Embedding
        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        # Forward pass through transformer
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps vit, a = convolution kernel
        x = x[:, self.num_prefix_tokens:, :]  # [B, num_patch_tokens, embed_dim]
        x = self.unflatten(x)  # [B, H, W, embed_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]
        ab = self.fc_landmarks(x)  # [B, num_landmarks + 1, H, W]
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, ab.shape[1], -1, -1).contiguous()
        a_sq = self.fc_landmarks.weight.pow(2).sum(1, keepdim=True).expand(-1, x.shape[0], x.shape[-2],
                                                                           x.shape[-1]).contiguous()
        a_sq = a_sq.permute(1, 0, 2, 3).contiguous()

        dist = b_sq - 2 * ab + a_sq
        maps = -dist  # [B, num_landmarks + 1, H, W]

        softmax_temp = self.softmax_temperature
        # Softmax so that the attention maps for each pixel add up to 1
        if hard_softmax:
            maps_hard, maps_soft, maps_hard_fg_bg, maps_soft_fg_bg = gumbel_softmax_straight_through_custom(maps, dim=1,
                                                                                                            tau=softmax_temp,
                                                                                                            training_mode=self.training,
                                                                                                            part_dropout=self.part_dropout)
            maps = (maps_hard, maps_soft)
            maps_fg_bg = (maps_hard_fg_bg, maps_soft_fg_bg)
        else:
            if self.gumbel_softmax:
                maps = torch.nn.functional.gumbel_softmax(maps, dim=1, tau=softmax_temp,
                                                          hard=self.gumbel_softmax_hard)
            else:
                maps = torch.nn.functional.softmax(maps / softmax_temp, dim=1)  # [B, num_landmarks + 1, H, W]
            maps_fg = maps[:, :-1, :, :]
            maps_bg = maps[:, -1, :, :]
            maps_fg = torch.nn.functional.dropout2d(maps_fg, p=self.part_dropout,
                                                    training=self.training)  # [B, num_landmarks, H, W]
            if self.training:
                # Rescale the foreground maps so that the sum of the foreground and background maps is 1
                maps_fg = maps_fg * (1 - self.part_dropout)
            maps_fg_full = maps_fg.amax(dim=1, keepdim=True)  # [B, 1, H, W]
            maps_fg_bg = torch.cat([maps_fg_full, maps_bg.unsqueeze(1)], dim=1)  # [B, 2, H, W]

            maps = torch.cat([maps_fg, maps_bg.unsqueeze(1)], dim=1)  # [B, num_landmarks + 1, H, W]
        return maps, maps_fg_bg

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


def test_landmark_vit():
    from timm import create_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height = 518
    dinov2_vits14 = create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        drop_path_rate=0,
        img_size=height,
    )
    model = IndividualLandmarkViTNoMod(dinov2_vits14, gumbel_softmax=False, num_landmarks=8)
    model.to(device)
    # from data_sets import FineGrainedBirdClassificationDataset
    # from torchvision import transforms
    #
    # test_transforms = transforms.Compose([
    #     transforms.Resize(height),
    #     transforms.CenterCrop(height),
    #     transforms.ToTensor()
    # ])
    # image_sub_path = "images"
    # dataset_test = FineGrainedBirdClassificationDataset("/user/aaniraj/home/Documents/Projects/data/cub200/CUB_200_2011", mode="test",
    #                                                     transform=test_transforms,
    #                                                     image_sub_path=image_sub_path)
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=4)
    # x = next(iter(dataloader_test))[0].to(device)
    x = torch.rand(8, 3, 518, 518).to(device)
    maps, constant_tensor = model(x)


if __name__ == "__main__":
    test_landmark_vit()
