import torch
from typing import List
from .individual_landmark_vit_2_stage_mod import IndividualLandmarkViT2StageMod
from .attn_mask_vit_2_stage import AttnMaskViT2Stage


class FullTwoStageModelDoubleClassify(torch.nn.Module):
    def __init__(self, init_model_stage_1: torch.nn.Module, init_model_stage_2: torch.nn.Module, num_landmarks: int = 8,
                 num_classes: int = 200, return_transformer_qkv: bool = False, gumbel_softmax: bool = True,
                 softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 part_dropout: float = 0.3, part_dropout_stage_2: float = 0.3,
                 part_logits_threshold: dict = None, use_soft_masks: bool = False) -> None:
        super().__init__()

        self.stage_1 = IndividualLandmarkViT2StageMod(init_model=init_model_stage_1, num_landmarks=num_landmarks,
                                                      return_transformer_qkv=return_transformer_qkv,
                                                      gumbel_softmax=gumbel_softmax,
                                                      softmax_temperature=softmax_temperature,
                                                      gumbel_softmax_hard=gumbel_softmax_hard,
                                                      part_dropout=part_dropout,
                                                      part_dropout_stage_2=part_dropout_stage_2,
                                                      num_classes=num_classes,
                                                      modulation_type="layer_norm",
                                                      classifier_type="linear",
                                                      part_logits_threshold=part_logits_threshold)
        self.stage_2 = AttnMaskViT2Stage(init_model=init_model_stage_2, num_classes=num_classes,
                                         reinit_fc_norm=True, use_soft_masks=use_soft_masks)
        self.feature_dim = init_model_stage_2.embed_dim
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.orig_prefix_tokens = init_model_stage_2.num_prefix_tokens
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_softmax_hard = gumbel_softmax_hard
        self.use_soft_masks = use_soft_masks

    def forward(self, x: torch.Tensor, part_ids_to_remove: List[int] = None) -> torch.Tensor:
        all_features_mod, attn_maps_combined, attn_maps_fg_bg_combined, pred_stage_1, part_logits = self.stage_1(
            x)  # [B, num_landmarks+1, H, W]

        if not self.training:
            if part_ids_to_remove is not None:
                # Remove the part from the attention maps and recompute the attention maps
                attn_maps_fg = attn_maps_combined[:, :-1, :, :]  # [B, num_landmarks, H, W]

                attn_maps_argmax = attn_maps_combined.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
                # Find locations of the parts to remove
                mask = torch.zeros_like(attn_maps_argmax, dtype=torch.bool)
                for part_id in part_ids_to_remove:
                    mask = mask | (attn_maps_argmax == part_id)
                attn_maps_bg = attn_maps_combined[:, -1, :, :].unsqueeze(1)  # [B, 1, H, W]
                # Set activation of bg to inf in the locations of the parts to remove
                attn_maps_bg[mask] = float("inf")
                # Set activation of the parts to remove to -inf
                attn_maps_fg[:, part_ids_to_remove, :, :] = -float("inf")
                attn_maps_fg_full = attn_maps_fg.amax(dim=1, keepdim=True)  # [B, 1, H, W]
                attn_maps_fg_bg_soft = torch.cat([attn_maps_fg_full, attn_maps_bg],
                                                 dim=1)  # [B, 2, H, W]

                # Re-compute the hard attention maps
                index = attn_maps_fg_bg_soft.max(1, keepdim=True)[1]
                attn_maps_fg_bg = torch.zeros_like(attn_maps_fg_bg_soft,
                                                   memory_format=torch.legacy_contiguous_format).scatter_(
                    1, index,
                    1.0)  # [B, 2, H, W]
                attn_maps_fg_bg_combined = attn_maps_fg_bg, attn_maps_fg_bg_soft

                attn_maps_combined = torch.cat([attn_maps_fg, attn_maps_bg], dim=1)  # [B, num_landmarks+1, H, W]

            else:
                attn_maps_fg_bg, attn_maps_fg_bg_soft = attn_maps_fg_bg_combined
        else:
            attn_maps_fg_bg, attn_maps_fg_bg_soft = attn_maps_fg_bg_combined

        # Convert 2 channel attention maps to 1 channel attention maps
        if self.use_soft_masks:
            attn_maps_fg = attn_maps_fg_bg_soft[:, 0, :, :]  # [B, H, W]
        else:
            attn_maps_fg = attn_maps_fg_bg[:, 0, :, :]  # [B, H, W]
        mask_per_patch = attn_maps_fg.flatten(start_dim=1)  # [B, num_patches]

        pred_stage_2 = self.stage_2(x, attn_mask_per_img=mask_per_patch)  # [B, num_classes]
        return pred_stage_2, attn_maps_combined, attn_maps_fg_bg_combined, all_features_mod, pred_stage_1, part_logits


def test_full_two_stage_model():
    from timm import create_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height = 518
    dinov2_vits14 = create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        drop_path_rate=0,
        img_size=height,
    )
    num_classes = 20
    model = FullTwoStageModelDoubleClassify(dinov2_vits14, dinov2_vits14, num_classes=num_classes, num_landmarks=7,
                                            gumbel_softmax=True, use_soft_masks=True)
    model.to(device)
    model.eval()
    x = torch.rand(8, 3, 518, 518).to(device)
    with torch.no_grad():
        y = model(x, part_ids_to_remove=[1, 2])
    print(y[0].shape)  # torch.Size([8, 20])
    print(y[0][0])
    print(y[-1].shape)  # torch.Size([8, 20])


if __name__ == "__main__":
    test_full_two_stage_model()
