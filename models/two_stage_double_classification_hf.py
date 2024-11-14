import torch
from typing import List
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from .hf_transformers_dinov2 import DinoV2ModelAttnMask2Stage, DinoV2PDiscoHF


class FullTwoStageModelDoubleClassifyHF(torch.nn.Module):
    def __init__(self, init_model: torch.nn.Module, config: Dinov2Config, num_landmarks: int = 8,
                 num_classes: int = 200, softmax_temperature: float = 1.0, part_dropout: float = 0.3,
                 part_dropout_stage_2: float = 0.3,
                 part_logits_threshold: dict = None, use_soft_masks: bool = False) -> None:
        super().__init__()
        self.stage_1 = DinoV2PDiscoHF(config=config, num_landmarks=num_landmarks,
                                      num_classes=num_classes, softmax_temperature=softmax_temperature,
                                      part_dropout=part_dropout, part_dropout_stage_2=part_dropout_stage_2,
                                      part_logits_threshold=part_logits_threshold)
        self.stage_2 = DinoV2ModelAttnMask2Stage(config=config, num_classes=num_classes, use_soft_masks=use_soft_masks)
        self.stage_1.load_state_dict(init_model.state_dict(), strict=False)
        self.stage_2.load_state_dict(init_model.state_dict(), strict=False)
        self.feature_dim = config.hidden_size
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.orig_prefix_tokens = 1
        self.gumbel_softmax = True
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
                attn_maps_fg_full = attn_maps_fg.amax(dim=1, keepdim=True)  # [B, 1, H, W]
                attn_maps_fg_bg_soft = torch.cat([attn_maps_fg_full, attn_maps_bg],
                                                 dim=1)  # [B, 2, H, W]

                # Re-compute the hard attention maps
                index = attn_maps_fg_bg_soft.max(1, keepdim=True)[1]
                attn_maps_fg_bg = torch.zeros_like(attn_maps_fg_bg_soft,
                                                   memory_format=torch.legacy_contiguous_format).scatter_(
                    1, index,
                    1.0)  # [B, 2, H, W]
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
    # Load model directly
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    dinov2_model = AutoModel.from_pretrained("microsoft/rad-dino")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_model.to(device)
    num_classes = 20
    model = FullTwoStageModelDoubleClassifyHF(dinov2_model, dinov2_model.config, num_classes=num_classes,
                                              num_landmarks=7, use_soft_masks=True)
    model.to(device)
    x = torch.rand(2, 3, 518, 518).to(device)

    y = model(x)
    print(y[0].shape)  # torch.Size([8, 20])
    print(y[0][0])
    rand_gt = torch.randint(0, 20, (2,)).to(device)
    print(rand_gt.shape)
    loss = torch.nn.CrossEntropyLoss()(y[0], rand_gt)
    loss.backward()


if __name__ == "__main__":
    test_full_two_stage_model()
