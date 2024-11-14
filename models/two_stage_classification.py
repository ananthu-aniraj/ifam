import torch
from .individual_landmark_vit_2_stage_no_mod import IndividualLandmarkViTNoMod
from .attn_mask_vit_2_stage_parallel import AttnMaskViTParallel2StageModulation


class FullTwoStageModel(torch.nn.Module):
    def __init__(self, init_model_stage_1: torch.nn.Module, init_model_stage_2: torch.nn.Module, num_landmarks: int = 8,
                 num_classes: int = 200,
                 return_transformer_qkv: bool = False, gumbel_softmax: bool = True,
                 softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 part_dropout: float = 0.3) -> None:
        super().__init__()
        self.stage_1 = IndividualLandmarkViTNoMod(init_model=init_model_stage_1, num_landmarks=num_landmarks,
                                                  return_transformer_qkv=return_transformer_qkv,
                                                  gumbel_softmax=gumbel_softmax,
                                                  softmax_temperature=softmax_temperature,
                                                  gumbel_softmax_hard=gumbel_softmax_hard,
                                                  part_dropout=part_dropout)
        self.stage_2 = AttnMaskViTParallel2StageModulation(init_model=init_model_stage_2, num_classes=num_classes,
                                                           num_copies_cls=2)
        self.feature_dim = init_model_stage_2.embed_dim
        self.num_landmarks = num_landmarks
        self.num_copy_cls = 2
        self.num_classes = num_classes
        self.orig_prefix_tokens = init_model_stage_2.num_prefix_tokens
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_softmax_hard = gumbel_softmax_hard

    def forward(self, x: torch.Tensor, hard_softmax: bool = False) -> torch.Tensor:
        if self.gumbel_softmax_hard:
            hard_softmax = True

        attn_maps_combined, attn_maps_fg_bg_combined = self.stage_1(x, hard_softmax)  # [B, num_landmarks+1, H, W]
        if hard_softmax:
            attn_maps, attn_maps_soft = attn_maps_combined
            attn_maps_fg_bg, attn_maps_fg_bg_soft = attn_maps_fg_bg_combined
        else:
            attn_maps = attn_maps_combined
            attn_maps_fg_bg = attn_maps_fg_bg_combined

        attn_maps_f = attn_maps_fg_bg.flatten(start_dim=2).permute(0, 2,
                                                                   1).contiguous()  # [B, num_patches, num_landmarks+1]
        mask_per_patch = attn_maps_f @ attn_maps_f.permute(0, 2, 1)  # [B, num_patches, num_patches]
        mask_per_prefix = torch.repeat_interleave(attn_maps_f, self.orig_prefix_tokens,
                                                  dim=-1).contiguous()  # [B, num_patches, (num_landmarks+1 * num_prefix_tokens)]
        x, all_features_mod = self.stage_2(x, attn_mask_per_img_patches=mask_per_patch,
                                           attn_mask_per_img_prefix=mask_per_prefix, hard_softmax=hard_softmax)  # [B, num_classes]
        return x, attn_maps_combined, attn_maps_fg_bg_combined, all_features_mod


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
    model = FullTwoStageModel(dinov2_vits14, dinov2_vits14, num_classes=num_classes, num_landmarks=7,
                              gumbel_softmax=True)
    model.to(device)
    x = torch.rand(8, 3, 518, 518).to(device)
    with torch.no_grad():
        y = model(x)
    print(y[0].shape)  # torch.Size([8, 20])


if __name__ == "__main__":
    test_full_two_stage_model()
