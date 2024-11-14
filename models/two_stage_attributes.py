import torch
from .individual_landmark_vit_2_stage_no_mod import IndividualLandmarkViTNoMod
from .attn_mask_vit_2_stage_independent_mlp_parallel import AttnMaskViTParallel2StageIndependentMLP


class FullTwoStageModelAttributes(torch.nn.Module):
    def __init__(self, init_model_stage_1: torch.nn.Module, init_model_stage_2: torch.nn.Module, num_landmarks: int = 8,
                 num_att_per_part=None,
                 return_transformer_qkv: bool = False, gumbel_softmax: bool = True,
                 softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 dropout_rate: float = 0.3, use_part_attention: bool = False) -> None:
        super().__init__()
        self.stage_1 = IndividualLandmarkViTNoMod(init_model=init_model_stage_1, num_landmarks=num_landmarks,
                                                  return_transformer_qkv=return_transformer_qkv,
                                                  gumbel_softmax=gumbel_softmax,
                                                  softmax_temperature=softmax_temperature,
                                                  gumbel_softmax_hard=gumbel_softmax_hard)
        self.stage_2 = AttnMaskViTParallel2StageIndependentMLP(init_model=init_model_stage_2,
                                                               num_att_per_part=num_att_per_part,
                                                               num_copies_cls=num_landmarks + 1,
                                                               dropout_rate=dropout_rate, use_part_attention=use_part_attention)
        self.feature_dim = init_model_stage_2.embed_dim
        self.num_landmarks = num_landmarks
        self.num_copy_cls = num_landmarks + 1
        self.num_att_per_part = num_att_per_part
        self.total_num_attributes = sum(num_att_per_part)
        self.orig_prefix_tokens = init_model_stage_2.num_prefix_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_maps = self.stage_1(x)  # [B, num_landmarks+1, H, W]
        attn_maps_f = attn_maps.flatten(start_dim=2).permute(0, 2,
                                                             1).contiguous()  # [B, num_patches, num_landmarks+1]
        mask_per_patch = attn_maps_f @ attn_maps_f.permute(0, 2, 1)  # [B, num_patches, num_patches]
        mask_per_prefix = torch.repeat_interleave(attn_maps_f, self.orig_prefix_tokens,
                                                  dim=-1).contiguous()  # [B, num_patches, (num_landmarks+1 * num_prefix_tokens)]
        mask_per_patch = torch.log(
            mask_per_patch)  # 1 gets mapped to 0 in log space and 0 gets mapped to -inf
        mask_per_prefix = torch.log(
            mask_per_prefix)  # 1 gets mapped to 0 in log space and 0 gets mapped to -inf
        x = self.stage_2(x, attn_mask_per_img_patches=mask_per_patch,
                         attn_mask_per_img_prefix=mask_per_prefix)  # [B, num_landmarks, num_classes]
        return x, attn_maps


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
    num_att_per_part = [10] * 7  # 8 landmarks
    model = FullTwoStageModelAttributes(dinov2_vits14, dinov2_vits14, num_att_per_part=num_att_per_part, num_landmarks=7, gumbel_softmax=True, use_part_attention=True)
    model.to(device)
    x = torch.rand(8, 3, 518, 518).to(device)
    with torch.no_grad():
        y = model(x)
    print(y[0].shape)  # torch.Size([8, 70])


if __name__ == "__main__":
    test_full_two_stage_model()
