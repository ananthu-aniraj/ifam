import torch
from timm.models.vision_transformer import Attention, Block
import torch.nn.functional as F
from torch.jit import Final


class MaskedBlock(Block):
    """
    Modifications:
        - Use MaskedAttention instead of Attention
    """

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaskedAttention(Attention):
    fused_attn: Final[bool]
    """
    Modifications: Enable the use of attention mask
    """

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attn_mask
            )
        else:
            raise NotImplementedError("Not implemented yet")

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
