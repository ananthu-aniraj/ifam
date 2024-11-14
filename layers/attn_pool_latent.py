# Modified version of the Attention Pooling Layer from the timm library.
# Reference: https://github.com/huggingface/pytorch-image-models/blob/e741370e2b95e0c2fa3e00808cd9014ee620ca62/timm/layers/attention_pool.py#L12
import torch
import torch.nn.functional as F
from timm.layers import AttentionPoolLatent


class AttentionPoolLatentMod(AttentionPoolLatent):
    """
    Modifications:
        - Return the qkv tensors from the attention
    """

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_bias = torch.zeros_like(attn, device=attn.device, dtype=attn.dtype)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += attn_mask
            attn += attn_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x
