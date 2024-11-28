"""
Patch Pruning for Vision Transformers
Inspired by PatchDropout: https://github.com/huggingface/pytorch-image-models/blob/730b907b4d45a4713cbc425cbf224c46089fd514/timm/layers/patch_dropout.py
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class PatchPruning(nn.Module):
    def __init__(
            self,
            keep_ratio: float = 1.0,
            num_prefix_tokens: int = 1,
            ordered: bool = True,
            return_indices: bool = True,
    ):
        super().__init__()
        self.prob = keep_ratio
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, mask: torch.Tensor, num_keep=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:

        if num_keep is None and self.prob == 1.0:
            return x

        # Prune out patches containing background
        # First, split the tokens into the class token and the patch tokens
        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        num_patches = x.shape[1]
        if num_keep is None:
            num_keep = max(1, int(num_patches * self.prob))
        else:
            num_keep = max(1, int(num_keep))

        # Give priority to patches with non bg values
        keep_indices = torch.argsort(mask, dim=-1, descending=True)[:, :num_keep]

        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]

        # Prune out patches
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        # Concatenate back prefix/class token
        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x


