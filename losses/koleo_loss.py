# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# Reference: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # pairwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward_2d_tensor(self, x, eps=1e-8):
        """
        Args:
            :param x: backbone output of student  (BxD)
            :param eps: Epsilon value to avoid division by zero
        """
        with torch.amp.autocast('cuda', enabled=False):
            x = F.normalize(x, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(x)  # noqa: E741
            distances = self.pdist(x, x[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss

    def forward_3d_tensor(self, x, eps=1e-8):
        """
        Args:
        :param x: backbone output of student  (BxNxD)
        :param eps: Epsilon value to avoid division by zero
        """
        with torch.amp.autocast('cuda', enabled=False):
            x_unbind = torch.unbind(x, dim=1)
            loss = torch.tensor(0.0, device=x.device)
            for x_ in x_unbind:
                loss = loss + self.forward_2d_tensor(x_, eps=eps)
        return loss/len(x_unbind)

    def forward(self, x, eps=1e-8):
        """
        Args:
        :param x: backbone output of student  (BxNxD)
        :param eps: Epsilon value to avoid division by zero
        """
        if x.dim() == 3:
            return self.forward_3d_tensor(x, eps=eps)
        elif x.dim() == 2:
            return self.forward_2d_tensor(x, eps=eps)
        else:
            raise ValueError("Input tensor should be 2D or 3D tensor")
