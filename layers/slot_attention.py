""" Based on SPOT, SLATE and BOQSA libraries:
https://github.com/singhgautam/slate/blob/master/slot_attn.py
https://github.com/YuLiu-LY/BO-QSA/blob/main/models/slot_attn.py
https://github.com/gkakogeorgiou/spot/blob/master/slot_attn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from utils.spot_utils import linear, gru_cell


class SlotAttention(nn.Module):
    def __init__(
            self,
            num_iter,
            input_size,
            slot_size,
            mlp_size,
            truncate,
            heads,
            epsilon=1e-8,
            drop_path=0,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.input_size = input_size
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.GELU(),
            linear(mlp_size, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        assert self.truncate in ['bi-level', 'fixed-point', 'none']

    def forward(self, inputs, slots_init):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].
        slots = slots_init
        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1,
                                                                               2)  # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1,
                                                                               2)  # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k

        # Multiple rounds of attention.
        for i in range(self.num_iter):
            if i == self.num_iter - 1:
                if self.truncate == 'bi-level':
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1,
                                                                                 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))  # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
                , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1,
                                                                       2)  # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)  # Shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # Shape: [batch_size, num_heads, num_inputs, num_slots].
            updates = torch.matmul(attn.transpose(-1, -2),
                                   v)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)  # Shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, N_q, self.slot_size)
            slots = slots + self.drop_path(self.mlp(self.norm_mlp(slots)))

        return slots, attn_vis, attn_logits



