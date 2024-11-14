""" Based on SPOT library:
https://github.com/gkakogeorgiou/spot/blob/master/slot_attn.py
"""
import torch
import torch.nn as nn
from timm.layers import DropPath
from utils.spot_utils import linear
from .slot_attention import SlotAttention


class SlotAttentionEncoder(nn.Module):

    def __init__(self, num_iterations, num_slots,
                 input_channels, slot_size, mlp_hidden_size, pos_channels, truncate='bi-level', init_method='embedding',
                 num_heads=1, drop_path=0.0):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels
        self.init_method = init_method
        self.drop_path_enc = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.GELU(),
            linear(input_channels, input_channels))

        assert init_method in ['shared_gaussian', 'embedding']
        if init_method == 'shared_gaussian':
            # Parameters for Gaussian init (shared by all slots).
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            self.slots_init = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        else:
            raise NotImplementedError

        self.slot_attention = SlotAttention(
            num_iterations,
            input_channels, slot_size, mlp_hidden_size, truncate, num_heads, drop_path=drop_path)

    def forward(self, x):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        B, *_ = x.size()
        dtype = x.dtype
        device = x.device
        x = x + self.drop_path_enc(self.mlp(self.layer_norm(x)))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

        # Slot Attention module.
        init_slots = self.slots_initialization(B, dtype, device)

        slots, attn, attn_logits = self.slot_attention(x, init_slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].

        return slots, attn, init_slots, attn_logits

    def slots_initialization(self, B, dtype, device):
        # The first frame, initialize slots.
        if self.init_method == 'shared_gaussian':
            slots_init = torch.empty((B, self.num_slots, self.slot_size), dtype=dtype, device=device).normal_()
            slots_init = self.slot_mu + torch.exp(self.slot_log_sigma) * slots_init
        elif self.init_method == 'embedding':
            slots_init = self.slots_init.weight.expand(B, -1, -1).contiguous()

        return slots_init