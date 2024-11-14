# This file contains the implementation of the IndependentMLPs class
import torch


class IndependentMLPsVariableOutputPredictor(torch.nn.Module):
    """
    This class implements the MLP used for classification with the option to use an additional independent MLP layer
    """

    def __init__(self, part_dim, latent_dim, out_dims, bias=False, num_lin_layers=1, hidden_dim=3072, dropout=0.0):
        """

        :param part_dim: Number of parts
        :param latent_dim: Latent dimension
        :param out_dims: Output dimensions for each part
        :param bias: Whether to use bias
        :param num_lin_layers: Number of linear layers
        :param hidden_dim: Hidden dimension
        :param dropout: Dropout rate
        """

        super().__init__()

        self.bias = bias
        self.latent_dim = latent_dim
        self.out_dims = out_dims
        assert len(out_dims) == part_dim, "The number of output dimensions must match the number of parts"
        self.part_dim = part_dim

        layer_stack = torch.nn.ModuleList()
        for i in range(part_dim):
            layer_stack.append(torch.nn.Sequential())
            if num_lin_layers == 1:
                layer_stack[i].add_module(f"norm", torch.nn.LayerNorm(latent_dim))
                layer_stack[i].add_module(f"dropout", torch.nn.Dropout(dropout))
                layer_stack[i].add_module(f"fc", torch.nn.Linear(latent_dim, self.out_dims[i], bias=bias))
            elif num_lin_layers == 2:
                layer_stack[i].add_module(f"norm", torch.nn.LayerNorm(latent_dim))
                layer_stack[i].add_module(f"fc1", torch.nn.Linear(latent_dim, hidden_dim, bias=bias))
                layer_stack[i].add_module(f"act1", torch.nn.GELU())
                layer_stack[i].add_module(f"dropout1", torch.nn.Dropout(dropout))
                layer_stack[i].add_module(f"fc2", torch.nn.Linear(hidden_dim, self.out_dims[i], bias=True))
            else:
                raise ValueError(f"Not supported number of linear layers: {num_lin_layers}")
        self.layers = layer_stack
        self.reset_weights()

    def reset_weights(self):
        """ Initialize weights with a truncated normal distribution"""
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear):
                    # Initialize weights with a truncated normal distribution
                    torch.nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward_inference(self, x):
        """ Input X has the dimensions batch x latent_dim x part_dim """

        per_part_outputs = []
        for i, layer in enumerate(self.layers):
            in_ = x[:, i, :]  # Select part feature i
            out = layer(in_)  # Apply MLP to feature i
            per_part_outputs.append(out)
        per_part_outputs = torch.cat(per_part_outputs, dim=-1)

        return per_part_outputs

    def forward(self, x):
        return self.forward_inference(x)
