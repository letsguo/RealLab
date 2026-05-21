import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Callable


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def fcnn_factory(input_dim: int, hidden_dims: list[int],
                 activation: Callable, output_dim: int = 1,
                 squash_output: bool = False,
                 use_spectral_norm: bool = False,
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float32,
                 ) -> torch.nn.Sequential:
    activation = resolve_nn_activation(activation)
    layers = []

    def maybe_apply_spectral_norm(layer):
        return spectral_norm(layer) if use_spectral_norm else layer

    # Input layer
    layers.append(maybe_apply_spectral_norm(nn.Linear(input_dim, hidden_dims[0])))
    layers.append(activation)

    # Hidden layers
    for layer_index in range(len(hidden_dims)):
        if layer_index == len(hidden_dims) - 1:
            layers.append(maybe_apply_spectral_norm(nn.Linear(hidden_dims[layer_index], output_dim)))
        else:
            layers.append(maybe_apply_spectral_norm(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1])))
            layers.append(activation)

    # Optional sigmoid output
    if squash_output:
        layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)
    model.to(device=device, dtype=dtype)
    return model