import torch.nn as nn
from config.config import ActivationType, NormalizationType

class ActivationFactory:
    @staticmethod
    def create_activation(activation_type: ActivationType, **kwargs):
        if activation_type == ActivationType.RELU:
            return nn.ReLU(inplace=True)
        elif activation_type == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(negative_slope=kwargs.get('negative_slope', 0.01), inplace=True)
        elif activation_type == ActivationType.SELU:
            return nn.SELU(inplace=True)
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        elif activation_type == ActivationType.SILU:
            return nn.SiLU(inplace=True)
        elif activation_type == ActivationType.MISH:
            return nn.Mish(inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

class NormalizationFactory:
    @staticmethod
    def create_norm(norm_type: NormalizationType, num_channels: int, num_groups: int = 8):
        if norm_type == NormalizationType.BATCH_NORM:
            return nn.BatchNorm3d(num_channels)
        elif norm_type == NormalizationType.GROUP_NORM:
            return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
        elif norm_type == NormalizationType.INSTANCE_NORM:
            return nn.InstanceNorm3d(num_channels)
        elif norm_type == NormalizationType.LAYER_NORM:
            return nn.GroupNorm(1, num_channels)  
        elif norm_type == NormalizationType.NONE:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
