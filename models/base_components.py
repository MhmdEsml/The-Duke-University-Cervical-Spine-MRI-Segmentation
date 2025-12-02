import torch.nn as nn
from config.config import ActivationType, NormalizationType, AttentionType
from utils.factories import ActivationFactory, NormalizationFactory
from models.attention_mechanisms import AttentionFactory

class ConvBlock3D(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 activation_type: ActivationType = ActivationType.RELU,
                 normalization_type: NormalizationType = NormalizationType.BATCH_NORM,
                 attention_type: AttentionType = AttentionType.NONE,
                 num_groups: int = 8,
                 dropout_rate: float = 0.0,
                 use_residual: bool = False,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        
        self.use_residual = use_residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        layers = []
        layers.extend([
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            NormalizationFactory.create_norm(normalization_type, out_channels, num_groups),
            ActivationFactory.create_activation(activation_type)
        ])
        
        layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            NormalizationFactory.create_norm(normalization_type, out_channels, num_groups),
        ])
        
        layers.append(ActivationFactory.create_activation(activation_type))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout3d(dropout_rate))
            
        layers.append(AttentionFactory.create_attention(attention_type, out_channels))
        
        self.conv_layers = nn.Sequential(*layers)
        
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                NormalizationFactory.create_norm(normalization_type, out_channels, num_groups)
            )
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            residual = self.residual_conv(x)
            return residual + self.conv_layers(x)
        else:
            return self.conv_layers(x)