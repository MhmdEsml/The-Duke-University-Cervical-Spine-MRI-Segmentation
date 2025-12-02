import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from config.config import ActivationType, NormalizationType, AttentionType
from models.base_components import ConvBlock3D
from models.attention_mechanisms import AttentionFactory
from utils.factories import NormalizationFactory

class ThreeDUNet(nn.Module):
    def __init__(self, 
                 input_channels: int = 1,
                 output_classes: int = 3,
                 encoder_channels: List[int] = None,
                 decoder_channels: List[int] = None,
                 bottleneck_channels: int = 1024,
                 activation_type: ActivationType = ActivationType.RELU,
                 normalization_type: NormalizationType = NormalizationType.BATCH_NORM,
                 attention_type: AttentionType = AttentionType.NONE,
                 num_groups: int = 8,
                 dropout_rate: float = 0.0,
                 use_residual: bool = False,
                 use_attention_bridge: bool = False,
                 use_deep_supervision: bool = False,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
            
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.use_deep_supervision = use_deep_supervision
        self.depth = len(encoder_channels)
        
        assert len(decoder_channels) == len(encoder_channels) - 1, \
            f"Decoder channels must be one less than encoder channels. Got {len(decoder_channels)} and {len(encoder_channels)}"

        self.encoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.pool_kernels = []

        self.encoder_blocks.append(
            ConvBlock3D(input_channels, encoder_channels[0], activation_type, 
                        normalization_type, attention_type, num_groups, dropout_rate, 
                        use_residual, kernel_size, padding)
        )
        
        for i in range(1, self.depth):
            self.encoder_blocks.append(
                ConvBlock3D(encoder_channels[i-1], encoder_channels[i], activation_type, 
                            normalization_type, attention_type, num_groups, dropout_rate, 
                            use_residual, kernel_size, padding)
            )
            pool_kernel = (2, 2, 2) if i < 3 else (2, 2, 1)
            self.pooling_layers.append(nn.MaxPool3d(pool_kernel))
            self.pool_kernels.append(pool_kernel)

        self.bottleneck = ConvBlock3D(
            encoder_channels[-1], bottleneck_channels,
            activation_type, normalization_type, attention_type,
            num_groups, dropout_rate, use_residual, kernel_size, padding
        )
        
        self.use_attention_bridge = use_attention_bridge
        if use_attention_bridge:
            self.attention_bridge = AttentionFactory.create_attention(attention_type, bottleneck_channels)
        
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        reversed_kernels = self.pool_kernels[::-1]
        
        for i in range(self.depth - 1):
            in_channels = bottleneck_channels if i == 0 else decoder_channels[i-1]
            k_size = reversed_kernels[i]
            
            self.upsample_layers.append(
                nn.ConvTranspose3d(in_channels, decoder_channels[i], 
                                   kernel_size=k_size, stride=k_size)
            )
            
            skip_channels = encoder_channels[-(i+2)]
            
            self.decoder_blocks.append(
                ConvBlock3D(decoder_channels[i] + skip_channels, decoder_channels[i],
                            activation_type, normalization_type, attention_type,
                            num_groups, dropout_rate, use_residual, kernel_size, padding)
            )

        self.output_convolution = nn.Conv3d(decoder_channels[-1], output_classes, kernel_size=1)
        
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(len(decoder_channels) - 1): 
                self.deep_supervision_heads.append(
                    nn.Conv3d(decoder_channels[i], output_classes, kernel_size=1)
                )

        self._initialize_weights()
    def _initialize_weights(self):
        """Initialize weights for all layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoder_outputs = []
        
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)
            if i < self.depth - 1:
                x = self.pooling_layers[i](x)
        
        x = self.bottleneck(x)
        
        if self.use_attention_bridge:
            x = self.attention_bridge(x)
        
        deep_supervision_outputs = []
        for i in range(self.depth - 1):
            x = self.upsample_layers[i](x)
            skip_connection = encoder_outputs[-(i+2)]
            
            if x.shape[2:] != skip_connection.shape[2:]:
                diff = [skip_connection.size(d) - x.size(d) for d in range(2, 5)]
                x = F.pad(x, [diff[2]//2, diff[2]-diff[2]//2,
                              diff[1]//2, diff[1]-diff[1]//2,
                              diff[0]//2, diff[0]-diff[0]//2])
            
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder_blocks[i](x)
            
            if self.use_deep_supervision and i < len(self.decoder_blocks) - 1:
                deep_out = self.deep_supervision_heads[i](x)
                deep_supervision_outputs.append(deep_out)
        
        main_output = self.output_convolution(x)
        
        if self.use_deep_supervision:
            return main_output, deep_supervision_outputs[::-1] 
        else:
            return main_output

def create_3d_unet(input_channels: int, output_classes: int, config: dict = None):
    if config is None:
        config = MODEL_CONFIG
    
    return ThreeDUNet(
        input_channels=input_channels,
        output_classes=output_classes,
        encoder_channels=config.get("encoder_channels", [32, 64, 128, 256, 512]),
        decoder_channels=config.get("decoder_channels", [512, 256, 128, 64]),
        bottleneck_channels=config.get("bottleneck_channels", 1024),
        activation_type=config.get("activation", ActivationType.SILU),
        normalization_type=config.get("normalization", NormalizationType.GROUP_NORM),
        attention_type=config.get("attention_mechanism", AttentionType.SE),
        num_groups=config.get("num_groups", 8),
        dropout_rate=config.get("dropout_rate", 0.1),
        use_residual=config.get("use_residual", True),
        use_attention_bridge=config.get("use_attention_bridge", True),
        use_deep_supervision=config.get("use_deep_supervision", False),
        kernel_size=config.get("kernel_size", 3),
        padding=config.get("padding", 1)
    )



