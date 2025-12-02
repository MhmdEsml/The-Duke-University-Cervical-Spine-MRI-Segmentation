import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import AttentionType

class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1, 1)
        return x * y.expand_as(x)

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = SqueezeExcitation3D(channels, reduction_ratio)
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.channel_attention(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        return x * spatial_weights

class AttentionFactory:
    @staticmethod
    def create_attention(attention_type: AttentionType, channels: int):
        if attention_type == AttentionType.NONE:
            return nn.Identity()
        elif attention_type == AttentionType.SE:
            return SqueezeExcitation3D(channels)
        elif attention_type == AttentionType.CBAM:
            return CBAM3D(channels)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
