# 3D Spine Segmentation with U-Net

A comprehensive 3D medical image segmentation pipeline for cervical spine CT scans using an enhanced 3D U-Net architecture.

## Features

- **Advanced 3D U-Net Architecture** with customizable encoder/decoder channels
- **Multiple Attention Mechanisms** (Squeeze-Excitation, CBAM)
- **Flexible Normalization** (BatchNorm, GroupNorm, InstanceNorm, LayerNorm)
- **Comprehensive Metrics** (Dice, IoU, Precision, Recall, Accuracy)
- **Mixed Precision Training** for faster computation
- **Gradient Accumulation** for larger effective batch sizes
- **Cosine Annealing with Warmup** for optimized learning rate scheduling
- **Early Stopping** and model checkpointing
- **MIDRC Data Download** utility for easy data acquisition

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CSpineSegmentation.git
cd CSpineSegmentation
