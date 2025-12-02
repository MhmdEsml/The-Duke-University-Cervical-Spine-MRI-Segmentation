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
- **MIDRC Data Download** utility for easy data acquisition

## Download Data
------------------------------------------------------------------------

### Required Files: `credentials.json` and `MIDRC_data_files_manifest.json`

To download MIDRC data using the Gen3 client, you must obtain a
**`credentials.json`** file from the MIDRC data portal:

1. Go to https://data.midrc.org  
2. Log in using NIH / Google / ORCID / Institutional login  
3. Open: **Profile → API Keys**  
4. Click **“Create API Key”**  
5. Your personal `credentials.json` file will download automatically

Upload this file to Colab to authenticate the Gen3 client.

You do **not** need to worry about `MIDRC_data_files_manifest.json`;  
it is already included in the repository.

link to source data: https://data.midrc.org/discovery/H6K0-A61V


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CSpineSegmentation.git
cd CSpineSegmentation
