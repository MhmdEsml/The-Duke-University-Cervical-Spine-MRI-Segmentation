# 3D Cervical Spine MRI Segmentation with Enhanced U-Net

A comprehensive 3D medical image segmentation pipeline for cervical spine MRI using an enhanced 3D U-Net architecture.

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
The dataset is hosted on the MIDRC portal:
https://data.midrc.org/discovery/H6K0-A61V

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

## How to use
```bash
# Clone the repository
git clone https://github.com/MhmdEsml/The-Duke-University-Cervical-Spine-MRI-Segmentation
cd The-Duke-University-Cervical-Spine-MRI-Segmentation

# Install requirements
pip install -r requirements.txt

# Download data
python main.py --download --credentials /kaggle/input/cspineseg-dataset-download/credentials.json --manifest /kaggle/input/cspineseg-dataset-download/MIDRC_data_files_manifest.json

# Train the model
python main.py --train

# Visualize results
python main.py --visualize --model-path best_model.pth --num-samples 5
