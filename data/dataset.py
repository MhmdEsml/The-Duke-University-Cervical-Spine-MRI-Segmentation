import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import nibabel as nib
from scipy.ndimage import zoom, rotate

class SpineSegmentationDataset(Dataset):
    def __init__(self, file_records, target_shape, num_classes=3, training_mode=True):
        self.file_records = file_records
        self.target_shape = target_shape
        self.num_classes = num_classes
        self.training_mode = training_mode

    def __len__(self):
        return len(self.file_records)

    def __getitem__(self, index):
        record = self.file_records[index]
        image_path = record["image"]
        segmentation_path = record["segmentation"]

        image_volume = nib.load(image_path).get_fdata().astype(np.float32)
        segmentation_volume = nib.load(segmentation_path).get_fdata().astype(np.int64)

        processed_image = self._preprocess_image(image_volume)
        processed_segmentation = self._preprocess_segmentation(segmentation_volume)

        if self.training_mode:
            processed_image, processed_segmentation = self._apply_augmentations(
                processed_image, processed_segmentation
            )
            
        processed_image = np.expand_dims(processed_image, axis=0)

        image_tensor = torch.from_numpy(processed_image.copy()).float()
        segmentation_tensor = torch.from_numpy(processed_segmentation.copy()).long()

        return image_tensor, segmentation_tensor

    def _preprocess_image(self, image_volume):
        image_volume = image_volume.astype(np.float32)

        non_zero_mask = image_volume > 0
        if np.any(non_zero_mask):
            p_low, p_high = np.percentile(image_volume[non_zero_mask], [1, 99])
            image_volume = np.clip(image_volume, p_low, p_high)
            image_volume = (image_volume - p_low) / (p_high - p_low + 1e-8)
        else:
            image_volume = np.zeros_like(image_volume)
            
        resize_factors = [
            self.target_shape[0] / image_volume.shape[0],
            self.target_shape[1] / image_volume.shape[1],
            self.target_shape[2] / image_volume.shape[2]
        ]
        image_volume = zoom(image_volume, resize_factors, order=3, mode='nearest')

        image_volume = np.clip(image_volume, 0.0, 1.0)

        return image_volume

    def _preprocess_segmentation(self, segmentation_volume):
        resize_factors = [
            self.target_shape[0] / segmentation_volume.shape[0],
            self.target_shape[1] / segmentation_volume.shape[1],
            self.target_shape[2] / segmentation_volume.shape[2]
        ]
        segmentation_volume = zoom(segmentation_volume, resize_factors, order=0, mode='nearest')
        return segmentation_volume.astype(np.int64)

    def _apply_augmentations(self, image_volume, segmentation_volume):
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15) 
            image_volume = rotate(image_volume, angle, axes=(0, 1), reshape=False, 
                                order=3, mode='constant', cval=0.0)
            segmentation_volume = rotate(segmentation_volume, angle, axes=(0, 1), reshape=False, 
                                       order=0, mode='constant', cval=0)

        if np.random.random() > 0.5:
            image_volume = np.flip(image_volume, axis=0)
            segmentation_volume = np.flip(segmentation_volume, axis=0)
        if np.random.random() > 0.5:
            image_volume = np.flip(image_volume, axis=1)
            segmentation_volume = np.flip(segmentation_volume, axis=1)

        if np.random.random() > 0.3:
            brightness = np.random.uniform(0.7, 1.3)
            image_volume = image_volume * brightness

        if np.random.random() > 0.3:
            noise_std = np.random.uniform(0.0, 0.05)
            image_volume = image_volume + np.random.normal(0, noise_std, image_volume.shape)

        image_volume = np.clip(image_volume, 0.0, 1.0)

        return image_volume, segmentation_volume

def initialize_data_loaders(config):
    image_paths = sorted(glob.glob(os.path.join(config["data_directory"], "images", "*.nii.gz")))
    segmentation_paths = sorted(glob.glob(os.path.join(config["data_directory"], "masks", "*.nii.gz")))
    
    print(f"Discovered {len(image_paths)} medical images and {len(segmentation_paths)} segmentation masks.")
    
    dataset_records = [
        {"image": img_path, "segmentation": seg_path}
        for img_path, seg_path in zip(image_paths, segmentation_paths)
    ]
    
    train_dicts, val_dicts = train_test_split(
        dataset_records, 
        test_size=config["validation_split"], 
        random_state=config["random_seed"]
    )
    
    train_ds = SpineSegmentationDataset(
        train_dicts, config["target_volume_shape"], config["output_classes"], training_mode=True
    )
    val_ds = SpineSegmentationDataset(
        val_dicts, config["target_volume_shape"], config["output_classes"], training_mode=False
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config.get("num_workers", 2), 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=True
    )
    
    return train_loader, val_loader

def initialize_data_loaders(config):
    image_paths = sorted(glob.glob(os.path.join(config["data_directory"], "images", "*.nii.gz")))
    segmentation_paths = sorted(glob.glob(os.path.join(config["data_directory"], "masks", "*.nii.gz")))
    
    print(f"Discovered {len(image_paths)} medical images and {len(segmentation_paths)} segmentation masks.")
    
    dataset_records = [
        {"image": img_path, "segmentation": seg_path}
        for img_path, seg_path in zip(image_paths, segmentation_paths)
    ]
    
    train_dicts, val_dicts = train_test_split(
        dataset_records, 
        test_size=config["validation_split"], 
        random_state=config["random_seed"]
    )
    
    train_ds = SpineSegmentationDataset(
        train_dicts, config["target_volume_shape"], config["output_classes"], training_mode=True
    )
    val_ds = SpineSegmentationDataset(
        val_dicts, config["target_volume_shape"], config["output_classes"], training_mode=False
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config.get("num_workers", 2), 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=True
    )
    
    return train_loader, val_loader