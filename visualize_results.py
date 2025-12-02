import torch
import numpy as np
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config.config import TRAINING_CONFIG, MODEL_CONFIG
from data.dataset import SpineSegmentationDataset
from models.unet_3d import create_optimized_3d_unet
from visualization.visualizer import SegmentationVisualizer
from visualization.utils import compute_metrics_per_slice, plot_metrics_over_slices


def visualize_validation_results(
    model_path: str = None,
    config: dict = None,
    num_samples: int = 3,
    output_dir: str = "visualization_results",
):
    if config is None:
        config = TRAINING_CONFIG

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path is None:
        model_path = config["model_checkpoint_path"]

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = create_optimized_3d_unet(
        input_channels=config["input_channels"],
        output_classes=config["output_classes"],
        config=MODEL_CONFIG,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("Model loaded successfully")

    visualizer = SegmentationVisualizer(config["class_names"])

    image_paths = sorted(
        glob.glob(os.path.join(config["data_directory"], "images", "*.nii.gz"))
    )
    segmentation_paths = sorted(
        glob.glob(os.path.join(config["data_directory"], "masks", "*.nii.gz"))
    )

    dataset_records = [
        {"image": img_path, "segmentation": seg_path}
        for img_path, seg_path in zip(image_paths, segmentation_paths)
    ]

    val_records = dataset_records[-num_samples:]
    all_metrics = []

    for idx, record in enumerate(val_records):
        print(
            f"\nProcessing sample {idx+1}/{len(val_records)}: "
            f"{os.path.basename(record['image'])}"
        )

        sample_dataset = SpineSegmentationDataset(
            [record],
            config["target_volume_shape"],
            config["output_classes"],
            training_mode=False,
        )

        image_tensor, mask_tensor = sample_dataset[0]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred_tensor = torch.argmax(output, dim=1)

        image_np = image_tensor.cpu().numpy()[0, 0]
        mask_np = mask_tensor.cpu().numpy()
        pred_np = pred_tensor.cpu().numpy()[0]

        sample_name = os.path.basename(record["image"]).replace(".nii.gz", "")
        sample_dir = os.path.join(output_dir, f"sample_{idx}_{sample_name}")
        os.makedirs(sample_dir, exist_ok=True)

        depth = image_np.shape[2]
        slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

        for slice_idx in slice_indices:
            visualizer.visualize_slice_comparison(
                image_np[:, :, slice_idx],
                mask_np[:, :, slice_idx],
                pred_np[:, :, slice_idx],
                slice_idx=slice_idx,
                save_path=os.path.join(
                    sample_dir, f"slice_{slice_idx}.png"
                ),
                show=False,
            )

        visualizer.visualize_3d_views(
            image_np,
            mask_np,
            pred_np,
            save_path=os.path.join(sample_dir, "3d_views.png"),
            show=False,
        )

        metrics = compute_metrics_per_slice(
            pred_np, mask_np, config["output_classes"]
        )
        plot_metrics_over_slices(
            metrics,
            save_path=os.path.join(sample_dir, "metrics_per_slice.png"),
        )

        all_metrics.append(
            {
                "sample": sample_name,
                "mean_dice": np.mean(metrics["dice_per_slice"]),
                "mean_iou": np.mean(metrics["iou_per_slice"]),
            }
        )

        np.savez_compressed(
            os.path.join(sample_dir, "volumes.npz"),
            image=image_np,
            ground_truth=mask_np,
            prediction=pred_np,
        )

        print(f"  Saved visualizations to: {sample_dir}")

    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    for metrics in all_metrics:
        print(f"{metrics['sample']}:")
        print(f"  Mean Dice: {metrics['mean_dice']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")

    print(f"\nAll visualizations saved to: {output_dir}")


def visualize_training_history(
    metrics_path: str = None,
    output_dir: str = "training_visualization",
):
    import json
    import matplotlib.pyplot as plt

    if metrics_path is None:
        metrics_path = TRAINING_CONFIG["metrics_save_path"]

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_data["training_losses"], label="Training Loss", linewidth=2)
    plt.plot(
        metrics_data["validation_losses"], label="Validation Loss", linewidth=2
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "loss_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    epochs = range(len(metrics_data["validation_metrics"]))
    dice_scores = [
        m["Dice_Coef"]["Mean_Dice"]
        for m in metrics_data["validation_metrics"]
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dice_scores, marker="o", linewidth=2, markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice Coefficient")
    plt.title("Validation Dice Score over Training")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "dice_progress.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"Training history visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("3D Spine Segmentation Visualization")
    print("=" * 50)

    visualize_validation_results(
        model_path="3d_unet_spine_segmentation.pth",
        num_samples=3,
        output_dir="validation_visualizations",
    )

    visualize_training_history(
        metrics_path="training_metrics.json",
        output_dir="training_visualizations",
    )
