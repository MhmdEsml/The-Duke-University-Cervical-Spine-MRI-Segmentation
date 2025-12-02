import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def compute_metrics_per_slice(predictions: np.ndarray,
                             ground_truth: np.ndarray,
                             num_classes: int) -> dict:
    depth = predictions.shape[2]
    metrics = {
        'dice_per_slice': [],
        'iou_per_slice': [],
        'slice_indices': list(range(depth))
    }

    for slice_idx in range(depth):
        pred_slice = predictions[:, :, slice_idx].flatten()
        gt_slice = ground_truth[:, :, slice_idx].flatten()

        dice_scores = []
        for class_idx in range(1, num_classes):
            tp = np.sum((pred_slice == class_idx) & (gt_slice == class_idx))
            fp = np.sum((pred_slice == class_idx) & (gt_slice != class_idx))
            fn = np.sum((pred_slice != class_idx) & (gt_slice == class_idx))
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            dice_scores.append(dice)

        metrics['dice_per_slice'].append(np.mean(dice_scores) if dice_scores else 0)

        iou_scores = []
        for class_idx in range(1, num_classes):
            intersection = np.sum((pred_slice == class_idx) & (gt_slice == class_idx))
            union = np.sum((pred_slice == class_idx) | (gt_slice == class_idx))
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou)

        metrics['iou_per_slice'].append(np.mean(iou_scores) if iou_scores else 0)

    return metrics


def plot_metrics_over_slices(metrics: dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        metrics['slice_indices'],
        metrics['dice_per_slice'],
        marker='o',
        linewidth=2,
        markersize=4
    )
    axes[0].set_xlabel('Slice Index')
    axes[0].set_ylabel('Dice Coefficient')
    axes[0].set_title('Dice Coefficient per Slice')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        metrics['slice_indices'],
        metrics['iou_per_slice'],
        marker='s',
        linewidth=2,
        markersize=4,
        color='orange'
    )
    axes[1].set_xlabel('Slice Index')
    axes[1].set_ylabel('IoU Score')
    axes[1].set_title('IoU Score per Slice')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          class_names: List[str],
                          save_path: Optional[str] = None):
    cm = confusion_matrix(ground_truth.flatten(), predictions.flatten())

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_gif_from_slices(volume: np.ndarray,
                           segmentation: np.ndarray,
                           predictions: np.ndarray,
                           output_path: str,
                           class_names: List[str],
                           fps: int = 5):
    try:
        import imageio
        from visualization.visualizer import SegmentationVisualizer

        visualizer = SegmentationVisualizer(class_names)
        images = []

        depth = volume.shape[2]
        for slice_idx in range(depth):
            fig = plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(volume[:, :, slice_idx], cmap='gray')
            plt.title(f'Slice {slice_idx}')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(
                segmentation[:, :, slice_idx],
                cmap=visualizer.seg_cmap,
                vmin=0,
                vmax=len(class_names) - 1
            )
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(
                predictions[:, :, slice_idx],
                cmap=visualizer.seg_cmap,
                vmin=0,
                vmax=len(class_names) - 1
            )
            plt.title('Prediction')
            plt.axis('off')

            plt.tight_layout()

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            images.append(image)

            plt.close(fig)

        imageio.mimsave(output_path, images, fps=fps)

    except ImportError:
        pass
