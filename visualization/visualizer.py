import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
import seaborn as sns
from typing import List, Tuple, Optional
import os
from datetime import datetime

class SegmentationVisualizer:
    def __init__(self, class_names: List[str], class_colors: Optional[List[str]] = None):
        self.class_names = class_names
        self.num_classes = len(class_names)

        if class_colors is None:
            self.class_colors = [
                '#000000',
                '#FF6B6B',
                '#4ECDC4',
                '#45B7D1',
                '#96CEB4',
                '#FFEAA7',
            ][:self.num_classes]
        else:
            self.class_colors = class_colors

        self.seg_cmap = mcolors.ListedColormap(self.class_colors)

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def visualize_slice_comparison(
        self,
        image_slice: np.ndarray,
        ground_truth_slice: np.ndarray,
        predicted_slice: np.ndarray,
        slice_idx: int,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (18, 6),
    ):
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title(f'Input Image\nSlice {slice_idx}')
        axes[0].axis('off')

        axes[1].imshow(
            ground_truth_slice,
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
        )
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(
            predicted_slice,
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
        )
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        overlay = self._create_overlay(image_slice, predicted_slice)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (Image + Prediction)')
        axes[3].axis('off')

        self._add_color_legend(axes[3])

        plt.suptitle(f'Slice {slice_idx} - Segmentation Results', fontsize=14, y=0.95)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_volume_slices(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        predictions: np.ndarray,
        slice_indices: List[int],
        save_dir: Optional[str] = None,
        show: bool = True,
    ):
        n_slices = len(slice_indices)
        fig, axes = plt.subplots(n_slices, 3, figsize=(15, 5 * n_slices))

        if n_slices == 1:
            axes = axes.reshape(1, -1)

        for idx, slice_idx in enumerate(slice_indices):
            img_slice = volume[:, :, slice_idx]
            gt_slice = segmentation[:, :, slice_idx]
            pred_slice = predictions[:, :, slice_idx]

            axes[idx, 0].imshow(img_slice, cmap='gray')
            axes[idx, 0].set_title(f'Slice {slice_idx}')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(
                gt_slice,
                cmap=self.seg_cmap,
                vmin=0,
                vmax=self.num_classes - 1,
            )
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(
                pred_slice,
                cmap=self.seg_cmap,
                vmin=0,
                vmax=self.num_classes - 1,
            )
            axes[idx, 2].set_title('Prediction')
            axes[idx, 2].axis('off')

        plt.suptitle('Volume Slice Comparison', fontsize=16, y=0.98)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f'volume_slices_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_3d_views(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        predictions: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        fig = plt.figure(figsize=(15, 10))

        axial_slice = volume.shape[2] // 2
        coronal_slice = volume.shape[0] // 2
        sagittal_slice = volume.shape[1] // 2

        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(volume[:, :, axial_slice], cmap='gray')
        ax1.set_title(f'Axial - Slice {axial_slice}\nInput')
        ax1.axis('off')

        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(
            segmentation[:, :, axial_slice],
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
        )
        ax2.set_title('Ground Truth')
        ax2.axis('off')

        ax3 = plt.subplot(3, 4, 3)
        ax3.imshow(
            predictions[:, :, axial_slice],
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
        )
        ax3.set_title('Prediction')
        ax3.axis('off')

        ax4 = plt.subplot(3, 4, 4)
        overlay = self._create_overlay(
            volume[:, :, axial_slice],
            predictions[:, :, axial_slice],
        )
        ax4.imshow(overlay)
        ax4.set_title('Overlay')
        ax4.axis('off')

        ax5 = plt.subplot(3, 4, 5)
        ax5.imshow(volume[coronal_slice, :, :].T, cmap='gray', aspect='auto')
        ax5.set_title(f'Coronal - Slice {coronal_slice}\nInput')
        ax5.axis('off')

        ax6 = plt.subplot(3, 4, 6)
        ax6.imshow(
            segmentation[coronal_slice, :, :].T,
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
            aspect='auto',
        )
        ax6.set_title('Ground Truth')
        ax6.axis('off')

        ax7 = plt.subplot(3, 4, 7)
        ax7.imshow(
            predictions[coronal_slice, :, :].T,
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
            aspect='auto',
        )
        ax7.set_title('Prediction')
        ax7.axis('off')

        ax8 = plt.subplot(3, 4, 8)
        overlay_coronal = self._create_overlay(
            volume[coronal_slice, :, :].T,
            predictions[coronal_slice, :, :].T,
        )
        ax8.imshow(overlay_coronal)
        ax8.set_title('Overlay')
        ax8.axis('off')

        ax9 = plt.subplot(3, 4, 9)
        ax9.imshow(volume[:, sagittal_slice, :].T, cmap='gray', aspect='auto')
        ax9.set_title(f'Sagittal - Slice {sagittal_slice}\nInput')
        ax9.axis('off')

        ax10 = plt.subplot(3, 4, 10)
        ax10.imshow(
            segmentation[:, sagittal_slice, :].T,
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
            aspect='auto',
        )
        ax10.set_title('Ground Truth')
        ax10.axis('off')

        ax11 = plt.subplot(3, 4, 11)
        ax11.imshow(
            predictions[:, sagittal_slice, :].T,
            cmap=self.seg_cmap,
            vmin=0,
            vmax=self.num_classes - 1,
            aspect='auto',
        )
        ax11.set_title('Prediction')
        ax11.axis('off')

        ax12 = plt.subplot(3, 4, 12)
        overlay_sagittal = self._create_overlay(
            volume[:, sagittal_slice, :].T,
            predictions[:, sagittal_slice, :].T,
        )
        ax12.imshow(overlay_sagittal)
        ax12.set_title('Overlay')
        ax12.axis('off')

        plt.suptitle('3D Orthogonal Views', fontsize=16, y=0.95)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_batch_predictions(
        self,
        batch_images: torch.Tensor,
        batch_masks: torch.Tensor,
        batch_predictions: torch.Tensor,
        model: torch.nn.Module,
        device: torch.device,
        n_samples: int = 4,
        save_dir: Optional[str] = None,
        show: bool = True,
    ):
        model.eval()
        with torch.no_grad():
            if batch_predictions is None:
                batch_predictions = model(batch_images.to(device))
                if isinstance(batch_predictions, tuple):
                    batch_predictions = batch_predictions[0]

            images_np = batch_images.cpu().numpy()
            masks_np = batch_masks.cpu().numpy()

            if batch_predictions.dim() == 5:
                preds_np = torch.argmax(batch_predictions, dim=1).cpu().numpy()
            else:
                preds_np = batch_predictions.cpu().numpy()

        n_samples = min(n_samples, len(images_np))
        for sample_idx in range(n_samples):
            middle_slice = images_np[sample_idx, 0].shape[2] // 2

            img_slice = images_np[sample_idx, 0, :, :, middle_slice]
            gt_slice = masks_np[sample_idx, :, :, middle_slice]
            pred_slice = preds_np[sample_idx, :, :, middle_slice]

            self.visualize_slice_comparison(
                img_slice,
                gt_slice,
                pred_slice,
                slice_idx=middle_slice,
                save_path=(
                    os.path.join(
                        save_dir,
                        f'sample_{sample_idx}_slice_{middle_slice}.png',
                    )
                    if save_dir
                    else None
                ),
                show=show,
            )

    def _create_overlay(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        alpha: float = 0.5,
    ):
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        rgb_image = np.stack([image_norm] * 3, axis=-1)

        seg_colored = np.zeros((*segmentation.shape, 3))
        for class_idx in range(self.num_classes):
            mask = segmentation == class_idx
            color = mcolors.to_rgb(self.class_colors[class_idx])
            for c in range(3):
                seg_colored[..., c][mask] = color[c]

        overlay = rgb_image * (1 - alpha) + seg_colored * alpha
        return np.clip(overlay, 0, 1)

    def _add_color_legend(self, ax):
        legend_elements = []
        for idx, (class_name, color) in enumerate(
            zip(self.class_names, self.class_colors)
        ):
            if idx == 0 and class_name == "Background":
                continue
            patch = patches.Patch(
                color=color, label=f'{class_name} (Class {idx})'
            )
            legend_elements.append(patch)

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc='upper right',
                bbox_to_anchor=(1.3, 1),
                fontsize=8,
            )
