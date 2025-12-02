import numpy as np
import json
from datetime import datetime

class TrainingProgressTracker:
    def __init__(self, class_names):
        self.class_names = class_names
        self.training_losses = []
        self.validation_losses = []
        self.training_metrics = []
        self.validation_metrics = []
        self.epoch_durations = []
        self.learning_rates = []
        
    def update_metrics(self, train_loss, val_loss, train_metrics, val_metrics, duration, lr=None):
        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)
        self.training_metrics.append(train_metrics)
        self.validation_metrics.append(val_metrics)
        self.epoch_durations.append(duration)
        if lr is not None:
            self.learning_rates.append(lr)
        
    def display_epoch_progress(self, epoch, total_epochs, train_loss, val_loss, val_metrics):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{total_epochs}")
        print(f"{'='*80}")
        print(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"\n--- VALIDATION METRICS ---")
        
        dice_metrics = val_metrics["Dice_Coef"]
        print(f"\nDice Coefficients:")
        for class_name in self.class_names:
            print(f"  {class_name}: {dice_metrics[class_name]:.4f}")
        print(f"  Mean Dice: {dice_metrics['Mean_Dice']:.4f}")
        
        iou_metrics = val_metrics["IoU"]
        print(f"\nIoU Scores:")
        for class_name in self.class_names:
            print(f"  {class_name}: {iou_metrics[class_name]:.4f}")
        print(f"  Mean IoU: {iou_metrics['Mean_IoU']:.4f}")
        
        precision_metrics = val_metrics["Precision"]
        recall_metrics = val_metrics["Recall"]
        print(f"\nPrecision & Recall:")
        for class_name in self.class_names:
            print(f"  {class_name}: Precision={precision_metrics[class_name]:.4f}, Recall={recall_metrics[class_name]:.4f}")
        
        print(f"  Overall Accuracy: {val_metrics['Overall_Accuracy']:.4f}")
        
    def display_training_summary(self):
        best_epoch = np.argmin(self.validation_losses)
        best_val_loss = self.validation_losses[best_epoch]
        best_dice = self.validation_metrics[best_epoch]["Dice_Coef"]["Mean_Dice"]
        
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Best Epoch: {best_epoch + 1}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Best Mean Dice: {best_dice:.4f}")
        print(f"Average Epoch Duration: {np.mean(self.epoch_durations):.2f}s")
        print(f"Total Training Time: {np.sum(self.epoch_durations):.2f}s")
        
        best_metrics = self.validation_metrics[best_epoch]
        print(f"\nBest Epoch Metrics:")
        for class_name in self.class_names:
            dice = best_metrics["Dice_Coef"][class_name]
            iou = best_metrics["IoU"][class_name]
            precision = best_metrics["Precision"][class_name]
            recall = best_metrics["Recall"][class_name]
            print(f"  {class_name}: Dice={dice:.4f}, IoU={iou:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    def save_metrics_to_file(self, file_path):
        metrics_data = {
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "epoch_durations": self.epoch_durations,
            "learning_rates": self.learning_rates,
            "class_names": self.class_names,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to: {file_path}")