import numpy as np
import torch
import torch.nn.functional as F

class SegmentationMetrics:
    def __init__(self, num_classes, class_names=None, smoothing_factor=1e-6):
        self.num_classes = num_classes
        self.smoothing_factor = smoothing_factor
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset_metrics()

    def reset_metrics(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0

    def update_metrics(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        batch_cm = np.bincount(
            self.num_classes * target_flat + pred_flat,
            minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += batch_cm
        self.total_samples += len(target_flat)

    def calculate_dice_scores(self):
        dice_scores = {}
        for class_idx in range(self.num_classes):
            true_positive = self.confusion_matrix[class_idx, class_idx]
            false_positive = np.sum(self.confusion_matrix[:, class_idx]) - true_positive
            false_negative = np.sum(self.confusion_matrix[class_idx, :]) - true_positive
            
            dice_numerator = 2 * true_positive + self.smoothing_factor
            dice_denominator = 2 * true_positive + false_positive + false_negative + self.smoothing_factor
            
            dice_scores[self.class_names[class_idx]] = dice_numerator / dice_denominator
        
        dice_scores["Mean_Dice"] = np.mean(list(dice_scores.values()))
        return dice_scores

    def calculate_iou_scores(self):
        iou_scores = {}
        for class_idx in range(self.num_classes):
            true_positive = self.confusion_matrix[class_idx, class_idx]
            false_positive = np.sum(self.confusion_matrix[:, class_idx]) - true_positive
            false_negative = np.sum(self.confusion_matrix[class_idx, :]) - true_positive
            
            iou_numerator = true_positive + self.smoothing_factor
            iou_denominator = true_positive + false_positive + false_negative + self.smoothing_factor
            
            iou_scores[self.class_names[class_idx]] = iou_numerator / iou_denominator
        
        iou_scores["Mean_IoU"] = np.mean(list(iou_scores.values()))
        return iou_scores

    def calculate_precision_recall(self):
        precision_scores = {}
        recall_scores = {}
        
        for class_idx in range(self.num_classes):
            true_positive = self.confusion_matrix[class_idx, class_idx]
            false_positive = np.sum(self.confusion_matrix[:, class_idx]) - true_positive
            false_negative = np.sum(self.confusion_matrix[class_idx, :]) - true_positive
            
            precision_numerator = true_positive + self.smoothing_factor
            precision_denominator = true_positive + false_positive + self.smoothing_factor
            precision_scores[self.class_names[class_idx]] = precision_numerator / precision_denominator
            
            recall_numerator = true_positive + self.smoothing_factor
            recall_denominator = true_positive + false_negative + self.smoothing_factor
            recall_scores[self.class_names[class_idx]] = recall_numerator / recall_denominator
        
        precision_scores["Mean_Precision"] = np.mean(list(precision_scores.values()))
        recall_scores["Mean_Recall"] = np.mean(list(recall_scores.values()))
        
        return precision_scores, recall_scores

    def calculate_accuracy(self):
        overall_accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
        
        class_accuracy = {}
        for class_idx in range(self.num_classes):
            true_positive = self.confusion_matrix[class_idx, class_idx]
            class_total = np.sum(self.confusion_matrix[class_idx, :])
            class_accuracy[self.class_names[class_idx]] = (
                true_positive / class_total if class_total > 0 else 0.0
            )
        
        return overall_accuracy, class_accuracy

    def get_comprehensive_metrics(self):
        dice_scores = self.calculate_dice_scores()
        iou_scores = self.calculate_iou_scores()
        precision_scores, recall_scores = self.calculate_precision_recall()
        overall_accuracy, class_accuracy = self.calculate_accuracy()
        
        return {
            "Dice_Coef": dice_scores,
            "IoU": iou_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "Overall_Accuracy": overall_accuracy,
            "Class_Accuracy": class_accuracy,
            "Confusion_Matrix": self.confusion_matrix.tolist()
        }