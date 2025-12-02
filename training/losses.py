import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedDiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, smoothing_factor=1.0, dice_weight=0.5, ce_weight=0.5):
        super(CombinedDiceCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing_factor = smoothing_factor
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def dice_loss(self, predicted_probabilities, ground_truth_one_hot):
        intersection = torch.sum(predicted_probabilities * ground_truth_one_hot, dim=(2, 3, 4))
        sum_predictions = torch.sum(predicted_probabilities, dim=(2, 3, 4))
        sum_ground_truth = torch.sum(ground_truth_one_hot, dim=(2, 3, 4))
        
        dice_coefficient = (2. * intersection + self.smoothing_factor) / (
            sum_predictions + sum_ground_truth + self.smoothing_factor
        )
        
        return 1.0 - torch.mean(dice_coefficient)

    def forward(self, network_output, ground_truth):
        ce_loss = self.cross_entropy_loss(network_output, ground_truth)
        
        pred_probs = F.softmax(network_output, dim=1)
        one_hot_ground_truth = F.one_hot(ground_truth, num_classes=self.num_classes)
        one_hot_ground_truth = one_hot_ground_truth.permute(0, 4, 1, 2, 3).float()
        
        dice_loss = self.dice_loss(pred_probs, one_hot_ground_truth)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        probs_flat = probs.contiguous().view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.contiguous().view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        tp = torch.sum(probs_flat * targets_flat, dim=2)
        fp = torch.sum(probs_flat * (1 - targets_flat), dim=2)
        fn = torch.sum((1 - probs_flat) * targets_flat, dim=2)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        
        return torch.mean(focal_tversky)
