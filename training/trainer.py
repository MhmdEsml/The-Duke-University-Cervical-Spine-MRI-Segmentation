import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import time
import math
from tqdm import tqdm

from models.unet_3d import create_3d_unet
from data.dataset import initialize_data_loaders
from training.losses import CombinedDiceCrossEntropyLoss
from training.metrics import SegmentationMetrics
from training.progress_tracker import TrainingProgressTracker
from config.config import TRAINING_CONFIG, MODEL_CONFIG

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr, max_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1.0
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            warmup_progress = float(current_step) / float(max(1, num_warmup_steps))
            return min_lr + (max_lr - min_lr) * warmup_progress
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + (max_lr - min_lr) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

def execute_training_pipeline(training_config=None, model_config=None):
    if training_config is None:
        training_config = TRAINING_CONFIG
    if model_config is None:
        model_config = MODEL_CONFIG
        
    print("Initializing 3D U-Net Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Computational Device: {device}")
    
    train_loader, val_loader = initialize_data_loaders(training_config)
    
    class_names = training_config.get("class_names", ["Background", "Class_1", "Class_2"])
    
    segmentation_model = create_optimized_3d_unet(
        input_channels=training_config["input_channels"],
        output_classes=training_config["output_classes"],
        config=model_config
    )
    
    print(f"INFO: Model with {sum(p.numel() for p in segmentation_model.parameters()):,} parameters")
    
    if torch.cuda.device_count() > 1:
        print(f"INFO: Utilizing {torch.cuda.device_count()} GPUs for parallel processing")
        segmentation_model = nn.DataParallel(segmentation_model)
    
    segmentation_model.to(device)
    
    loss_function = CombinedDiceCrossEntropyLoss(
        num_classes=training_config["output_classes"],
        dice_weight=0.6,
        ce_weight=0.4
    )
    
    optimizer = torch.optim.AdamW(
        segmentation_model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    total_training_steps = len(train_loader) * training_config["max_epochs"]
    warmup_steps = len(train_loader) * training_config["warmup_epochs"]
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr=training_config["min_lr"],
        max_lr=training_config["max_lr"]
    )

    scaler = torch.cuda.amp.GradScaler()
    progress_tracker = TrainingProgressTracker(class_names)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    print("\nStarting Training Process...")
    for epoch in range(training_config["max_epochs"]):
        epoch_start_time = time.time()
        
        segmentation_model.train()
        epoch_train_loss = 0.0
        train_metrics = SegmentationMetrics(training_config["output_classes"], class_names)
        
        train_progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{training_config['max_epochs']} [Training]",
            leave=False
        )

        accumulation_steps = training_config.get("accumulation_steps", 1)
        optimizer.zero_grad()
        
        for batch_idx, (batch_images, batch_masks) in enumerate(train_progress_bar):
            batch_images = batch_images.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = segmentation_model(batch_images)
                
                if model_config["use_deep_supervision"]:
                    main_output, deep_outputs = outputs
                    main_loss = loss_function(main_output, batch_masks)
                    
                    target_size = batch_masks.shape[1:]
                    deep_loss = 0
                    for deep_out in deep_outputs:
                        deep_out_upsampled = F.interpolate(
                            deep_out, 
                            size=target_size, 
                            mode='trilinear', 
                            align_corners=False
                        )
                        deep_loss += loss_function(deep_out_upsampled, batch_masks)
                    
                    loss = main_loss + model_config["deep_supervision_weight"] * deep_loss
                else:
                    loss = loss_function(outputs, batch_masks)
                
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    segmentation_model.parameters(), 
                    max_norm=training_config["max_norm"]
                )
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                scheduler.step() 
            
            epoch_train_loss += loss.item() * accumulation_steps
            
            with torch.no_grad():
                if model_config["use_deep_supervision"]:
                    pred_classes = torch.argmax(main_output, dim=1)
                else:
                    pred_classes = torch.argmax(outputs, dim=1)
                train_metrics.update_metrics(pred_classes, batch_masks)

            current_accum = (batch_idx % accumulation_steps) + 1
            train_progress_bar.set_postfix({
                "Batch Loss": f"{loss.item() * accumulation_steps:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                "Accum": f"{current_accum}/{accumulation_steps}"
            })
        
        epoch_train_loss /= len(train_loader)
        train_epoch_metrics = train_metrics.get_comprehensive_metrics()
        
        segmentation_model.eval()
        epoch_val_loss = 0.0
        val_metrics = SegmentationMetrics(training_config["output_classes"], class_names)
        
        val_progress_bar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{training_config['max_epochs']} [Validation]",
            leave=False
        )
        
        with torch.no_grad():
            for batch_images, batch_masks in val_progress_bar:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_masks = batch_masks.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = segmentation_model(batch_images)
                    
                    if model_config["use_deep_supervision"]:
                        main_output, deep_outputs = outputs
                        loss = loss_function(main_output, batch_masks)
                        
                        target_size = batch_masks.shape[1:]
                        deep_loss = 0
                        for deep_out in deep_outputs:
                            deep_out_upsampled = F.interpolate(
                                deep_out, 
                                size=target_size, 
                                mode='trilinear', 
                                align_corners=False
                            )
                            deep_loss += loss_function(deep_out_upsampled, batch_masks)
                        
                        loss = loss + model_config["deep_supervision_weight"] * deep_loss
                    else:
                        loss = loss_function(outputs, batch_masks)
                
                epoch_val_loss += loss.item()
                
                if model_config["use_deep_supervision"]:
                    pred_classes = torch.argmax(main_output, dim=1)
                else:
                    pred_classes = torch.argmax(outputs, dim=1)
                    
                val_metrics.update_metrics(pred_classes, batch_masks)
                val_progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
        
        epoch_val_loss /= len(val_loader)
        val_epoch_metrics = val_metrics.get_comprehensive_metrics()
        epoch_duration = time.time() - epoch_start_time
        
        current_lr = scheduler.get_last_lr()[0]

        progress_tracker.update_metrics(
            epoch_train_loss, epoch_val_loss,
            train_epoch_metrics, val_epoch_metrics,
            epoch_duration, current_lr
        )

        progress_tracker.display_epoch_progress(
            epoch, training_config["max_epochs"], epoch_train_loss, 
            epoch_val_loss, val_epoch_metrics
        )
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stopping_counter = 0
            model_state = (
                segmentation_model.module.state_dict() 
                if isinstance(segmentation_model, nn.DataParallel) 
                else segmentation_model.state_dict()
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'training': training_config,
                    'model': model_config
                }
            }, training_config["model_checkpoint_path"])
            print(f"INFO: New optimal model saved to {training_config['model_checkpoint_path']}")
        else:
            early_stopping_counter += 1
            print(f"INFO: Early stopping counter: {early_stopping_counter}/{training_config['early_stopping_patience']}")
        
        if early_stopping_counter >= training_config["early_stopping_patience"]:
            print(f"INFO: Early stopping triggered after {epoch + 1} epochs")
            break

    progress_tracker.display_training_summary()
    progress_tracker.save_metrics_to_file(training_config["metrics_save_path"])
    print("INFO: Training pipeline completed.")
    

    return segmentation_model, progress_tracker
