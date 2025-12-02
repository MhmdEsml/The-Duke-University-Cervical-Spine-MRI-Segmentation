import enum

class ActivationType(enum.Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SELU = "selu"
    GELU = "gelu"
    SILU = "silu"
    MISH = "mish"

class NormalizationType(enum.Enum):
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"
    LAYER_NORM = "layer_norm"
    NONE = "none"

class AttentionType(enum.Enum):
    NONE = "none"
    SE = "squeeze_excitation"
    CBAM = "cbam"

TRAINING_CONFIG = {
    "data_directory": "CSpineSeg/",
    "target_volume_shape": (256, 256, 16),
    "input_channels": 1,
    "output_classes": 3,
    "batch_size": 4,
    "accumulation_steps": 1,
    "max_epochs": 55,
    "model_checkpoint_path": "3d_unet_spine_segmentation.pth",
    "metrics_save_path": "training_metrics.json",
    "validation_split": 0.2,
    "random_seed": 42,
    "early_stopping_patience": 15,
    "learning_rate_patience": 10,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,
    "min_lr": 5e-5,
    "max_lr": 1e-4,
    "max_norm": 1.0,
    "class_names": ["Background", "Vertebral bodies", "Intervertebral discs"],
    "num_workers": 2
}

MODEL_CONFIG = {
    "encoder_channels": [32, 64, 128, 256, 512],
    "decoder_channels": [512, 256, 128, 64],
    "bottleneck_channels": 1024,
    "activation": ActivationType.SILU,
    "normalization": NormalizationType.GROUP_NORM,
    "attention_mechanism": AttentionType.SE,
    "num_groups": 8,
    "dropout_rate": 0.1,
    "use_residual": True,
    "use_attention_bridge": True,
    "use_deep_supervision": False,
    "deep_supervision_weight": 0.3,
    "kernel_size": 3,
    "padding": 1,

}


