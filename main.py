import warnings
warnings.filterwarnings("ignore")

from training.trainer import execute_training_pipeline
from config.config import TRAINING_CONFIG, MODEL_CONFIG

if __name__ == "__main__":
    print("C-Spine 3D Segmentation Training Pipeline")
    print("=" * 50)
    
    model, tracker = execute_training_pipeline(TRAINING_CONFIG, MODEL_CONFIG)