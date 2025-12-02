import warnings
import argparse
warnings.filterwarnings("ignore")

from training.trainer import execute_training_pipeline
from config.config import TRAINING_CONFIG, MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description='3D Spine Segmentation Pipeline')
    parser.add_argument('--download', action='store_true', help='Download data from MIDRC')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--model-path', type=str, default='3d_unet_spine_segmentation.pth',
                       help='Path to model checkpoint for visualization')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    if args.download:
        print("Starting data download...")
        from download_data import download_midrc_data
        download_midrc_data()
    
    elif args.visualize:
        print("Starting visualization...")
        from visualize_results import visualize_validation_results, visualize_training_history
        visualize_validation_results(
            model_path=args.model_path,
            num_samples=args.num_samples
        )
        visualize_training_history()
    
    elif args.train:
        print("C-Spine 3D Segmentation Training Pipeline")
        print("=" * 50)
        model, tracker = execute_training_pipeline(TRAINING_CONFIG, MODEL_CONFIG)
    
    else:
        # Default: train the model
        print("C-Spine 3D Segmentation Training Pipeline")
        print("=" * 50)
        model, tracker = execute_training_pipeline(TRAINING_CONFIG, MODEL_CONFIG)

if __name__ == "__main__":
    main()
