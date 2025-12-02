import warnings
import argparse
import sys
import os
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='3D Spine Segmentation Pipeline')
    
    # Main commands
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    # Download options
    parser.add_argument('--download', action='store_true', help='Download data from MIDRC')
    parser.add_argument('--credentials', type=str, help='Path to credentials.json')
    parser.add_argument('--manifest', type=str, help='Path to manifest.json (optional)')
    parser.add_argument('--output-dir', type=str, default='CSpineSeg', help='Directory to save downloaded data')
    parser.add_argument('--num-parallel', type=int, default=8, help='Number of parallel downloads')
    
    # Visualization options
    parser.add_argument('--model-path', type=str, default='3d_unet_spine_segmentation.pth',
                       help='Path to model checkpoint for visualization')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Handle download command
    if args.download:
        print("Starting data download from MIDRC...")
        
        # Check if credentials is provided
        if not args.credentials:
            print("Error: --credentials argument is required for download")
            print("Usage: python main.py --download --credentials /path/to/credentials.json [--manifest /path/to/manifest.json]")
            sys.exit(1)
        
        # Run download directly
        from download_data import download_midrc_data
        
        download_midrc_data(
            credentials_path=args.credentials,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            num_parallel=args.num_parallel
        )
        
        # After download, update the config if output_dir is not default
        if args.output_dir != 'CSpineSeg':
            print(f"\nNOTE: Data downloaded to {args.output_dir}")
            print("You may need to update config/config.py to point to this directory")
    
    # Handle visualization command
    elif args.visualize:
        print("Starting visualization...")
        try:
            from visualize_results import visualize_validation_results, visualize_training_history
            visualize_validation_results(
                model_path=args.model_path,
                num_samples=args.num_samples
            )
            visualize_training_history()
        except ImportError as e:
            print(f"Error importing visualization modules: {e}")
            print("Make sure all visualization files are in place.")
            sys.exit(1)
    
    # Handle training command (default)
    elif args.train or (not args.download and not args.visualize):
        print("C-Spine 3D Segmentation Training Pipeline")
        print("=" * 50)
        
        try:
            from training.trainer import execute_training_pipeline
            from config.config import TRAINING_CONFIG, MODEL_CONFIG
            model, tracker = execute_training_pipeline(TRAINING_CONFIG, MODEL_CONFIG)
        except ImportError as e:
            print(f"Error importing training modules: {e}")
            print("Make sure all required modules are installed and in the correct location.")
            sys.exit(1)
    
    else:
        # Show help
        parser.print_help()

if __name__ == "__main__":
    main()
