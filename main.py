import warnings
import argparse
import sys
import os
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='3D Spine Segmentation Pipeline')
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    parser.add_argument('--download', action='store_true', help='Download data from MIDRC')
    parser.add_argument('--credentials', type=str, help='Path to credentials.json')
    parser.add_argument('--manifest', type=str, help='Path to manifest.json (optional)')
    parser.add_argument('--output-dir', type=str, default='CSpineSeg', help='Directory to save downloaded data')
    parser.add_argument('--num-parallel', type=int, default=8, help='Number of parallel downloads')
    
    parser.add_argument('--model-path', type=str, default='3d_unet_spine_segmentation.pth',
                       help='Path to model checkpoint for visualization')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    if args.download:
        print("Starting data download from MIDRC...")
        try:
            from download_data import download_midrc_data
        except ImportError:
            print("Error: download_data.py not found!")
            sys.exit(1)
        
        if not args.credentials:
            print("Error: --credentials argument is required for download")
            print("Usage: python main.py --download --credentials /path/to/credentials.json [--manifest /path/to/manifest.json]")
            sys.exit(1)
        
        download_midrc_data(
            credentials_path=args.credentials,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            num_parallel=args.num_parallel
        )
        
        if args.output_dir != 'CSpineSeg':
            print(f"\nNOTE: Data downloaded to {args.output_dir}")
            print("You need to update config/config.py to point to this directory:")
            print(f'Change "data_directory": "CSpineSeg/" to "data_directory": "{args.output_dir}/"')
    
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
    
    elif args.train or (not args.download and not args.visualize):
        print("C-Spine 3D Segmentation Training Pipeline")
        print("=" * 50)
        
        from config.config import TRAINING_CONFIG
        data_dir = TRAINING_CONFIG["data_directory"]
        
        if not os.path.exists(data_dir):
            print(f"\nWARNING: Data directory '{data_dir}' does not exist!")
            print("You need to download data first.")
            print("\nTo download data, run:")
            print("python main.py --download --credentials /path/to/credentials.json")
            print("\nOr if you have data in a different location, update config/config.py")
            sys.exit(1)
        
        import glob
        image_files = glob.glob(os.path.join(data_dir, "images", "*.nii.gz"))
        mask_files = glob.glob(os.path.join(data_dir, "masks", "*.nii.gz"))
        
        if not image_files:
            print(f"\nERROR: No NIfTI files found in {data_dir}/images/")
            print("Make sure your data is in the correct structure:")
            print(f"{data_dir}/images/*.nii.gz")
            print(f"{data_dir}/masks/*.nii.gz")
            print("\nOr you may need to download data first:")
            print("python main.py --download --credentials /path/to/credentials.json")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images and {len(mask_files)} masks")
        
        try:
            from training.trainer import execute_training_pipeline
            from config.config import TRAINING_CONFIG, MODEL_CONFIG
            model, tracker = execute_training_pipeline(TRAINING_CONFIG, MODEL_CONFIG)
        except ImportError as e:
            print(f"Error importing training modules: {e}")
            print("Make sure all required modules are installed and in the correct location.")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
