import os
import argparse
import shutil
from pathlib import Path

def sample_and_rename_images(source_dir, dest_dir, downsample, downsample_threshold):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all image files from the source directory
    image_files = sorted(os.listdir(source_dir))
    
    # Determine whether to downsample based on image count and threshold
    total_images = len(image_files)
    actual_downsample = downsample if total_images >= downsample_threshold else 1
    
    if actual_downsample == 1:
        print(f"Number of images ({total_images}) is below threshold ({downsample_threshold}). Skipping downsampling.")
    
    # Sample images at the specified interval
    sampled_images = image_files[::actual_downsample]
    
    # Copy and rename sampled images
    for i, img in enumerate(sampled_images):
        src_path = os.path.join(source_dir, img)
        dest_path = os.path.join(dest_dir, f"{i:04d}{os.path.splitext(img)[1]}")
        shutil.copy2(src_path, dest_path)
    
    print(f"Sampled and renamed {len(sampled_images)} images to {dest_dir}")

def main(input_dir, downsample, downsample_threshold=200):
    """
    Returns:
        wether to delete the cache folder
    """
    scene_dir = Path(input_dir)
    raw_images_dir = scene_dir / "images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find source directory first
    source_dir = None
    
    # Check polycam directory
    polycam_dir = scene_dir / "polycam"
    if polycam_dir.is_dir():
        images_dir = polycam_dir / "keyframes" / "images"
        if images_dir.is_dir():
            source_dir = images_dir
    
    # Check raw_data directory
    raw_data_dir = scene_dir / "raw_data" / "images"
    if raw_data_dir.is_dir():
        source_dir = raw_data_dir
        
    if source_dir is None:
        print(f"Error: No valid source directory found in {scene_dir}")
        return True

    # Calculate expected number of images after downsampling
    source_images = sorted(os.listdir(source_dir))
    total_images = len(source_images)
    actual_downsample = downsample if total_images >= downsample_threshold else 1
    expected_count = len(source_images[::actual_downsample])

    # Check if existing images match the expected count
    if raw_images_dir.is_dir():
        existing_images = os.listdir(raw_images_dir)
        if len(existing_images) != expected_count:
            print(f"Existing images count ({len(existing_images)}) doesn't match expected count ({expected_count})")
            print("Removing existing images and recreating...")
            for img in existing_images:
                (raw_images_dir / img).unlink()
        else:
            print("\n=== Correct number of downsampled images already exist, skipping downsampling step ===")
            return False

    sample_and_rename_images(source_dir, raw_images_dir, downsample, downsample_threshold)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample and rename images from Polycam scene directory.")
    parser.add_argument("-i", "--input-dir", required=True, help="Path to the scene directory")
    parser.add_argument("-d", "--downsample", type=int, default=1, help="Downsampling factor (default: 1)")
    parser.add_argument("-t", "--downsample-threshold", type=int, default=200, 
                       help="Minimum number of images required for downsampling (default: 200)")
    args = parser.parse_args()
    
    main(
        input_dir=args.input_dir,
        downsample=args.downsample,
        downsample_threshold=args.downsample_threshold,
    )