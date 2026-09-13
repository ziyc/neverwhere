import os
from PIL import Image
import pillow_heif

def convert_any_format_to_png(img_path, target_path):
    """
    Convert any format image to png format
    """
    if img_path.lower().endswith(('.heif', '.heic')):
        heif_file = pillow_heif.read_heif(img_path)
        img = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw"
        )
    else:
        img = Image.open(img_path)
    
    img.save(target_path)

def downsample_image(img_path, downsample_factor=0.5):
    """
    Downsample an image by a given factor (factor should be less than 1)
    """
    if downsample_factor >= 1:
        raise ValueError("Downsample factor must be less than 1")
        
    img = Image.open(img_path)
    width, height = img.size
    new_size = (int(width * downsample_factor), int(height * downsample_factor))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    img.save(img_path)

def batch_convert_any_format_to_png(source_dir, target_dir):
    """
    Batch convert any format image to png format
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for img_name in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_name)
        target_path = os.path.join(target_dir, os.path.splitext(img_name)[0] + '.png')
        try:
            convert_any_format_to_png(img_path, target_path)
        except Exception as e:
            print(f"An error occurred while converting '{img_path}': {e}")

if __name__ == "__main__":
    source_dir = input("Please enter the source directory: ")
    target_dir = input("Please enter the target directory: ")
    batch_convert_any_format_to_png(source_dir, target_dir)