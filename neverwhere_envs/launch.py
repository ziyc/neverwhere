import os
import shutil
import argparse
from pathlib import Path

from neverwhere_envs.runners.sort_images import main as frame_downsample_main
from neverwhere_envs.runners.colmap_runner import main as colmap_main
from neverwhere_envs.runners.openmvs_runner import main as openmvs_main
from neverwhere_envs.runners.geometry_processor import main as geometry_main
from neverwhere_envs.runners.gsplat_runner import main as gsplat_main
from neverwhere_envs.runners.gsplat2d_runner import main as gsplat2d_main
from neverwhere_envs.runners.gsplat_processor import main as gsplat_export_main
from neverwhere_envs.runners.openmvs_extractor import main as openmvs_export_main

from neverwhere_envs.utils.gs_utils import DefaultStrategy
from neverwhere_envs.utils.format_handler import convert_any_format_to_png, downsample_image

def run_pipeline(args):
    dataset_path = Path(args.dataset_dir)
    
    if args.scene_name == 'all':
        scene_dirs = [d for d in sorted(dataset_path.iterdir()) if d.is_dir()]
        for scene_dir in scene_dirs:
            try:
                print(f"\n==================== Processing scene: {scene_dir.name} ====================")
                process_scene(scene_dir.name, dataset_path, args)
            except Exception as e:
                print(f"Error processing scene {scene_dir.name}: {e}")
    else:
        process_scene(args.scene_name, dataset_path, args)

def run_3dgs_training(scene_dir, gsplat_3d_dir, gpu_index):
    print("\n=== Training 3DGS ===")
    
    gsplat_main(
        data_dir=str(scene_dir),
        result_dir=str(gsplat_3d_dir),
        gpu_index=gpu_index,
        init_type="openmvs",
        normalize_world_space=False,
        random_bkgd=True,
        random_points_bg=True,
        antialiased=True,
        disable_viewer=True,
        pose_opt=True,
        needle_reg=0.05,
        depth_loss=True,
        depth_lambda=0.03,
        normal_loss=False,
        strategy=DefaultStrategy(
            verbose=True,
            absgrad=True,
            grow_grad2d=0.0005,
        ),
    )

def run_2dgs_training(scene_dir, gsplat_2d_dir, gpu_index):
    print("\n=== Training 2DGS ===")
    gsplat2d_main(
        data_dir=str(scene_dir),
        result_dir=str(gsplat_2d_dir),
        gpu_index=gpu_index,
        init_type="openmvs",
        random_bkgd=True,
        disable_viewer=True,
        pose_opt=True,
        normal_loss=True,
        dist_loss=True,
    )

def process_scene(scene_name: str, dataset_dir: Path, args):
    scene_dir = dataset_dir / scene_name
    colmap_path = scene_dir / "colmap"
    openmvs_dir = scene_dir / "openmvs"
    gsplat_3d_dir = scene_dir / "3dgs"
    gsplat_2d_dir = scene_dir / "2dgs"
    images_dir = scene_dir / "images"
    
    if not (images_dir.exists() or (scene_dir / "polycam" / "keyframes" / "images").exists() or (scene_dir / "raw_data" / "images").exists()):
        print(f"Error: No images directory found in {scene_dir}. Expected one of:")
        print(f"  - {images_dir}")
        print(f"  - {scene_dir / 'polycam' / 'keyframes' / 'images'}")
        print(f"  - {scene_dir / 'raw_data' / 'images'}")
        raise FileNotFoundError("No images directory found")
    print("\n=== Processing images ===")
    if not args.skip_frame_downsample:
        delete_cache = frame_downsample_main(
            input_dir=str(scene_dir), 
            downsample=args.frame_downsample,
            downsample_threshold=args.frame_downsample_threshold,
        )
    
        if delete_cache:
            for folder in ["colmap", "openmvs", "geometry", "geo2d", "3dgs", "2dgs"]:
                folder_path = scene_dir / folder
                if folder_path.exists():
                    shutil.rmtree(folder_path)
    
    for img_path in images_dir.glob("*"):
        if not img_path.suffix.lower() == '.png':
            png_path = str(img_path.with_suffix(".png"))
            convert_any_format_to_png(str(img_path), png_path)
            try:
                os.remove(str(img_path))  # Delete original file after conversion
            except:
                pass
            img_path = png_path
        
        if not args.skip_image_downscale and args.image_downscale < 1.0:
            downsample_image(str(img_path), args.image_downscale)
    print("\n=== Running COLMAP pipeline ===")
    colmap_success = colmap_main(
        img_dir=str(images_dir),
        output_dir=str(colmap_path),
        gpu_index=args.gpu_index,
        single_camera=not args.multi_camera
    )
            
    if not colmap_success:
        print("Scene processing stopped due to failure of both COLMAP matching methods")
        error_log = scene_dir / "error.log"
        with open(error_log, 'w') as f:
            f.write("COLMAP Processing Failed\n")
            f.write("Both sequential and exhaustive matching methods failed\n")
            
        for folder in ["colmap", "openmvs", "geometry", "geo2d", "3dgs", "2dgs"]:
            folder_path = scene_dir / folder
            if folder_path.exists():
                shutil.rmtree(folder_path)
        return
    
    # minor fix here in case file paths in colmap database point to the mvs images directory
    colmap_images_link = os.path.join(colmap_path, 'images')
    if not os.path.exists(colmap_images_link):
        os.symlink(os.path.join(colmap_path, 'mvs', 'images'), colmap_images_link)
    print("\n=== Running OpenMVS pipeline ===")
    openmvs_main(
        working_dir=str(openmvs_dir),
        colmap_dir=str(colmap_path),
        image_dir=str(images_dir),
        gpu_index=args.gpu_index,
        refine_mesh=args.refine_mesh
    )
    print("\n=== Processing depth maps ===")
    openmvs_export_main(str(scene_dir), verbose=True, keys=args.geo_keys)
    print("\n=== Processing geometry ===")
    geometry_main(
        str(scene_dir),
        num_samples=200000,
        simplify_factor=0.95
    )
    if not args.skip_gs:
        if args.gs_type == '2dgs+3dgs':
            run_3dgs_training(scene_dir, gsplat_3d_dir, args.gpu_index)
            run_2dgs_training(scene_dir, gsplat_2d_dir, args.gpu_index)
        elif args.gs_type == '3dgs':
            run_3dgs_training(scene_dir, gsplat_3d_dir, args.gpu_index)
        else:  # 2dgs
            run_2dgs_training(scene_dir, gsplat_2d_dir, args.gpu_index)
        if args.gs_type in ['3dgs', '2dgs+3dgs']:
            print("\n=== Exporting 3DGS model to PLY ===")
            
            model_ckpt = gsplat_3d_dir / "model.pt"
            if not model_ckpt.exists():
                ckpt_dir = gsplat_3d_dir / "ckpts"
                if ckpt_dir.exists():
                    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_29999_")])
                    if ckpt_files:
                        model_ckpt = ckpt_dir / ckpt_files[0]
            
            if model_ckpt.exists():
                gsplat_export_main(
                    ckpt_path=str(model_ckpt),
                    gsplat_dir=str(gsplat_3d_dir),
                    ply_color_mode="sh_coeffs"  # Use SH coefficients for better quality
                )
            else:
                print("Warning: No 3DGS model checkpoint found to export")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Neverwhere reconstruction pipeline")
    parser.add_argument("--dataset-dir", required=True, help="Path to the datasets directory")
    parser.add_argument("--scene-name", required=True, help="Name of the scene to process, or 'all' to process all scenes")
    parser.add_argument("--gpu-index", type=str, default='-1', help="GPU index to use for processing")
    parser.add_argument("--multi-camera", action="store_true", help="images are from multiple cameras")
    parser.add_argument("--geo-keys", nargs='+', default=['depth', 'confidence'],
                      choices=['depth', 'confidence', 'normal', 'views'],
                      help='Types of depth data to process (default: depth confidence)')
    parser.add_argument("--refine-mesh", action="store_true", help="Refine mesh using OpenMVS")
    parser.add_argument("--frame-downsample", type=int, default=2, help="Frame downsampling factor - keep every Nth frame (default: 2)")
    parser.add_argument("--frame-downsample-threshold", type=int, default=200, 
                       help="Minimum number of images required for frame downsampling (default: 200)")
    parser.add_argument("--image-downscale", type=float, default=1.0, help="Image resolution downscale factor (default: 1.0)")
    parser.add_argument("--gs-type", type=str, choices=['3dgs', '2dgs', '2dgs+3dgs'], default='3dgs',
                       help="Type of Gaussian Splatting to use (default: 3dgs)")
    parser.add_argument("--skip-frame-downsample", action="store_true", help="Skip frame downsampling")
    parser.add_argument("--skip-image-downscale", action="store_true", help="Skip image downscaling")
    parser.add_argument("--skip-gs", action="store_true", help="Skip Gaussian Splatting training")

    args = parser.parse_args()
    run_pipeline(args)
