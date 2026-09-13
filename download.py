"""Download Neverwhere scenes (and the example policy checkpoints) from Hugging Face.

Each scene is one self-contained zip: the 3DGS model, collision geometry, COLMAP/OpenMVS reconstruction,
the original capture and the MuJoCo XMLs.

Examples:
    python download.py --scenes hurdle_stata_v1 ramp_bricks_v2
    python download.py --task stairs
    python download.py --all --dest /big/disk/scenes
    python download.py --checkpoints
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent
SCENES_FILE = REPO_ROOT / "neverwhere" / "scenes.txt"
DEFAULT_DEST = REPO_ROOT / "neverwhere" / "tasks" / "real_scenes"
DEFAULT_REPO = "ziyc/neverwhere"


def read_scenes():
    rows = []
    for line in open(SCENES_FILE):
        if line.strip() and not line.startswith("#"):
            scene, task, split, *_ = line.split()
            rows.append(dict(scene=scene, task=task, split=split))
    return rows


def link_scenes_root(dest: Path):
    """Make neverwhere/tasks/real_scenes point at dest so MuJoCo XMLs resolve scene meshes."""
    dest = dest.resolve()
    if DEFAULT_DEST.resolve() == dest:
        return
    if DEFAULT_DEST.is_symlink() or DEFAULT_DEST.exists():
        if DEFAULT_DEST.resolve() != dest:
            raise RuntimeError(f"{DEFAULT_DEST} already exists and does not point to {dest}; remove it first.")
        return
    DEFAULT_DEST.symlink_to(dest, target_is_directory=True)
    print(f"linked {DEFAULT_DEST} -> {dest}")


def download_scene(repo, scene, dest: Path, keep_zip=False):
    scene_dir = dest / scene
    marker = scene_dir / ".downloaded"
    if marker.exists():
        print(f"[skip] {scene} already extracted")
        return
    zip_path = hf_hub_download(repo, f"scenes/{scene}.zip", repo_type="dataset", local_dir=dest / "_zips")
    print(f"[extract] {scene} -> {scene_dir}")
    scene_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(scene_dir)
    marker.touch()
    if not keep_zip:
        os.remove(zip_path)


def download_checkpoints(repo, dest: Path):
    for name in ["hurdle.pt", "stairs.pt"]:
        p = hf_hub_download(repo, f"checkpoints/lucidsim/{name}", repo_type="dataset", local_dir=dest)
        print(f"[ckpt] {p}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="*", default=[], help="scene names (see neverwhere/scenes.txt)")
    parser.add_argument("--task", choices=["gaps", "hurdle", "stairs", "ramp"], help="download every scene of this task type")
    parser.add_argument("--all", action="store_true", help="download all benchmark scenes")
    parser.add_argument("--extra", action="store_true", help="also include the 4 sim-to-real parallel-evaluation scenes")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="where scene folders go (default: neverwhere/tasks/real_scenes)")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--checkpoints", action="store_true", help="download the LucidSim example checkpoints to ./checkpoints")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    if args.checkpoints:
        download_checkpoints(args.repo, REPO_ROOT)

    rows = read_scenes()
    selected = []
    for r in rows:
        if r["split"] == "extra" and not (args.extra or r["scene"] in args.scenes):
            continue
        if args.all or args.extra or (args.task and r["task"] == args.task) or r["scene"] in args.scenes:
            selected.append(r["scene"])
    unknown = set(args.scenes) - {r["scene"] for r in rows}
    if unknown:
        parser.error(f"unknown scenes: {sorted(unknown)}")
    if not selected:
        if not args.checkpoints:
            parser.error("nothing selected: pass --scenes, --task, --all, --extra or --checkpoints")
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    link_scenes_root(args.dest)
    for scene in selected:
        download_scene(args.repo, scene, args.dest, keep_zip=args.keep_zip)
    shutil.rmtree(args.dest / "_zips", ignore_errors=True) if not args.keep_zip else None
    print(f"done: {len(selected)} scene(s) in {args.dest}")


if __name__ == "__main__":
    main()
