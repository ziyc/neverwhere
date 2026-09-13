from pathlib import Path

ROOT = Path(__file__).parent.resolve()
# Scene data (3DGS models + collision meshes) live here, one folder per scene.
# This may be a symlink to a larger disk; see download.py.
SCENES_ROOT = ROOT / "real_scenes"
