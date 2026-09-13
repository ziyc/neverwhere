"""One environment per (scene, robot, observation mode); ids look like Neverwhere-{robot}-{mode}-{scene}[_Nocones]-cones."""

from pathlib import Path

from neverwhere import add_env
from neverwhere.tasks.parkour import entrypoint

SCENES_FILE = Path(__file__).resolve().parent.parent / "scenes.txt"

PREFIX = "nw"
SPAWN_X_RAND = 0.1
SPAWN_Y_RAND = 0.1
SPAWN_YAW_RAND = 0.1
N_PROPRIO = 53

ROBOTS = ["go1", "go2"]
MODES = ["vision_depth_act", "heightmap_splat", "splat_rgb_act"]

STACK_SIZE = 7  # frame stack the released policies expect


def load_scenes(path=SCENES_FILE):
    scenes = {}
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            scene, task, split, x_noise, y_noise = line.split()
            scenes[scene] = dict(task=task, split=split, x_noise=float(x_noise), y_noise=float(y_noise))
    return scenes


SCENES = load_scenes()


def _scene_kwargs(scene, info, robot, mode, use_cones):
    return dict(
        check_contact_termination=True,
        mode=mode,
        stack_size=STACK_SIZE,
        scene=scene,
        xml_path=f"{PREFIX}-{robot}-{scene}.xml",
        n_proprio=N_PROPRIO,
        x_noise=info["x_noise"],
        y_noise=info["y_noise"],
        spawn_x_rand=SPAWN_X_RAND,
        spawn_y_rand=SPAWN_Y_RAND,
        spawn_yaw_rand=SPAWN_YAW_RAND,
        splat_render_keys=["rgb"],
        use_cones=use_cones,
        robot=robot,
    )


for scene, info in SCENES.items():
    for robot in ROBOTS:
        for mode in MODES:
            add_env(f"Neverwhere-{robot}-{mode}-{scene}-cones", entrypoint, _scene_kwargs(scene, info, robot, mode, True))
            add_env(f"Neverwhere-{robot}-{mode}-{scene}_Nocones-cones", entrypoint, _scene_kwargs(scene, info, robot, mode, False))

    # stacked 3DGS RGB for transformer policies; the LucidSim checkpoints are Go1 only
    add_env(
        f"Neverwhere-go1-transformer_vision_splat-{scene}-cones",
        entrypoint,
        _scene_kwargs(scene, info, "go1", "transformer_vision_splat", True),
    )
