import json

import gym
import numpy as np
import torch

from neverwhere.gsplat.neverwhere_splat_model import Model
from neverwhere.tasks import SCENES_ROOT
from neverwhere.utils.tf_utils import get_camera_extrinsic_matrix


class GSplatWrapper(gym.Wrapper):
    """Renders the scene's 3D Gaussian splat from the ego camera and adds it to info as `splat_rgb`.

    fill_masks: paste the MuJoCo RGB render over the splat wherever the segmentation wrapper found a
    task object (cones), since those are not part of the scanned scene.
    """

    def __init__(
        self,
        env,
        *,
        scene,
        device,
        width=1280,
        height=768,
        camera_id="ego-rgb",
        splat_render_keys=["rgb"],
        fill_masks=True,
        **_,
    ):
        super().__init__(env)

        self.env = env
        self.width = width
        self.height = height
        self.camera_id = camera_id
        self.fovy = self.unwrapped.env.physics.named.model.cam_fovy[camera_id]
        self.fill_masks = fill_masks
        self.render_keys = splat_render_keys

        self.fx = 0.5 * self.height / np.tan(self.fovy * np.pi / 360)
        self.fy = self.fx

        scene_dir = SCENES_ROOT / scene
        model_path = scene_dir / "3dgs" / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"3DGS model not found at {model_path}. Download the scene first, e.g.\n"
                f"  python download.py --scenes {scene}"
            )
        self.model = Model(device=device)
        self.model.load_ckpt(torch.load(model_path, map_location=device))
        self.model.eval()

        # splat -> mesh transform written by the 3DGS trainer; absent when the splat is already
        # in the mesh frame
        tf_path = scene_dir / "3dgs" / "data_transforms.json"
        self.scale, self.transform = 1.0, np.eye(4)
        if tf_path.exists():
            with open(tf_path) as f:
                data = json.load(f)
            self.scale = data["scale"]
            self.transform[:3, :4] = np.array(data["transform"])

        physics = self.unwrapped.env.physics
        mesh_translation = physics.named.data.xpos["mesh"]
        mesh_rot = physics.named.data.xmat["mesh"].reshape(3, 3)
        # use the raw model: dm_control's named indexer does not expose mesh_scale on every mujoco version
        mesh_scale = physics.model.mesh_scale[physics.model.mesh("visual_mesh").id]

        # mesh -> mujoco: scale first, then rotate and translate
        mesh_tf = np.eye(4)
        mesh_tf[:3, :3] = mesh_rot @ np.diag(mesh_scale)
        mesh_tf[:3, 3] = mesh_translation

        # splat -> mujoco (self.transform already includes the dataparser scale)
        self.model_tf = mesh_tf @ np.linalg.inv(self.transform)

    def splat_render(self, env_info, cam_info, *render_keys):
        outputs = self.model.get_simple_outputs(**cam_info)

        renders = {}
        if "rgb" in render_keys:
            rgb_np = (outputs["rgb"].clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
            if self.fill_masks and "masks" in env_info and "render_rgb" in env_info:
                for group_id, (mask_in, mask_out) in env_info["masks"].items():
                    if group_id == "sky":
                        continue
                    rgb_np[mask_in] = env_info["render_rgb"][mask_in]
            renders["splat_rgb"] = rgb_np
        if "depth" in render_keys:
            raise NotImplementedError("depth comes from the collision mesh (RenderDepthWrapper), not the splat")

        return renders

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        c2w = get_camera_extrinsic_matrix(self.unwrapped.env.physics, self.camera_id, axis_correction=False)
        c2w = np.linalg.inv(self.model_tf) @ c2w

        cam_info = dict(c2w=c2w, fx=self.fx, fy=self.fy, cx=self.width / 2, cy=self.height / 2, width=self.width, height=self.height)
        info["cam_info"] = cam_info
        info.update(self.splat_render(info, cam_info, *self.render_keys))

        return obs, rew, done, info
