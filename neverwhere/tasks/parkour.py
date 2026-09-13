import os
from dm_control.rl import control
from gym_dmc.wrappers import FlattenObservation
from typing import Literal

from neverwhere.tasks import ROOT
from neverwhere.wrappers.depth_midas_render_wrapper import MidasRenderDepthWrapper
from neverwhere.wrappers.act_observation_wrapper import ACTObservationWrapper
from neverwhere.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from neverwhere.wrappers.history_wrapper import HistoryWrapper
from neverwhere.wrappers.dmc_env import DMCEnv
from neverwhere.wrappers.render_depth_wrapper import RenderDepthWrapper
from neverwhere.wrappers.render_rgb_wrapper import RenderRGBWrapper
from neverwhere.wrappers.reset_wrapper import ResetWrapper
from neverwhere.wrappers.scandots_wrapper import ScandotsWrapper
from neverwhere.wrappers.segmentation_wrapper import SegmentationWrapper
from neverwhere.wrappers.terrain_randomization_wrapper import TerrainRandomizationWrapper
from neverwhere.wrappers.gsplat_wrapper import GSplatWrapper
from neverwhere.wrappers.vision_wrapper import VisionWrapper
from neverwhere.wrappers.transformer_observation_wrapper import TransformerObservationWrapper
DEFAULT_TIME_LIMIT = 25

PHYSICS_TIMESTEP = 0.005  # in XML
DECIMATION = 4
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * DECIMATION


def entrypoint(
    xml_path,
    # waypoint randomization
    y_noise,
    x_noise,
    mode: Literal["vision_depth_act", "heightmap_splat", "splat_rgb_act", "transformer_vision_splat"],
    # whether to add cones as visible geoms at waypoints
    use_cones=False,
    time_limit=DEFAULT_TIME_LIMIT,
    random=None,
    device=None,
    terrain_rand_params=None,
    domain_rand=False,
    move_speed_range=[0.8, 0.8],
    stack_size=1,
    img_memory_length=1,
    robot="go1",
    check_contact_termination=False,
    **kwargs,
):
    if robot == "go1":
        from neverwhere.tasks.base.go1_base import Go1 as RobotModel
        from neverwhere.tasks.base.go1_base import Physics
    elif robot == "go2":
        from neverwhere.tasks.base.go2_base import Go2 as RobotModel
        from neverwhere.tasks.base.go2_base import Physics
    else:
        raise ValueError(f"Unknown robot: {robot}")

    physics = Physics.from_xml_path(os.path.join(ROOT, xml_path))

    if not use_cones:
        model = physics.model
        named_model = physics.named.model
        all_geom_names = [model.geom(i).name for i in range(model.ngeom)]
        # set transparency to 0
        for geom_name in all_geom_names:
            if geom_name.startswith("cone"):
                named_model.geom_rgba[geom_name] = [0, 0, 0, 0]

    task = RobotModel(vision=True, move_speed_range=move_speed_range, y_noise=y_noise, x_noise=x_noise, random=random, **kwargs)
    env = control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=CONTROL_TIMESTEP,
        flat_observation=True,
    )
    env = DMCEnv(env)
    env = FlattenObservation(env)
    env = HistoryWrapper(env, history_len=10)
    env = ResetWrapper(
        env,
        check_contact_termination=check_contact_termination
    )

    if domain_rand:
        env = DomainRandomizationWrapper(
            env,
            randomize_color=True,
            randomize_lighting=True,
            randomize_camera=False,
            randomize_dynamics=False,
            seed=random,
            **kwargs,
        )
        env = RenderRGBWrapper(
            env,
            camera_id="ego-rgb",
            width=1280,
            height=720,
        )

    if terrain_rand_params is not None:
        terrain_type = terrain_rand_params.pop("terrain_type")
        env = TerrainRandomizationWrapper(env, terrain_type=terrain_type, rand_params=terrain_rand_params, random=random)
    
    if mode == "heightmap_splat":
        env = ScandotsWrapper(env, **kwargs, device=device)
        env = MidasRenderDepthWrapper(
            env,
            width=1280,
            height=720,
            camera_id="ego-rgb",
            device=device,
        )
        fill_masks = False
        if use_cones: # need seg and RGB
            env = SegmentationWrapper(
                env,
                width=1280,
                height=720,
                camera_id="ego-rgb",
                groups=[["soccer*", "basketball*", "ball*", "cone*"]],
                **kwargs,
            )
            env = RenderRGBWrapper(
                env,
                camera_id="ego-rgb",
                width=1280,
                height=720,
                **kwargs,
            )
            fill_masks = True
        env = GSplatWrapper(
            env,
            camera_id="ego-rgb",
            width=1280,
            height=720, 
            device=device,
            fill_masks=fill_masks,
            **kwargs,
        )
    elif mode == "vision_depth_act":
        # needed to keep track of the expert observations
        env = ScandotsWrapper(
            env,
            device=device,
        )
        env = VisionWrapper(
            env,
            device=device,
            stack_size=stack_size,
            **kwargs,
        )
        # for render depth
        env = RenderDepthWrapper(
            env,
            width=1280,
            height=720,
            camera_id="ego-rgb",
        )
        env = ACTObservationWrapper(
            env,
            image_key="render_depth",
            img_memory_length = img_memory_length,
            **kwargs,
        )
    elif mode == "splat_rgb_act":
        # needed to keep track of the expert observations
        env = ScandotsWrapper(
            env,
            device=device,
        )
        env = VisionWrapper(
            env,
            device=device,
            stack_size=stack_size,
            **kwargs,
        )
        # for render depth
        env = RenderDepthWrapper(
            env,
            width=1280,
            height=720,
            camera_id="ego-rgb",
        )
        fill_masks = False
        if use_cones: # need seg and RGB
            env = SegmentationWrapper(
                env,
                width=1280,
                height=720,
                camera_id="ego-rgb",
                groups=[["soccer*", "basketball*", "ball*", "cone*"]],
                **kwargs,
            )
            env = RenderRGBWrapper(
                env,
                camera_id="ego-rgb",
                width=1280,
                height=720,
                **kwargs,
            )
            fill_masks = True
        env = GSplatWrapper(
            env,
            camera_id="ego-rgb",
            width=1280,
            height=720, 
            device=device,
            fill_masks=fill_masks,
            **kwargs,
        )
        env = ACTObservationWrapper(
            env,
            image_key="splat_rgb",
            img_memory_length = img_memory_length,
            **kwargs,
        )
    elif mode == "transformer_vision_splat":
        fill_masks = False
        if use_cones:
            # need seg and RGB
            env = SegmentationWrapper(
                env,
                width=1280,
                height=720,
                camera_id="ego-rgb",
                groups=[["soccer*", "basketball*", "ball*", "cone*"]],
                **kwargs,
            )
            env = RenderRGBWrapper(
                env,
                camera_id="ego-rgb",
                width=1280,
                height=720,
                **kwargs,
            )
            fill_masks = True
        env = GSplatWrapper(
            env,
            camera_id="ego-rgb",
            width=1280,
            height=720,
            device=device,
            fill_masks=fill_masks,
            **kwargs,
        )
        env = VisionWrapper(
            env,
            device=device,
            stack_size=stack_size,
            render_type="splat_rgb",
            imagenet_pipe=False,
        )
        env = TransformerObservationWrapper(
            env,
            stack_size=stack_size,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return env