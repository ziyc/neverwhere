import gym


class MidasRenderDepthWrapper(gym.Wrapper):
    """Inverse depth normalized per frame, mimicking MiDaS output."""

    def __init__(
        self,
        env,
        *,
        width=1280,
        height=768,
        camera_id="ego_rgb-render",
        # near=0.05,
        # far=40,
        **_,
    ):
        super().__init__(env)

        self.env = env
        self.width = width
        self.height = height
        self.camera_id = camera_id
        # self.near = near
        # self.far = far

        self.fovy = self.unwrapped.env.physics.named.model.cam_fovy[self.camera_id]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if "depth_frame" in info:
            frame = info["depth_frame"]
        else:
            frame = self.render(
                "depth",
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            )
            info["depth_frame"] = frame

        # invert
        frame = 1 / frame

        # normalize each frame by its min and max
        frame = (frame - frame.min()) / (frame.max() - frame.min())

        info["midas_depth"] = frame
        return obs, rew, done, info
