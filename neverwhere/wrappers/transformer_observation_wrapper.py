from collections import deque

import gym
import numpy as np


class TransformerObservationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        stack_size=7,
        **_,
    ):
        super().__init__(env)

        self.env = env
        self.stack_size = stack_size

        self.obs_buf = None

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # kept for teacher-student training: the privileged observation before stacking
        info["teacher_obs"] = obs.copy()

        # NOTE: obs_buf is never assigned, so the buffer is rebuilt every step and the stack holds
        # stack_size copies of the current obs. The released LucidSim checkpoints were trained and
        # evaluated with this behaviour, so it is kept on purpose.
        if self.obs_buf is None:
            self.obs_buffer = deque([obs] * self.stack_size, maxlen=self.stack_size)
        else:
            self.obs_buffer.append(obs)

        obs_buf = np.array(self.obs_buffer).squeeze(1)[None, ...]
        obs_history = obs_buf[:, :, :53]

        return obs_history, rew, done, info
