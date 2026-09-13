import torch
import torch.nn as nn
import warnings
from einops import rearrange
from typing import Mapping, Any

from neverwhere.lucidsim.transformer import PalmTransformer

class TransformerPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        img_dim,
        act_dim,
        img_latent_dim,
        head_dim,
        num_layers,
        num_heads=8,
        num_steps=300,
        dropout=0.1,
        causal=False,
        pred_yaw=False,
        batchnorm=False,
    ):
        super().__init__()
        self.transformer = PalmTransformer(
            head_dim,
            num_layers,
            heads=num_heads,
            attn_dropout=dropout,
            ff_dropout=dropout,
            causal=causal,
        )

        self.head_d = head_dim

        self.steps = num_steps
        # 20ms per step, 50 per second, 300 per 3 seconds

        self.obs_enc = nn.Conv1d(obs_dim, head_dim, 1)

        self.img_dim = img_dim

        img_pipe = []
        if batchnorm:
            img_pipe.append(nn.BatchNorm2d(img_dim))
        img_pipe.extend(
            [
                nn.Conv2d(img_dim, 64, 8, 8),
                nn.ReLU(),
                nn.Conv2d(64, head_dim, 2, 1),
            ]
        )

        self.img_enc = nn.Sequential(
            *img_pipe,
        )

        # We use a special set of action tokens. Channel first.
        self.act_tokens = nn.Parameter(torch.randn(1, num_steps, head_dim))

        self.pred_yaw = pred_yaw

        # just one action output right now.
        self.act_proj = nn.Sequential(
            nn.Linear(head_dim, act_dim),
        )

        self.yaw_proj = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Tanh(),
        )


    def forward_yaw(self, frames, obs, **extras):
        """
        dagger_mode: if True, use mixed teacher student supervision -- obs will always contain gt yaw
        """
        B, T, C, H, W = frames.shape

        frames_flattened = rearrange(frames, "b t c h w -> (b t) c h w")
        image_tokens = self.img_enc(frames_flattened).reshape(B, T, self.head_d, -1)  # B, T, head_d, 2x6
        image_tokens_sequence_first = rearrange(image_tokens, "b t c f -> b (t f) c")
        pred_yaw = self.yaw_proj(image_tokens_sequence_first[:, -1]) / 1.5

        obs = obs.clone()

        # # mask out and replace yaw observation
        obs[:, :, 6:8] = 0
        obs[:, -1, 6:8] = pred_yaw

        obs_channel_first = rearrange(obs, "b t c -> b c t")
        obs_tokens = self.obs_enc(obs_channel_first).transpose(1, 2)

        act_input_tokens = self.act_tokens[:, 0].repeat(B, 1, 1)

        all_tokens = torch.cat([obs_tokens, image_tokens_sequence_first, act_input_tokens], dim=1)

        output = self.transformer(all_tokens)

        act = self.act_proj(output[:, -1])

        return act, pred_yaw

    def forward(self, frames, obs, **extras):
        if self.pred_yaw:
            return self.forward_yaw(frames, obs, **extras)

        B, img_T, C, H, W = frames.shape

        frames = frames[:, :, : self.img_dim, :, :]



        obs_channel_first = rearrange(obs, "b t c -> b c t")

        obs_tokens = self.obs_enc(obs_channel_first).transpose(1, 2)

        frames_flattened = rearrange(frames, "b t c h w -> (b t) c h w")
        image_tokens = self.img_enc(frames_flattened).reshape(B, img_T, self.head_d, -1)  # B, T, head_d, 2x6
        image_tokens_sequence_first = rearrange(image_tokens, "b t c f -> b (t f) c")

        # we only predict one action right now, but we can predict a lot more.
        act_input_tokens = self.act_tokens[:, 0].repeat(B, 1, 1)

        all_tokens = torch.cat([obs_tokens, image_tokens_sequence_first, act_input_tokens], dim=1)

        output = self.transformer(all_tokens)

        act = self.act_proj(output[:, -1])

        return act

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # Hack to avoid issues with loading yaw projection
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            # find all keys with 'yaw' in them
            yaw_keys = [k for k in state_dict.keys() if "yaw" in k]
            for k in yaw_keys:
                state_dict.pop(k)
            super().load_state_dict(state_dict, strict=False)

            warnings.warn(
                "Had to pop out the yaw keys. If these were needed, you will deal with the consequences of your actions later",
            )
