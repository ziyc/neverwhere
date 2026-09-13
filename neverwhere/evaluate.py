"""Evaluate the LucidSim transformer policy on a Neverwhere scene.

Example:
    MUJOCO_GL=egl python -m neverwhere.evaluate \
        --scene hurdle_stata_v1 --checkpoint checkpoints/lucidsim/hurdle.pt --episodes 10 --video

To evaluate your own policy, replace `load_policy` and the `policy(info["vision"], obs)` call in `run_episode`,
or build the env with `neverwhere.make(...)` and reuse `run_episode`.

Writes ``<out>/<scene>/episode_XXXX.json`` per episode, ``<out>/<scene>/summary.json``,
and optionally a third-person mp4 per episode.
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path

import dm_control
import numpy as np
import torch
from tqdm import trange

import neverwhere
from neverwhere.lucidsim import TransformerPolicy

NUM_STEPS = 600  # 600 control steps at 50 Hz = 12 s
ACTION_DELAY = 1  # apply the action from one control step ago (matches deployment latency)
ACT_DIM = 12


def load_policy(checkpoint, device):
    policy = TransformerPolicy(
        obs_dim=53, img_dim=3, act_dim=ACT_DIM, img_latent_dim=64, head_dim=128, batchnorm=True, num_layers=5, dropout=0.1, causal=False
    )
    state_dict = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(state_dict)
    return policy.to(device).eval()


def run_episode(env_name, policy, seed, device, render=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = neverwhere.make(env_name, device=device, random=seed, img_memory_length=1)
    env.reset()
    task = env.unwrapped.env.task

    action_buffer = deque([np.zeros(ACT_DIM)] * 5, maxlen=5)
    frames = {"render": [], "splat_rgb": []}  # third-person MuJoCo view, and the 3DGS ego view the policy sees
    metrics = {}
    status = "timeout"
    try:
        for _ in trange(NUM_STEPS, leave=False):
            obs, _, done, info = env.step(action_buffer[-1 - ACTION_DELAY])
            metrics = task.get_metrics()
            if metrics["frac_goals_reached"] == 1.0:
                status = "success"
                break
            if done:
                status = "terminated"
                break
            if render:
                frames["render"].append(env.render(camera_id="tracking-2", width=640, height=300))
                frames["splat_rgb"].append(info["splat_rgb"])

            with torch.no_grad():
                action = policy(info["vision"], torch.from_numpy(obs).float().to(device))
            action_buffer.append(action.cpu().numpy())
    except dm_control.rl.control.PhysicsError:
        status = "physics_error"
    finally:
        env.close()

    return dict(seed=seed, status=status, **{k: float(v) for k, v in metrics.items()}), frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene name, see neverwhere/scenes.txt")
    parser.add_argument("--checkpoint", required=True, help="path to hurdle.pt / stairs.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000, help="first seed; episode i uses seed + i")
    parser.add_argument("--out", default="results/lucidsim")
    parser.add_argument("--video", action="store_true", help="save a third-person mp4 per episode")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    env_name = f"Neverwhere-go1-transformer_vision_splat-{args.scene}-cones"
    out = Path(args.out) / args.scene
    out.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.checkpoint, args.device)
    episodes = []
    for i in range(args.episodes):
        seed = args.seed + i
        result, frames = run_episode(env_name, policy, seed, args.device, render=args.video)
        episodes.append(result)
        (out / f"episode_{i:04d}.json").write_text(json.dumps(result, indent=2))
        if args.video and frames["render"]:
            import imageio

            imageio.mimwrite(out / f"episode_{i:04d}.mp4", np.stack(frames["render"]), fps=48)
            imageio.mimwrite(out / f"episode_{i:04d}_ego_splat.mp4", np.stack(frames["splat_rgb"]), fps=48)
        print(f"[{args.scene}] episode {i}: {result}")

    summary = dict(
        scene=args.scene,
        env=env_name,
        checkpoint=str(args.checkpoint),
        num_episodes=len(episodes),
        success_rate=float(np.mean([e["status"] == "success" for e in episodes])),
        avg_frac_goals_reached=float(np.mean([e.get("frac_goals_reached", 0.0) for e in episodes])),
        avg_x_displacement=float(np.mean([e.get("x_displacement", 0.0) for e in episodes])),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
