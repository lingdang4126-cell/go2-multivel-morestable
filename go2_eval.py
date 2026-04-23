import argparse
import os
import pickle
from importlib import metadata
import torch
try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env import Go2Env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=None)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    original_build = gs.Scene.build
    cam_holder = {}

    def patched_build(self, *args, **kwargs):
        cam_holder['cam'] = self.add_camera(
            res=(1280, 720),
            pos=(0.0, 2.5, 0.8),
            lookat=(0.0, 0.0, 0.3),
            fov=45,
            GUI=False,
        )
        return original_build(self, *args, **kwargs)

    gs.Scene.build = patched_build

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    gs.Scene.build = original_build
    cam = cam_holder['cam']

    # 自动找最新 checkpoint
    if args.ckpt is None:
        ckpts = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        latest = ckpts[-1]
        ckpt_num = int(latest.split("_")[1].split(".")[0])
        print(f"Auto-selected checkpoint: {latest}")
    else:
        ckpt_num = args.ckpt

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt_num}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()

    # 跟随：X轴跟机器人，Y轴固定2.5（侧面），Z轴固定0.8（高度）
    cam.follow_entity(
        env.robot,
        fixed_axis=(None, 2.5, 0.8),  # X跟随，YZ固定
        smoothing=0.1,
    )

    output_path = os.path.abspath(f"logs/{args.exp_name}/go2_follow.mp4")
    print(f"Recording to: {output_path}")
    cam.start_recording()

    max_frames = 500
    print(f"Running {max_frames} steps...")
    with torch.no_grad():
        for i in range(max_frames):
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)
            cam.render()
            if (i + 1) % 100 == 0:
                print(f"  Step {i+1}/{max_frames}")

    cam.stop_recording(save_to_filename=output_path, fps=50)
    print(f"\n✅ Done!")
    print(f"cp {output_path} /mnt/d/RL/Genesis/")

if __name__ == "__main__":
    main()
