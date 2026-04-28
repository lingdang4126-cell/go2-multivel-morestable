"""
go2_train_slope_curriculum.py
=============================
Fine-tune Go2 on slope curriculum terrain from the flat baseline `go2_v2`.

Target curriculum:
- slope degrees: 5, 10, 15, 20, 25
- per-env terrain curriculum enabled
- success: the robot must traverse the full slope tile (uphill + downhill)
- success on the max level: randomize level for the next episode
- failure: demote one level

Example:
  python go2_train_slope_curriculum.py \
      --ckpt logs/go2_v2/model_999.pt \
      --exp_name go2-slope-5to25 \
      --num_envs 1024 \
      --max_iterations 5000
"""

import argparse
import copy
import os
import time

import torch

from gpu_monitor import GPUMonitor, print_gpu_summary
from training_snapshot import save_training_snapshot


def parse_float_list(raw_text):
    values = [x.strip() for x in raw_text.split(",") if x.strip()]
    if not values:
        raise ValueError("list cannot be empty")
    return [float(x) for x in values]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("-e", "--exp_name", type=str, default="go2-slope-from-multivel")
    p.add_argument("--max_iterations", type=int, default=5000)
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--finetune_lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=0.003)
    # 上一轮存活时间约 280 且 entropy/action std 偏高；fine-tune 阶段降低初始探索。
    p.add_argument("--reset_std", type=float, default=0.25)
    p.add_argument("--target_lin_vel", type=float, default=0.35)
    p.add_argument("--horizontal_scale", type=float, default=0.1)
    # 从多速度平地策略迁移时先用低坡度 smoke test，确认 gait 没被地形奖励打散。
    p.add_argument("--slope_levels", type=str, default="0,3,5,8,10")
    p.add_argument("--show_viewer", action="store_true", default=False)
    p.add_argument("--monitor_gpu", action="store_true", default=False)
    p.add_argument("--gpu_monitor_interval", type=float, default=1.0)
    return p.parse_args()


def get_train_cfg(exp_name, entropy_coef):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": entropy_coef,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs(target_lin_vel, slope_levels):
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 20,
        "termination_if_pitch_greater_than": 20,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "terrain_curriculum_cfg": {
            "enabled": True,
            "cache_enabled": True,
            "cache_dir": "logs/terrain_cache",
            "num_levels": len(slope_levels),
            "terrain_types": ["slope_track"],
            "slope_deg_max": max(slope_levels),
            "slope_deg_levels": slope_levels,
            "promote_boundary_ratio": 0.9,
            "demote_expected_dist_ratio": 0.5,
            # 最高难度成功后保持在高难度附近，而不是随机回低难度，增加 12-15deg 采样密度。
            "randomize_on_max_success": False,
            "subterrain_size": (6.0, 3.0),
            "spawn_mode": "track_start",
            "spawn_track_start_x_ratio": -0.30,
            "spawn_track_y_ratio": 0.04,
            "complete_track_ratio": 0.80,
        },
    }

    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.34,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -1.0,
            "action_rate": -0.02,
            "similar_to_default": -0.2,
            "orientation": -2.0,
            "ang_vel_xy": -0.05,
            # 上一轮失败主要是高坡 timeout 而不是摔倒；这里把“持续向前爬”和
            # “完成整段坡道”显式变成更大的学习信号，并惩罚到期仍没走完的轨迹。
            "forward_progress": 2.5,
            "uphill_progress": 2.0,
            "stall": -1.5,
            "completion": 2.0,
            "timeout_incomplete": -0.5,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [target_lin_vel, target_lin_vel],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }

    dr_cfg = {
        "friction_ratio_range": [0.9, 1.1],
        "mass_scale_range": [0.95, 1.05],
        "dof_frictionloss_range": [0.0, 0.03],
        "enable_dof_frictionloss_randomization": False,
        "push_force_max": 0.0,
        "push_interval_steps": 100000000,
        "push_vel_max": 0.0,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, dr_cfg


def print_post_training_commands(exp_name, ckpt_num):
    print("\n[commands] evaluate latest slope policy:")
    print(
        "  python go2_eval_slope_traverse.py "
        f"-e {exp_name} --ckpt {ckpt_num} "
        "--num_envs 16 --episodes_per_level 20 --max_episode_steps 800"
    )
    print("\n[commands] record key slope videos:")
    print(
        "  python go2_eval_slope_traverse.py "
        f"-e {exp_name} --ckpt {ckpt_num} "
        "--num_envs 16 --episodes_per_level 5 --max_episode_steps 800 "
        "--levels 10,12,15 --record --record_steps 500"
    )


def main():
    args = get_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    slope_levels = parse_float_list(args.slope_levels)
    print("\n" + "=" * 64)
    print("Go2 slope curriculum fine-tune")
    print(f"  ckpt            : {args.ckpt}")
    print(f"  exp_name        : {args.exp_name}")
    print(f"  num_envs        : {args.num_envs}")
    print(f"  max_iterations  : {args.max_iterations}")
    print(f"  target_lin_vel  : {args.target_lin_vel}")
    print(f"  horizontal_scale: {args.horizontal_scale}")
    print(f"  slope_levels    : {slope_levels}")
    print(f"  monitor_gpu     : {args.monitor_gpu}")
    print("=" * 64 + "\n")

    from importlib import metadata

    try:
        if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
            raise ImportError
    except Exception as e:
        raise ImportError("please install rsl-rl-lib>=5.0.0") from e

    import genesis as gs
    from go2_env_dr import Go2EnvDR
    from rsl_rl.runners import OnPolicyRunner

    env_cfg, obs_cfg, reward_cfg, command_cfg, dr_cfg = get_cfgs(args.target_lin_vel, slope_levels)
    train_cfg = get_train_cfg(args.exp_name, args.entropy_coef)

    log_dir = f"logs/{args.exp_name}"

    snapshot = {
        "source_ckpt": args.ckpt,
        "args": vars(args),
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "command_cfg": command_cfg,
        "train_cfg": train_cfg,
        "dr_cfg": dr_cfg,
        "use_terrain": True,
        "horizontal_scale": args.horizontal_scale,
    }
    final_ckpt = max(0, int(args.max_iterations) - 1)
    snapshot_paths = save_training_snapshot(
        log_dir,
        snapshot,
        title="Go2 slope curriculum training parameters",
        notes=[
            "reset_std defaults to 0.25 because the previous run had high entropy/action std.",
            f"entropy_coef={args.entropy_coef} controls exploration during fine-tuning.",
            "completion and timeout_incomplete are softened to reduce critic target jumps.",
        ],
        commands=[
            (
                "eval",
                f"python go2_eval_slope_traverse.py -e {args.exp_name} --ckpt {final_ckpt} "
                "--num_envs 16 --episodes_per_level 20 --max_episode_steps 800",
            ),
            (
                "record key slopes",
                f"python go2_eval_slope_traverse.py -e {args.exp_name} --ckpt {final_ckpt} "
                "--num_envs 16 --episodes_per_level 5 --max_episode_steps 800 "
                "--levels 10,12,15 --record --record_steps 500",
            ),
        ],
        extra_pickle_filenames=["finetune_cfgs.pkl"],
    )
    print(f"[cfg] training params txt: {snapshot_paths['txt_path']}")

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.show_viewer,
    )

    env = Go2EnvDR(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dr_cfg=dr_cfg,
        use_terrain=True,
        horizontal_scale=args.horizontal_scale,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.load(args.ckpt)
    runner.current_learning_iteration = 0

    if args.reset_std > 0:
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.fill_(args.reset_std)
        print(f"[init] reset std_param to {args.reset_std}")

    for pg in runner.alg.optimizer.param_groups:
        pg["lr"] = args.finetune_lr

    print(f"[train] lr={args.finetune_lr}")
    print(f"[train] entropy_coef={args.entropy_coef}")
    gpu_monitor = GPUMonitor(args.monitor_gpu, log_dir, args.gpu_monitor_interval)
    train_start = time.perf_counter()
    try:
        gpu_monitor.start()
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        gpu_monitor.stop()

    train_elapsed = time.perf_counter() - train_start
    gpu_summary = gpu_monitor.summary()

    print(f"\n[done] logs in: {log_dir}")
    print(f"[done] wall time: {train_elapsed / 60.0:.2f} min")
    print_gpu_summary(gpu_summary)

    final_ckpt = max(0, int(args.max_iterations) - 1)
    print_post_training_commands(args.exp_name, final_ckpt)


if __name__ == "__main__":
    main()
