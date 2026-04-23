"""
go2_train_multivel.py
=====================
多速度指令训练入口。从头训练，支持全向行走。

用法：
  # 本机（WSL2 / Linux）
  python go2_train_multivel.py --num_envs 4096 --max_iterations 5000

  # AutoDL 5090（显存大，可以跑更多env）
  python go2_train_multivel.py --num_envs 8192 --max_iterations 5000

训练健康指标参考（1000it时）：
  episode_length  > 500   ✅  机器人没有频繁摔倒
  tracking_lin_vel > 0.5  ✅  开始学会跟踪速度
  action_std       < 0.8  ✅  策略在收敛
"""

import argparse
import os
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",     type=str,   default="go2-multivel")
    parser.add_argument("--max_iterations",     type=int,   default=5000)
    parser.add_argument("--num_envs",           type=int,   default=4096)
    parser.add_argument("--seed",               type=int,   default=1)
    parser.add_argument("--show_viewer",        action="store_true", default=False,
                        help="实时显示 viewer（会显著降低训练速度）")
    return parser.parse_args()


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name":             "PPO",
            "clip_param":             0.2,
            "desired_kl":             0.01,
            "entropy_coef":           0.005,  # 略高于0，保持一定探索
            "gamma":                  0.99,
            "lam":                    0.95,
            "learning_rate":          1e-3,
            "max_grad_norm":          1.0,
            "num_learning_epochs":    5,
            "num_mini_batches":       4,
            "schedule":               "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef":        1.0,
        },
        "actor": {
            "class_name":  "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation":  "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std":   1.0,
                "std_type":   "scalar",
            },
        },
        "critic": {
            "class_name":  "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation":  "elu",
        },
        "obs_groups":        {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval":     200,
        "run_name":          exp_name,
        "logger":            "tensorboard",
    }


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint":    0.0,  "FR_hip_joint":    0.0,
            "RL_hip_joint":    0.0,  "RR_hip_joint":    0.0,
            "FL_thigh_joint":  0.8,  "FR_thigh_joint":  0.8,
            "RL_thigh_joint":  1.0,  "RR_thigh_joint":  1.0,
            "FL_calf_joint":  -1.5,  "FR_calf_joint":  -1.5,
            "RL_calf_joint":  -1.5,  "RR_calf_joint":  -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        # 平地用25度，比原来10度宽松，避免DR扰动下误触发
        "termination_if_roll_greater_than":  25,
        "termination_if_pitch_greater_than": 25,
        "base_init_pos":  [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s":    20.0,
        "resampling_time_s":   4.0,   # 每4秒随机切换速度指令
        "action_scale":        0.25,
        "simulate_action_latency": True,
        "clip_actions":        100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma":     0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            # ── 主要追踪目标 ──────────────────────────────────────
            "tracking_lin_vel":    2.0,   # 线速度追踪（前进/后退/侧移）
            "tracking_ang_vel":    0.5,   # 角速度追踪（转向）

            # ── 稳定性惩罚 ────────────────────────────────────────
            "lin_vel_z":          -2.0,   # 禁止跳跃 v1 = -1.0
            "ang_vel_xy":         -0.1,  # 禁止不必要的俯仰/横滚角速度 v1=-0.05
            "base_height":        -1.0,   # 保持身体高度0.3m
            "action_rate":        -0.05, # 平滑动作 v1=-0.005
            "similar_to_default": -0.5,   # 自然站姿 v1=-0.1
        },
    }
    command_cfg = {
        "num_commands": 3,
        # ── 全向速度范围 ──────────────────────────────────────────
        "lin_vel_x_range": [-1.0,  1.0],  # 前进0~1 m/s，后退0~-1 m/s
        "lin_vel_y_range": [-0.5,  0.5],  # 侧移
        "ang_vel_range":   [-1.0,  1.0],  # 转向（rad/s）
    }
    dr_cfg = {
        # B-v2 已验证的温和DR参数
        "friction_ratio_range":   [0.8,1.2], #[0.7, 1.3],
        "dof_frictionloss_range": [0.0,0.05],#[0.0, 0.08],
        "push_vel_max":           0.1,#0.2,
        "push_interval_steps":    800,#500,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, dr_cfg


def main():
    args = get_args()

    print(f"\n{'='*60}")
    print(f"  Go2 多速度训练")
    print(f"  exp_name    : {args.exp_name}")
    print(f"  num_envs    : {args.num_envs}")
    print(f"  iterations  : {args.max_iterations}")
    print(f"  show_viewer : {args.show_viewer}")
    print(f"  速度范围    : vx∈[-1,1]  vy∈[-0.5,0.5]  ω∈[-1,1]")
    print(f"{'='*60}\n")

    if args.show_viewer and args.num_envs > 64:
        print("[WARN] show_viewer=True 且 num_envs 很大，实时渲染可能非常卡。")
        print("       建议先用 --num_envs 8~32 做可视化检查。\n")

    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from go2_env_multivel import Go2EnvMultiVel

    env_cfg, obs_cfg, reward_cfg, command_cfg, dr_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    log_dir = f"logs/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)

    # 保存配置（eval脚本会读取）
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump({
            "env_cfg":     env_cfg,
            "obs_cfg":     obs_cfg,
            "reward_cfg":  reward_cfg,
            "command_cfg": command_cfg,
            "train_cfg":   train_cfg,
            "dr_cfg":      dr_cfg,
            "use_terrain": False,
            "env_class":   "Go2EnvMultiVel",
        }, f)

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.show_viewer,
    )

    env = Go2EnvMultiVel(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dr_cfg=dr_cfg,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    print("[Train] 开始训练...\n")
    print("  健康指标（1000it时）：")
    print("    episode_length   > 500   机器人不频繁摔倒")
    print("    tracking_lin_vel > 0.5   开始跟踪速度")
    print("    action_std       < 0.8   策略收敛中\n")

    try:
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )
    except KeyboardInterrupt:
        save_path = f"{log_dir}/model_interrupted_{runner.current_learning_iteration}.pt"
        runner.save(save_path)
        print(f"\n[Train] 中断，已保存: {save_path}")

    print(f"\n[Train] 完成！模型在: {log_dir}/")
    print(f"\neval 命令：")
    print(f"  python go2_eval_multivel.py -e {args.exp_name}")


if __name__ == "__main__":
    main()