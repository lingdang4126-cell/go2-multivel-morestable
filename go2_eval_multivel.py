"""
go2_eval_multivel.py
====================
多速度策略评估 + 录像。

亮点：支持"速度序列演示"模式（--demo），自动切换不同速度指令，
展示机器人全向行走能力，适合求职作品集视频。

用法：
  # 普通录像（随机速度）
  python go2_eval_multivel.py -e go2-multivel

  # 指定 checkpoint
  python go2_eval_multivel.py -e go2-multivel --ckpt 5000

  # 演示模式：自动展示前进→转向→后退→侧移→原地转
  python go2_eval_multivel.py -e go2-multivel --demo

  # 演示模式 + 自定义每段时长
  python go2_eval_multivel.py -e go2-multivel --demo --seg_frames 150

复制到 Windows：
  cp logs/go2-multivel/eval_*.mp4 /mnt/d/RL/Genesis/
"""

import argparse
import os
import pickle
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",  type=str, nargs="+", required=True)
    parser.add_argument("--ckpt",            type=int,   default=None,
                        help="指定 checkpoint 编号，默认取最新")
    parser.add_argument("--frames",          type=int,   default=600,
                        help="普通模式录制帧数（默认600=12s@50fps）")
    parser.add_argument("--demo",            action="store_true", default=False,
                        help="演示模式：按预设序列切换速度，展示全向能力")
    parser.add_argument("--seg_frames",      type=int,   default=200,
                        help="演示模式每段帧数（默认200=4s@50fps）")
    parser.add_argument("--cpu",             action="store_true", default=False)
    return parser.parse_args()


# ── 演示速度序列（面试作品集用）─────────────────────────────────────
# 每个元素：(vx, vy, ang_vel, 描述)
DEMO_SEQUENCE = [
    ( 0.0,  0.0,  0.0, "静止站立"),
    ( 1.8,  0.0,  0.0, "前进 1.8 m/s"),
    ( 0.0,  0.0,  0.8, "原地左转"),
    ( 1.8,  0.0,  0.0, "前进 1.8 m/s"),
    ( 0.0,  0.0, -0.8, "原地右转"),
    ( 0.0,  0.4,  0.0, "左侧移"),
    ( 0.0, -0.4,  0.0, "右侧移"),
    (-1.5,  0.0,  0.0, "后退 1.5 m/s"),
    ( 0.6,  0.0,  0.5, "前进+左转"),
    ( 0.0,  0.0,  0.0, "停止"),
]


def load_exp_cfg(log_dir):
    cfgs_pkl = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.isfile(cfgs_pkl):
        raise FileNotFoundError(f"{log_dir} 里没有 cfgs.pkl")

    with open(cfgs_pkl, "rb") as f:
        saved = pickle.load(f)

    env_cfg     = saved["env_cfg"]
    obs_cfg     = saved["obs_cfg"]
    reward_cfg  = saved["reward_cfg"]
    command_cfg = saved["command_cfg"]
    train_cfg   = saved["train_cfg"]
    dr_cfg      = saved.get("dr_cfg", None)

    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, dr_cfg


def pick_ckpt(log_dir, ckpt_num=None):
    ckpts = [f for f in os.listdir(log_dir)
             if f.startswith("model_") and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"{log_dir} 里没有 model_*.pt")
    ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    if ckpt_num is None:
        num = int(ckpts[-1].split("_")[1].split(".")[0])
        print(f"  [ckpt] 自动选最新: {ckpts[-1]}")
        return num
    return ckpt_num


def run_eval(exp_name, args, gs):
    print(f"\n{'='*55}")
    print(f"  {exp_name}  {'[演示模式]' if args.demo else '[普通模式]'}")
    print(f"{'='*55}")

    log_dir = f"logs/{exp_name}"
    if not os.path.isdir(log_dir):
        print(f"  [ERROR] 目录不存在: {log_dir}")
        return None

    try:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, dr_cfg = load_exp_cfg(log_dir)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # eval 时不计算奖励
    reward_cfg["reward_scales"] = {}

    try:
        ckpt_num = pick_ckpt(log_dir, args.ckpt)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 相机注入 ──────────────────────────────────────────────────
    original_build = gs.Scene.build
    cam_holder = {}

    def patched_build(self, *a, **kw):
        cam_holder["cam"] = self.add_camera(
            res=(1280, 720),
            pos=(1.5, -1.5, 0.8),
            lookat=(0.0, 0.0, 0.3),
            fov=40,
            GUI=False,
        )
        return original_build(self, *a, **kw)

    gs.Scene.build = patched_build

    # ── 创建环境 ──────────────────────────────────────────────────
    from go2_env_multivel import Go2EnvMultiVel
    env = Go2EnvMultiVel(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dr_cfg=None,   # eval 关闭 DR
        show_viewer=False,
    )

    gs.Scene.build = original_build
    cam = cam_holder["cam"]

    cam.follow_entity(
        env.robot,
        fixed_axis=(None, -1.5, 0.8),
        smoothing=0.05,
        fix_orientation=False,
    )

    # ── 加载策略 ──────────────────────────────────────────────────
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt_num}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    print(f"  [load] model_{ckpt_num}.pt")

    # ── 录像 ──────────────────────────────────────────────────────
    suffix = "demo" if args.demo else "eval"
    output_path = os.path.abspath(
        os.path.join(log_dir, f"{suffix}_{ckpt_num}.mp4")
    )

    dt = env.dt
    obs_dict = env.reset()

    cam.start_recording()

    if args.demo:
        # ── 演示模式：按序列切换速度 ──────────────────────────────
        total_frames = 0
        fall_count   = 0

        for seg_idx, (vx, vy, w, desc) in enumerate(DEMO_SEQUENCE):
            env.commands[:, 0] = vx
            env.commands[:, 1] = vy
            env.commands[:, 2] = w
            print(f"  [{seg_idx+1:2d}/{len(DEMO_SEQUENCE)}] {desc:20s}  "
                  f"vx={vx:+.1f}  vy={vy:+.1f}  ω={w:+.1f}")

            seg_fell = False
            with torch.no_grad():
                for step in range(args.seg_frames):
                    # 保持速度指令（覆盖env内部的重采样）
                    env.commands[:, 0] = vx
                    env.commands[:, 1] = vy
                    env.commands[:, 2] = w

                    actions = policy(obs_dict)
                    obs_dict, _, dones, infos = env.step(actions)
                    cam.render()
                    total_frames += 1

                    time_out = infos.get("time_outs", torch.zeros(1, device=gs.device))
                    is_fall  = dones[0].item() and not time_out[0].item()

                    if is_fall and not seg_fell:
                        print(f"             ⚠ 摔倒（step {step}），重置继续...")
                        fall_count += 1
                        seg_fell = True
                        # 演示模式：摔倒后重置并继续当前段
                        obs_dict = env.reset()
                        env.commands[:, 0] = vx
                        env.commands[:, 1] = vy
                        env.commands[:, 2] = w

        print(f"\n  演示完成：{total_frames}帧 ({total_frames*dt:.1f}s)，摔倒{fall_count}次")

    else:
        # ── 普通模式：随机速度，连续录制 ──────────────────────────
        print(f"  [rec] {args.frames} 帧 ({args.frames*dt:.1f}s)")
        fall_count   = 0
        total_frames = 0

        with torch.no_grad():
            for step in range(args.frames):
                actions = policy(obs_dict)
                obs_dict, _, dones, infos = env.step(actions)
                cam.render()
                total_frames += 1

                if (step + 1) % 100 == 0:
                    pos = env.base_pos[0].cpu()
                    cmd = env.commands[0].cpu()
                    vel = env.base_lin_vel[0].cpu()
                    print(f"    step {step+1:4d}  "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})  "
                          f"vel_x={vel[0]:+.2f}  "
                          f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f})")

                time_out = infos.get("time_outs", torch.zeros(1, device=gs.device))
                is_fall  = dones[0].item() and not time_out[0].item()
                if is_fall:
                    fall_count += 1
                    obs_dict = env.reset()

        print(f"\n  完成：{total_frames}帧，摔倒{fall_count}次")

    cam.stop_recording(save_to_filename=output_path, fps=50)
    print(f"  ✅ {output_path}")
    return output_path


def main():
    args = get_args()

    import genesis as gs
    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning")

    saved = []
    for exp_name in args.exp_name:
        path = run_eval(exp_name, args, gs)
        if path:
            saved.append(path)

    if saved:
        print(f"\n{'='*55}")
        print("  复制到 Windows：")
        for p in saved:
            print(f"  cp '{p}' /mnt/d/RL/Genesis/")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()