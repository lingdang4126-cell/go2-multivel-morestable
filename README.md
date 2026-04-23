# Go2 Locomotion (Genesis + rsl-rl)

本项目是一个 Unitree Go2 强化学习训练与评估工程，核心目标是让策略在仿真中实现稳定行走与多速度全向运动。

## 1. 项目基于什么

本项目主要基于以下技术栈：

- Genesis: 物理仿真与并行环境执行。
- rsl-rl (OnPolicyRunner + PPO): 策略训练框架。
- PyTorch: 网络与优化器实现。
- Go2 URDF: 机器人模型与关节定义。

训练脚本采用统一结构：

- 环境配置: env_cfg / obs_cfg / reward_cfg / command_cfg。
- 训练配置: PPO 算法参数、Actor/Critic 网络结构、采样步数等。
- 日志与模型: 保存到 logs/<exp_name>/。

## 2. 环境依赖

- Python 3.10+
- Genesis
- rsl-rl-lib >= 5.0.0
- PyTorch (CUDA)

建议在 linux 环境运行。

## 3. 普通训练 (normal train)

普通训练入口是 go2_train.py，对应环境 go2_env.py。

示例：

```bash
python go2_train.py -e go2_v2 -B 4096 --max_iterations 5000 --seed 1
```

参数说明：

- -e: 实验名，对应 logs/<exp_name>/。
- -B: 并行环境数 (num_envs)。
- --max_iterations: 训练迭代数。

## 4. 多速度训练 (multivel) 

多速度训练入口是 go2_train_multivel.py，对应环境 go2_env_multivel.py。

核心思想是把速度指令作为策略输入，并在训练中持续随机切换目标速度，让策略学习从“当前状态”跟踪“变化中的速度命令”。

具体做法：

1. 指令空间扩展为全向三维命令：
  - vx: 前进/后退
  - vy: 左右侧移
  - wz: 左右转向

2. 环境定时重采样命令：
  - 每隔固定秒数随机采样新命令
  - 含小速度 dead-zone，鼓励策略学会真正“停住”而非微抖

3. 观测中显式包含命令项：
  - 让策略同时感知机体状态和目标速度

4. 奖励围绕“跟踪 + 稳定 + 平滑”：
  - tracking_lin_vel / tracking_ang_vel
  - 惩罚竖直速度、姿态抖动、动作变化率等

5. 训练后用 demo 或交互脚本做行为验证：
  - 离线分段命令录像
  - 实时按键控制检查响应性

基础训练示例：

```bash
python go2_train_multivel.py -e go2-multivel --num_envs 4096 --max_iterations 5000 --seed 1
```

可视化调试（小环境数）：

```bash
python go2_train_multivel.py -e go2-multivel --num_envs 16 --max_iterations 300 --show_viewer
```

## 5. 从 go2-multivel 到 go2-multivel_morestable

go2-multivel_morestable 的目标是：

- 保持全向跟踪能力
- 降低摔倒率
- 提升动作平滑性与可控性

实践上通过“同框架继续训练 + 稳定性导向调参”实现：

- 更强的平滑与姿态约束。
- 更温和的扰动随机化。
- 保持相同命令空间，避免能力回退。

示例：

```bash
python go2_train_multivel.py -e go2-multivel_morestable --num_envs 4096 --max_iterations 5000 --seed 1
```

## 6. 评估与演示

离线录像评估：

```bash
python go2_eval_multivel.py -e go2-multivel --ckpt 5000 --demo
python go2_eval_multivel.py -e go2-multivel_morestable --ckpt 5000 --demo
```

交互控制评估（支持方向键/WASD与录像）：

```bash
python go2_eval_multivel_command.py -e go2-multivel_morestable --ckpt 5000 --live_keys --show_viewer
python go2_eval_multivel_command.py -e go2-multivel_morestable --ckpt 5000 --live_keys --show_viewer --record
```

## 7. 建议关注指标

训练与对比时建议重点关注：

- episode length
- tracking_lin_vel / tracking_ang_vel
- action_std 或动作方差
- 评估中的摔倒次数与恢复能力

## 8. 备注

- 若 checkpoint 体积较大，建议保留关键里程碑模型。
- 本仓库已提供 LICENSE 与 requirements.txt，可直接用于开源发布与环境安装。
