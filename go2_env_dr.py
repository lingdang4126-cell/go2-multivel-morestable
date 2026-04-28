import math
import hashlib
import json
import os
import time
import numpy as np

import genesis as gs
import torch
from tensordict import TensorDict

from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
    xyz_to_quat,
)
from go2_urdf_utils import ensure_go2_neutral_safe_urdf, restore_go2_calf_limits


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape),
        dtype=gs.tc_float,
        device=gs.device,
    ) + lower


class Go2EnvDR:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        dr_cfg=None,
        use_terrain=True,
        horizontal_scale=0.1,
        debug_cfg=None,
    ):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.enable_dr = dr_cfg is not None
        self.dr_cfg = dr_cfg if dr_cfg is not None else {}
        self.use_terrain = use_terrain
        self.horizontal_scale = horizontal_scale
        self._step_count = 0

        self.debug_cfg = debug_cfg if debug_cfg is not None else {}
        self.enable_debug_print = bool(self.debug_cfg.get("enable_print", False))
        self.debug_print_interval = max(1, int(self.debug_cfg.get("print_interval", 100)))

        self.terrain_layout = None
        self.terrain_curriculum_cfg = self.env_cfg.get("terrain_curriculum_cfg", {})
        self.terrain_curriculum_enabled = bool(
            self.use_terrain and self.terrain_curriculum_cfg.get("enabled", False)
        )

        self.slope_deg_levels_cfg = self.terrain_curriculum_cfg.get("slope_deg_levels", None)
        self.stairs_step_height_levels_cfg = self.terrain_curriculum_cfg.get(
            "stairs_step_height_levels", None
        )
        self.stairs_num_steps_levels_cfg = self.terrain_curriculum_cfg.get(
            "stairs_num_steps_levels", None
        )

        self.curriculum_num_levels = int(self.terrain_curriculum_cfg.get("num_levels", 4))
        explicit_level_counts = []
        for raw_levels in (
            self.slope_deg_levels_cfg,
            self.stairs_step_height_levels_cfg,
            self.stairs_num_steps_levels_cfg,
        ):
            if raw_levels is not None and len(raw_levels) > 0:
                explicit_level_counts.append(len(raw_levels))
        if explicit_level_counts:
            self.curriculum_num_levels = max(explicit_level_counts)

        self.curriculum_type_names = list(
            self.terrain_curriculum_cfg.get(
                "terrain_types",
                [
                    "random_uniform_terrain",
                    "pyramid_sloped_terrain",
                    "pyramid_stairs_terrain",
                    "discrete_obstacles_terrain",
                ],
            )
        )
        self.curriculum_num_types = len(self.curriculum_type_names)
        self.curriculum_subterrain_size = tuple(
            self.terrain_curriculum_cfg.get("subterrain_size", (6.0, 6.0))
        )
        self.curriculum_boundary_ratio = float(
            self.terrain_curriculum_cfg.get("promote_boundary_ratio", 0.9)
        )
        self.curriculum_demote_ratio = float(
            self.terrain_curriculum_cfg.get("demote_expected_dist_ratio", 0.5)
        )
        self.curriculum_randomize_max = bool(
            self.terrain_curriculum_cfg.get("randomize_on_max_success", True)
        )
        self.curriculum_cache_enabled = bool(
            self.terrain_curriculum_cfg.get("cache_enabled", True)
        )
        self.curriculum_cache_dir = self.terrain_curriculum_cfg.get(
            "cache_dir", "logs/terrain_cache"
        )
        self.curriculum_spawn_mode = str(
            self.terrain_curriculum_cfg.get("spawn_mode", "center")
        )
        self.curriculum_spawn_random_x_ratio = float(
            self.terrain_curriculum_cfg.get("spawn_random_x_ratio", 0.35)
        )
        self.curriculum_spawn_random_y_ratio = float(
            self.terrain_curriculum_cfg.get("spawn_random_y_ratio", 0.12)
        )
        self.curriculum_spawn_track_start_x_ratio = float(
            self.terrain_curriculum_cfg.get("spawn_track_start_x_ratio", -0.38)
        )
        self.curriculum_spawn_track_y_ratio = float(
            self.terrain_curriculum_cfg.get("spawn_track_y_ratio", 0.06)
        )
        self.curriculum_complete_track_ratio = float(
            self.terrain_curriculum_cfg.get("complete_track_ratio", 0.80)
        )
        self.curriculum_edge_reset_ratio = float(
            self.terrain_curriculum_cfg.get("edge_reset_ratio", 0.45)
        )
        self.curriculum_edge_behavior = str(
            self.terrain_curriculum_cfg.get("edge_behavior", "reset")
        )
        self.curriculum_turnaround_inset = float(
            self.terrain_curriculum_cfg.get("turn_around_inset", 0.25)
        )
        self.curriculum_spawn_center_jitter = float(
            self.terrain_curriculum_cfg.get("spawn_center_jitter", 0.05)
        )
        self.slope_start_offset_m = float(
            self.terrain_curriculum_cfg.get("slope_start_offset", 1.2)
        )
        self.slope_ramp_length_m = float(
            self.terrain_curriculum_cfg.get("slope_ramp_length", 1.2)
        )

        # 每个机器人维护独立地形类型/难度等级，reset 时按规则升降。
        self.env_terrain_type = None
        self.env_terrain_level = None
        self.env_start_pos = None
        self.env_traverse_dir = None
        self.terrain_centers_x = None
        self.terrain_centers_y = None

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=64,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        if self.use_terrain:
            if self.terrain_curriculum_enabled:
                # 按“地形类型 x 难度等级”组织地形网格：
                # 行: 随机起伏/斜坡/台阶/离散障碍物，列: 难度 0..N-1。
                n_rows = self.curriculum_num_types
                n_cols = self.curriculum_num_levels
                self.terrain_layout = [
                    [self.curriculum_type_names[r] for _ in range(n_cols)]
                    for r in range(n_rows)
                ]

                curriculum_hf = self._build_curriculum_height_field()
                terrain_origin = self._get_centered_terrain_origin(n_rows, n_cols)

                self.scene.add_entity(
                    morph=gs.morphs.Terrain(
                        pos=terrain_origin,
                        n_subterrains=(n_rows, n_cols),
                        subterrain_size=self.curriculum_subterrain_size,
                        horizontal_scale=self.horizontal_scale,
                        vertical_scale=0.005,
                        # 传入自定义高度场后，subterrain_types 仅用于日志描述。
                        subterrain_types=self.terrain_layout,
                        height_field=curriculum_hf,
                        randomize=False,
                    ),
                )
            else:
                self.terrain_layout = [
                    ["flat_terrain", "flat_terrain", "flat_terrain"],
                    ["flat_terrain", "pyramid_sloped_terrain", "flat_terrain"],
                    ["flat_terrain", "flat_terrain", "flat_terrain"],
                ]
                terrain_origin = self._get_centered_terrain_origin(3, 3, subterrain_size=(6.0, 6.0))
                self.scene.add_entity(
                    morph=gs.morphs.Terrain(
                        pos=terrain_origin,
                        n_subterrains=(3, 3),
                        subterrain_size=(6.0, 6.0),
                        horizontal_scale=self.horizontal_scale,
                        vertical_scale=0.005,
                        subterrain_types=self.terrain_layout,
                        randomize=True,
                    ),
                )
        else:
            self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=ensure_go2_neutral_safe_urdf(),
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        build_start = time.perf_counter()
        print(f"[Go2EnvDR] scene.build 开始: n_envs={num_envs}")
        self.scene.build(n_envs=num_envs)
        print(f"[Go2EnvDR] scene.build 完成, 用时 {time.perf_counter() - build_start:.2f}s")

        if show_viewer:
            try:
                cam = self.scene.add_camera(
                    res=(1280, 720),
                    pos=(1.6, -1.6, 0.9),
                    lookat=(0.0, 0.0, 0.3),
                    fov=40,
                    GUI=False,
                )
                cam.follow_entity(
                    self.robot,
                    fixed_axis=(None, -1.6, 0.9),
                    smoothing=0.08,
                    fix_orientation=False,
                )
            except Exception as e:
                print(f"[Go2EnvDR] warning: add_camera/follow_entity failed: {e}")

        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        calf_dof_idx = [
            self.robot.get_joint(name).dof_start
            for name in self.env_cfg["joint_names"]
            if name.endswith("_calf_joint")
        ]
        restore_go2_calf_limits(self.robot, calf_dof_idx)

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        self.init_base_pos = torch.tensor(
            self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device
        )
        self.init_base_quat = torch.tensor(
            self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate(
            (self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        )
        self.init_projected_gravity = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )

        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
        )
        self.rew_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), dtype=gs.tc_int, device=gs.device
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.commands_limits = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = {}

        if self.terrain_curriculum_enabled:
            # 课程地形块中心坐标：用于将不同 env 重置到不同难度区域。
            xs = [
                (ri - (self.curriculum_num_types - 1) * 0.5) * self.curriculum_subterrain_size[0]
                for ri in range(self.curriculum_num_types)
            ]
            ys = [
                (ci - (self.curriculum_num_levels - 1) * 0.5) * self.curriculum_subterrain_size[1]
                for ci in range(self.curriculum_num_levels)
            ]
            self.terrain_centers_x = torch.tensor(xs, dtype=gs.tc_float, device=gs.device)
            self.terrain_centers_y = torch.tensor(ys, dtype=gs.tc_float, device=gs.device)
            self.env_terrain_type = torch.randint(
                low=0,
                high=self.curriculum_num_types,
                size=(self.num_envs,),
                dtype=gs.tc_int,
                device=gs.device,
            )
            self.env_terrain_level = torch.zeros(
                (self.num_envs,), dtype=gs.tc_int, device=gs.device
            )
            self.env_start_pos = torch.zeros(
                (self.num_envs, 3), dtype=gs.tc_float, device=gs.device
            )
            self.env_traverse_dir = torch.ones(
                (self.num_envs,), dtype=gs.tc_float, device=gs.device
            )

        self.last_reset_masks = {
            "time_out": torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device),
            "roll_fall": torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device),
            "pitch_fall": torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device),
            "solver_err": torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device),
            "completed_track": torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device),
        }
        self.current_completed_track = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.current_timeout_incomplete = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), dtype=gs.tc_float, device=gs.device
            )

        if self.enable_debug_print:
            print(
                "[Go2EnvDR][debug] "
                f"use_terrain={self.use_terrain} "
                f"horizontal_scale={self.horizontal_scale} "
                f"spawn={self.init_base_pos.tolist()}"
            )
            if self.terrain_layout is not None:
                center_i = len(self.terrain_layout) // 2
                center_j = len(self.terrain_layout[0]) // 2
                print(
                    "[Go2EnvDR][debug] "
                    f"center_tile={self.terrain_layout[center_i][center_j]}"
                )

        self.reset()

    def _get_centered_terrain_origin(self, n_rows, n_cols, subterrain_size=None):
        if subterrain_size is None:
            subterrain_size = self.curriculum_subterrain_size
        total_x = float(n_rows) * float(subterrain_size[0])
        total_y = float(n_cols) * float(subterrain_size[1])
        return (-0.5 * total_x, -0.5 * total_y, 0.0)

    def _build_centered_slope_tile(self, sub_rows, sub_cols, slope_deg, vertical_scale):
        tile = np.zeros((sub_rows, sub_cols), dtype=np.float32)

        plateau_len_m = float(self.terrain_curriculum_cfg.get("slope_plateau_length", 0.8))
        ramp_len_m = float(self.terrain_curriculum_cfg.get("slope_ramp_length", 1.2))
        mound_width_m = float(self.terrain_curriculum_cfg.get("slope_mound_width", 1.8))
        start_offset_m = float(self.terrain_curriculum_cfg.get("slope_start_offset", 1.2))
        end_margin_m = float(self.terrain_curriculum_cfg.get("slope_end_margin", 1.0))
        peak_h_m = math.tan(math.radians(float(slope_deg))) * ramp_len_m
        peak_h_m = min(
            peak_h_m,
            float(self.terrain_curriculum_cfg.get("slope_track_height_cap", 0.35)),
        )
        peak_raw = max(1, int(round(peak_h_m / vertical_scale)))

        plateau_cells = max(1, int(round(plateau_len_m / self.horizontal_scale)))
        ramp_cells = max(1, int(round(ramp_len_m / self.horizontal_scale)))
        width_cells = max(3, int(round(mound_width_m / self.horizontal_scale)))
        start_offset_cells = max(1, int(round(start_offset_m / self.horizontal_scale)))
        end_margin_cells = max(1, int(round(end_margin_m / self.horizontal_scale)))

        cx = sub_rows // 2
        cy = sub_cols // 2
        total_len = start_offset_cells + 2 * ramp_cells + plateau_cells + end_margin_cells
        x1 = max(0, cx - total_len // 2)
        x2 = min(sub_rows, x1 + total_len)
        y1 = max(0, cy - width_cells // 2)
        y2 = min(sub_cols, y1 + width_cells)

        ramp_start = min(x2, x1 + start_offset_cells)
        up_end = min(x2, ramp_start + ramp_cells)
        flat_end = min(x2, up_end + plateau_cells)

        if up_end > ramp_start:
            tile[ramp_start:up_end, y1:y2] = np.linspace(
                0.0,
                float(peak_raw),
                up_end - ramp_start,
                endpoint=True,
                dtype=np.float32,
            )[:, None]
        if flat_end > up_end:
            tile[up_end:flat_end, y1:y2] = float(peak_raw)
        if x2 > flat_end:
            tile[flat_end:x2, y1:y2] = np.linspace(
                float(peak_raw),
                0.0,
                x2 - flat_end,
                endpoint=True,
                dtype=np.float32,
            )[:, None]

        return tile

    def _build_centered_stairs_tile(self, sub_rows, sub_cols, step_height_m, step_count, vertical_scale):
        tile = np.zeros((sub_rows, sub_cols), dtype=np.float32)

        step_count = max(1, int(step_count))
        step_h_raw = max(1, int(round(float(step_height_m) / vertical_scale)))
        step_width_m = float(self.terrain_curriculum_cfg.get("stairs_step_width", 0.3))
        mound_width_m = float(self.terrain_curriculum_cfg.get("stairs_mound_width", 1.8))
        start_offset_m = float(self.terrain_curriculum_cfg.get("stairs_start_offset", 1.2))
        end_margin_m = float(self.terrain_curriculum_cfg.get("stairs_end_margin", 0.6))

        step_width_cells = max(1, int(round(step_width_m / self.horizontal_scale)))
        width_cells = max(3, int(round(mound_width_m / self.horizontal_scale)))
        start_offset_cells = max(1, int(round(start_offset_m / self.horizontal_scale)))
        end_margin_cells = max(1, int(round(end_margin_m / self.horizontal_scale)))
        total_len = start_offset_cells + step_count * step_width_cells + end_margin_cells

        cx = sub_rows // 2
        cy = sub_cols // 2
        x1 = max(0, cx - total_len // 2)
        x2 = min(sub_rows, x1 + total_len)
        y1 = max(0, cy - width_cells // 2)
        y2 = min(sub_cols, y1 + width_cells)

        cur = min(x2, x1 + start_offset_cells)
        for level in range(1, step_count + 1):
            nxt = min(x2, cur + step_width_cells)
            tile[cur:nxt, y1:y2] = float(level * step_h_raw)
            cur = nxt

        return tile

    def _sample_curriculum_spawn_xy(self, idx):
        center_x = self.terrain_centers_x[self.env_terrain_type[idx]]
        center_y = self.terrain_centers_y[self.env_terrain_level[idx]]
        size_x = float(self.curriculum_subterrain_size[0])
        size_y = float(self.curriculum_subterrain_size[1])

        mode = self.curriculum_spawn_mode
        if mode == "random_tile":
            jitter_x = torch.empty_like(center_x).uniform_(
                -self.curriculum_spawn_random_x_ratio * size_x,
                self.curriculum_spawn_random_x_ratio * size_x,
            )
            jitter_y = torch.empty_like(center_y).uniform_(
                -self.curriculum_spawn_random_y_ratio * size_y,
                self.curriculum_spawn_random_y_ratio * size_y,
            )
        elif mode == "track_start":
            jitter_x = torch.empty_like(center_x).uniform_(-0.03, 0.03)
            jitter_x += self.curriculum_spawn_track_start_x_ratio * size_x
            jitter_y = torch.empty_like(center_y).uniform_(
                -self.curriculum_spawn_track_y_ratio * size_y,
                self.curriculum_spawn_track_y_ratio * size_y,
            )
        else:
            jitter_x = torch.empty_like(center_x).uniform_(
                -self.curriculum_spawn_center_jitter,
                self.curriculum_spawn_center_jitter,
            )
            jitter_y = torch.empty_like(center_y).uniform_(
                -self.curriculum_spawn_center_jitter,
                self.curriculum_spawn_center_jitter,
            )

        return center_x + jitter_x, center_y + jitter_y

    def _sync_sim_state_buffers(self):
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

    def _get_curriculum_progress(self, idx=None):
        if idx is None:
            idx = slice(None)

        traverse_dir = self.env_traverse_dir[idx]
        forward_progress = torch.clamp(
            (self.base_pos[idx, 0] - self.env_start_pos[idx, 0]) * traverse_dir,
            min=0.0,
        )
        edge_world_x = (
            self.terrain_centers_x[self.env_terrain_type[idx]]
            + traverse_dir * self.curriculum_edge_reset_ratio * self.curriculum_subterrain_size[0]
        )
        boundary_reached = (
            (self.base_pos[idx, 0] - edge_world_x) * traverse_dir
        ) >= 0.0
        return forward_progress, boundary_reached

    def _turn_around_at_edge(self, turn_mask):
        idx = torch.where(turn_mask)[0]
        if idx.numel() == 0:
            return

        qpos_batch = self.robot.get_qpos(envs_idx=idx).clone()
        old_dir = self.env_traverse_dir[idx].clone()
        edge_world_x = (
            self.terrain_centers_x[self.env_terrain_type[idx]]
            + old_dir * self.curriculum_edge_reset_ratio * self.curriculum_subterrain_size[0]
        )

        base_rpy = quat_to_xyz(qpos_batch[:, 3:7], rpy=True, degrees=False)
        base_rpy[:, 2] += math.pi
        qpos_batch[:, 3:7] = xyz_to_quat(base_rpy, rpy=True)
        qpos_batch[:, 0] = edge_world_x - old_dir * self.curriculum_turnaround_inset

        self.robot.set_qpos(
            qpos_batch,
            envs_idx=idx,
            zero_velocity=True,
            skip_forward=True,
        )

        self.env_traverse_dir[idx] = -old_dir
        self.env_start_pos[idx] = qpos_batch[:, :3]
        self.base_pos[idx] = qpos_batch[:, :3]
        self.base_quat[idx] = qpos_batch[:, 3:7]

    def _build_curriculum_height_field(self):
        """
        构建“地形类型 x 难度等级”高度场：
        - 随机崎岖：高度幅度最高 ±0.1m
        - 斜坡：0° -> 25°
        - 台阶：高度 5cm -> 20cm，台阶宽度 0.3m
        - 障碍物：高度波动 5cm -> 20cm

        返回值是 Genesis/IsaacGym 约定的 height_field_raw（单位为 vertical_scale 的离散格）。
        """
        n_rows = self.curriculum_num_types
        n_cols = self.curriculum_num_levels

        vertical_scale = 0.005
        cache_key = {
            "terrain_types": self.curriculum_type_names,
            "num_levels": self.curriculum_num_levels,
            "subterrain_size": list(self.curriculum_subterrain_size),
            "horizontal_scale": float(self.horizontal_scale),
            "vertical_scale": vertical_scale,
            "rough_height_amplitude": float(
                self.terrain_curriculum_cfg.get("rough_height_amplitude", 0.1)
            ),
            "slope_deg_max": float(self.terrain_curriculum_cfg.get("slope_deg_max", 25.0)),
            "stairs_step_width": float(self.terrain_curriculum_cfg.get("stairs_step_width", 0.3)),
            "stairs_step_height_max": float(
                self.terrain_curriculum_cfg.get("stairs_step_height_max", 0.2)
            ),
            "slope_deg_levels": self.slope_deg_levels_cfg,
            "stairs_step_height_levels": self.stairs_step_height_levels_cfg,
            "stairs_num_steps_levels": self.stairs_num_steps_levels_cfg,
            "obstacle_height_max": float(self.terrain_curriculum_cfg.get("obstacle_height_max", 0.2)),
            "slope_plateau_length": float(self.terrain_curriculum_cfg.get("slope_plateau_length", 0.8)),
            "slope_ramp_length": float(self.terrain_curriculum_cfg.get("slope_ramp_length", 1.0)),
            "slope_mound_width": float(self.terrain_curriculum_cfg.get("slope_mound_width", 1.8)),
            "slope_start_offset": float(self.terrain_curriculum_cfg.get("slope_start_offset", 1.2)),
            "slope_end_margin": float(self.terrain_curriculum_cfg.get("slope_end_margin", 1.0)),
            "stairs_plateau_length": float(self.terrain_curriculum_cfg.get("stairs_plateau_length", 0.6)),
            "stairs_mound_width": float(self.terrain_curriculum_cfg.get("stairs_mound_width", 1.8)),
            "stairs_start_offset": float(self.terrain_curriculum_cfg.get("stairs_start_offset", 1.0)),
            "stairs_end_margin": float(self.terrain_curriculum_cfg.get("stairs_end_margin", 0.6)),
            "generator": "curriculum_centered_blocks_v5",
        }
        cache_hash = hashlib.md5(
            json.dumps(cache_key, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = os.path.join(self.curriculum_cache_dir, f"curriculum_hf_{cache_hash}.npy")

        if self.curriculum_cache_enabled and os.path.isfile(cache_path):
            load_start = time.perf_counter()
            hf_cached = np.load(cache_path)
            print(
                f"[Go2EnvDR] 课程地形 height_field 命中缓存: {cache_path} "
                f"(用时 {time.perf_counter() - load_start:.2f}s)"
            )
            return hf_cached

        gen_start = time.perf_counter()
        print("[Go2EnvDR] 课程地形 height_field 生成中（official terrain_utils）...")

        sub_rows = int(self.curriculum_subterrain_size[0] / self.horizontal_scale + gs.EPS) + 1
        sub_cols = int(self.curriculum_subterrain_size[1] / self.horizontal_scale + gs.EPS) + 1
        hf = np.zeros(
            (n_rows * (sub_rows - 1) + 1, n_cols * (sub_cols - 1) + 1),
            dtype=np.float32,
        )

        rough_amp = float(self.terrain_curriculum_cfg.get("rough_height_amplitude", 0.1))
        slope_deg_max = float(self.terrain_curriculum_cfg.get("slope_deg_max", 25.0))
        stairs_step_width_m = float(self.terrain_curriculum_cfg.get("stairs_step_width", 0.3))
        stairs_step_height_max = float(self.terrain_curriculum_cfg.get("stairs_step_height_max", 0.2))
        obstacle_height_max = float(self.terrain_curriculum_cfg.get("obstacle_height_max", 0.2))
        platform_size_m = float(self.terrain_curriculum_cfg.get("platform_size", 1.0))

        slope_levels_deg = np.linspace(0.0, slope_deg_max, n_cols)
        if self.slope_deg_levels_cfg is not None and len(self.slope_deg_levels_cfg) > 0:
            slope_levels_deg = np.array(self.slope_deg_levels_cfg, dtype=np.float32)

        stairs_height_levels_m = np.linspace(0.05, stairs_step_height_max, n_cols)
        if self.stairs_step_height_levels_cfg is not None and len(self.stairs_step_height_levels_cfg) > 0:
            stairs_height_levels_m = np.array(self.stairs_step_height_levels_cfg, dtype=np.float32)

        obstacle_levels_m = np.linspace(0.05, obstacle_height_max, n_cols)

        stairs_num_steps_levels = None
        if self.stairs_num_steps_levels_cfg is not None and len(self.stairs_num_steps_levels_cfg) > 0:
            stairs_num_steps_levels = np.array(self.stairs_num_steps_levels_cfg, dtype=np.int32)

        if len(slope_levels_deg) != n_cols:
            xp = np.linspace(0.0, 1.0, len(slope_levels_deg), dtype=np.float32)
            xq = np.linspace(0.0, 1.0, n_cols, dtype=np.float32)
            slope_levels_deg = np.interp(xq, xp, slope_levels_deg).astype(np.float32)

        if len(stairs_height_levels_m) != n_cols:
            xp = np.linspace(0.0, 1.0, len(stairs_height_levels_m), dtype=np.float32)
            xq = np.linspace(0.0, 1.0, n_cols, dtype=np.float32)
            stairs_height_levels_m = np.interp(xq, xp, stairs_height_levels_m).astype(np.float32)

        if stairs_num_steps_levels is not None and len(stairs_num_steps_levels) != n_cols:
            xp = np.linspace(0.0, 1.0, len(stairs_num_steps_levels), dtype=np.float32)
            xq = np.linspace(0.0, 1.0, n_cols, dtype=np.float32)
            stairs_num_steps_levels = np.round(np.interp(xq, xp, stairs_num_steps_levels)).astype(np.int32)

        # 随机崎岖地形也按等级递进：从较平缓起步，最高达到 ±0.1m。
        rough_levels_m = np.linspace(min(0.02, rough_amp), rough_amp, n_cols)

        for ri, terrain_type in enumerate(self.curriculum_type_names):
            for ci in range(n_cols):
                subterrain = isaacgym_terrain_utils.SubTerrain(
                    width=sub_rows,
                    length=sub_cols,
                    vertical_scale=vertical_scale,
                    horizontal_scale=self.horizontal_scale,
                )

                np.random.seed(0)
                if terrain_type == "random_uniform_terrain":
                    rough_h = float(rough_levels_m[ci])
                    step_h = max(vertical_scale, round(rough_h / 8.0, 4))
                    subterrain = isaacgym_terrain_utils.random_uniform_terrain(
                        subterrain,
                        min_height=-rough_h,
                        max_height=rough_h,
                        step=step_h,
                        downsampled_scale=max(self.horizontal_scale, 0.5),
                    )

                elif terrain_type == "pyramid_sloped_terrain":
                    slope = math.tan(math.radians(float(slope_levels_deg[ci])))
                    subterrain = isaacgym_terrain_utils.pyramid_sloped_terrain(
                        subterrain,
                        slope=slope,
                        platform_size=platform_size_m,
                    )

                elif terrain_type == "slope_track":
                    subterrain.height_field_raw[:, :] = self._build_centered_slope_tile(
                        sub_rows,
                        sub_cols,
                        float(slope_levels_deg[ci]),
                        vertical_scale,
                    )

                elif terrain_type == "pyramid_stairs_terrain":
                    step_h = float(stairs_height_levels_m[ci])
                    if stairs_num_steps_levels is not None:
                        num_steps = max(1, int(stairs_num_steps_levels[ci]))
                        margin_cells = max(1, int(round(platform_size_m / self.horizontal_scale)))
                        usable_rows = max(1, sub_rows - 2 * margin_cells)
                        step_cells = max(1, usable_rows // num_steps)
                        stair_idx = np.zeros((sub_rows,), dtype=np.int32)
                        for sr in range(sub_rows):
                            if sr <= margin_cells:
                                stair_idx[sr] = 0
                            elif sr >= sub_rows - margin_cells:
                                stair_idx[sr] = num_steps
                            else:
                                stair_idx[sr] = min(num_steps, max(0, (sr - margin_cells) // step_cells))
                        subterrain.height_field_raw[:, :] = (
                            stair_idx[:, None] * int(round(step_h / vertical_scale))
                        )
                        center_half = max(1, int(round(0.5 * platform_size_m / self.horizontal_scale)))
                        cx = sub_rows // 2
                        cy = sub_cols // 2
                        x1 = max(0, cx - center_half)
                        x2 = min(sub_rows, cx + center_half)
                        y1 = max(0, cy - center_half)
                        y2 = min(sub_cols, cy + center_half)
                        subterrain.height_field_raw[x1:x2, y1:y2] = 0
                    else:
                        subterrain = isaacgym_terrain_utils.pyramid_stairs_terrain(
                            subterrain,
                            step_width=stairs_step_width_m,
                            step_height=step_h,
                            platform_size=platform_size_m,
                        )

                elif terrain_type == "stairs_track":
                    num_steps = (
                        max(1, int(stairs_num_steps_levels[ci]))
                        if stairs_num_steps_levels is not None
                        else max(1, int(round(float(self.curriculum_subterrain_size[0]) / max(stairs_step_width_m, self.horizontal_scale))))
                    )
                    subterrain.height_field_raw[:, :] = self._build_centered_stairs_tile(
                        sub_rows,
                        sub_cols,
                        float(stairs_height_levels_m[ci]),
                        num_steps,
                        vertical_scale,
                    )

                elif terrain_type == "discrete_obstacles_terrain":
                    max_h = float(obstacle_levels_m[ci])
                    subterrain = isaacgym_terrain_utils.discrete_obstacles_terrain(
                        subterrain,
                        max_height=max_h,
                        min_size=0.2,
                        max_size=0.8,
                        num_rects=24,
                        platform_size=platform_size_m,
                    )

                elif terrain_type == "flat_terrain":
                    pass

                else:
                    raise ValueError(f"Unsupported terrain type in curriculum: {terrain_type}")

                tile_raw = np.asarray(subterrain.height_field_raw, dtype=np.float32)
                x0 = ri * (sub_rows - 1)
                y0 = ci * (sub_cols - 1)
                hf[x0 : x0 + sub_rows, y0 : y0 + sub_cols] = tile_raw

        if self.curriculum_cache_enabled:
            os.makedirs(self.curriculum_cache_dir, exist_ok=True)
            np.save(cache_path, hf)
            print(
                f"[Go2EnvDR] 课程地形 height_field 已缓存: {cache_path} "
                f"(生成用时 {time.perf_counter() - gen_start:.2f}s)"
            )
        else:
            print(f"[Go2EnvDR] 课程地形 height_field 生成完成, 用时 {time.perf_counter() - gen_start:.2f}s")

        return hf

    def _randomize_physics(self, envs_idx):
        if not self.enable_dr:
            return

        idx = (
            torch.where(envs_idx)[0]
            if envs_idx is not None
            else torch.arange(self.num_envs, device=gs.device)
        )
        if idx.numel() == 0:
            return

        n = idx.numel()

        fr_lo = self.dr_cfg.get("friction_ratio_range", [0.5, 1.5])[0]
        fr_hi = self.dr_cfg.get("friction_ratio_range", [0.5, 1.5])[1]
        num_links = len(self.robot.links)
        friction_ratios = torch.empty(
            n, num_links, dtype=gs.tc_float, device=gs.device
        ).uniform_(fr_lo, fr_hi)
        self.robot.set_friction_ratio(friction_ratios, envs_idx=idx)

        # 该接口在当前 Genesis 版本不支持批量 envs_idx，逐 env×逐关节调用极慢。
        # 默认关闭以避免“环境创建卡很久”；需要时可在 dr_cfg 显式开启。
        if self.dr_cfg.get("enable_dof_frictionloss_randomization", False):
            df_lo = self.dr_cfg.get("dof_frictionloss_range", [0.0, 0.1])[0]
            df_hi = self.dr_cfg.get("dof_frictionloss_range", [0.0, 0.1])[1]
            dof_fl_all = torch.empty(
                n, len(self.motors_dof_idx), dtype=gs.tc_float, device=gs.device
            ).uniform_(df_lo, df_hi)

            for env_offset, env_idx in enumerate(idx):
                env_idx_t = torch.tensor([env_idx.item()], dtype=gs.tc_int, device=gs.device)
                for joint_offset, dof_idx in enumerate(self.motors_dof_idx):
                    self.robot.set_dofs_frictionloss(
                        dof_fl_all[env_offset, joint_offset : joint_offset + 1],
                        dof_idx,
                        envs_idx=env_idx_t,
                    )

    def _apply_push(self):
        if not self.enable_dr:
            return
        interval = self.dr_cfg.get("push_interval_steps", 250)
        if interval <= 0 or self._step_count % interval != 0:
            return

        vmax = self.dr_cfg.get("push_vel_max", 0.5)
        cur_vel = self.robot.get_dofs_velocity()
        cur_vel[:, 0] += torch.empty(
            self.num_envs, dtype=gs.tc_float, device=gs.device
        ).uniform_(-vmax, vmax)
        cur_vel[:, 1] += torch.empty(
            self.num_envs, dtype=gs.tc_float, device=gs.device
        ).uniform_(-vmax, vmax)
        self.robot.set_dofs_velocity(cur_vel)

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _compute_reset_masks(self):
        time_out = self.episode_length_buf > self.max_episode_length
        roll_fall = (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )
        pitch_fall = (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        solver_err = self.scene.rigid_solver.get_error_envs_mask()
        return time_out, roll_fall, pitch_fall, solver_err

    def _record_reset_debug(self, time_out, roll_fall, pitch_fall, solver_err, completed_track=None):
        if completed_track is None:
            completed_track = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)

        self.last_reset_masks["time_out"] = time_out
        self.last_reset_masks["roll_fall"] = roll_fall
        self.last_reset_masks["pitch_fall"] = pitch_fall
        self.last_reset_masks["solver_err"] = solver_err
        self.last_reset_masks["completed_track"] = completed_track

        self.extras["time_outs"] = time_out.to(dtype=gs.tc_float)
        self.extras["reset_reasons"] = {
            "time_out": time_out.clone(),
            "roll_fall": roll_fall.clone(),
            "pitch_fall": pitch_fall.clone(),
            "solver_err": solver_err.clone(),
            "completed_track": completed_track.clone(),
        }
        self.extras["reset_counts"] = {
            "all": int((time_out | roll_fall | pitch_fall | solver_err | completed_track).sum().item()),
            "time_out": int(time_out.sum().item()),
            "roll_fall": int(roll_fall.sum().item()),
            "pitch_fall": int(pitch_fall.sum().item()),
            "solver_err": int(solver_err.sum().item()),
            "completed_track": int(completed_track.sum().item()),
        }
        self.extras["debug_stats"] = {
            "mean_abs_roll_deg": float(torch.abs(self.base_euler[:, 0]).mean().item()),
            "mean_abs_pitch_deg": float(torch.abs(self.base_euler[:, 1]).mean().item()),
            "mean_base_height": float(self.base_pos[:, 2].mean().item()),
            "mean_forward_vel": float(self.base_lin_vel[:, 0].mean().item()),
            "mean_ep_len": float(self.episode_length_buf.float().mean().item()),
        }
        if self.terrain_curriculum_enabled:
            self.extras["debug_stats"]["mean_curr_level"] = float(
                self.env_terrain_level.float().mean().item()
            )

        if self.enable_debug_print and self._step_count % self.debug_print_interval == 0:
            print(
                "[Go2EnvDR][debug] "
                f"step={self._step_count} "
                f"ep_len_mean={self.extras['debug_stats']['mean_ep_len']:.1f} "
                f"roll_abs_mean={self.extras['debug_stats']['mean_abs_roll_deg']:.2f} "
                f"pitch_abs_mean={self.extras['debug_stats']['mean_abs_pitch_deg']:.2f} "
                f"base_z_mean={self.extras['debug_stats']['mean_base_height']:.3f} "
                f"vel_x_mean={self.extras['debug_stats']['mean_forward_vel']:.3f} "
                f"resets={self.extras['reset_counts']['all']} "
                f"(roll={self.extras['reset_counts']['roll_fall']}, "
                f"pitch={self.extras['reset_counts']['pitch_fall']}, "
                f"solver={self.extras['reset_counts']['solver_err']}, "
                f"timeout={self.extras['reset_counts']['time_out']}, "
                f"done_track={self.extras['reset_counts']['completed_track']})"
            )

    def step(self, actions):
        self._step_count += 1

        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))

        self._apply_push()
        self.scene.step()

        self.episode_length_buf += 1
        self.last_base_pos.copy_(self.base_pos)
        self._sync_sim_state_buffers()

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            if name in ("completion", "timeout_incomplete"):
                continue
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._resample_commands(
            self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0
        )

        time_out, roll_fall, pitch_fall, solver_err = self._compute_reset_masks()
        completed_track = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        turnaround_event = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        if self.terrain_curriculum_enabled:
            forward_progress, boundary_reached = self._get_curriculum_progress()
            complete_track_dist = self.curriculum_complete_track_ratio * self.curriculum_subterrain_size[0]
            completed_track = forward_progress >= complete_track_dist

            if self.curriculum_edge_behavior == "turn_around":
                turnaround_event = boundary_reached & ~(
                    time_out | roll_fall | pitch_fall | solver_err | completed_track
                )
                if turnaround_event.any():
                    self._turn_around_at_edge(turnaround_event)
                    self._sync_sim_state_buffers()
            else:
                completed_track = completed_track | boundary_reached

            self.extras["curriculum_forward_progress"] = forward_progress.clone()
            self.extras["curriculum_completed_track"] = completed_track.clone()
            self.extras["curriculum_boundary_reached"] = boundary_reached.clone()
            self.extras["curriculum_turnaround_event"] = turnaround_event.clone()
            self.extras["curriculum_success_event"] = (completed_track | turnaround_event).clone()
            self.extras["curriculum_complete_track_dist"] = float(complete_track_dist)
        self.current_completed_track = completed_track.clone()
        self.current_timeout_incomplete = time_out & ~completed_track

        # completion/timeout 是终止事件，只有算完 reset mask 后才知道；
        # 单独在这里加，避免使用上一帧的终止标志造成奖励错位。
        for name in ("completion", "timeout_incomplete"):
            if name in self.reward_functions:
                rew = self.reward_functions[name]() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        self.reset_buf = time_out | roll_fall | pitch_fall | solver_err | completed_track
        self._record_reset_debug(time_out, roll_fall, pitch_fall, solver_err, completed_track=completed_track)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # 课程地形：在 reset 前根据上一段轨迹表现更新每个 env 的难度等级。
        if self.terrain_curriculum_enabled and envs_idx is not None:
            idx = torch.where(envs_idx)[0]
            if idx.numel() > 0:
                complete_track_dist = (
                    self.curriculum_complete_track_ratio * self.curriculum_subterrain_size[0]
                )
                dt_episode = self.episode_length_buf[idx].float() * self.dt
                expected_dist = torch.abs(self.commands[idx, 0]) * dt_episode
                forward_progress, _ = self._get_curriculum_progress(idx)
                completed_track = forward_progress >= complete_track_dist
                under_target = forward_progress < (self.curriculum_demote_ratio * expected_dist)

                cur_level = self.env_terrain_level[idx].clone()
                max_level = self.curriculum_num_levels - 1

                promote_mask = completed_track
                demote_mask = (~completed_track) & under_target

                next_level = cur_level.clone()
                next_level[promote_mask] = torch.minimum(
                    cur_level[promote_mask] + 1,
                    torch.full_like(cur_level[promote_mask], max_level),
                )
                next_level[demote_mask] = torch.maximum(
                    cur_level[demote_mask] - 1,
                    torch.zeros_like(cur_level[demote_mask]),
                )

                # 循环重置机制：若在最高难度成功，则重采样到随机难度并随机地形类型。
                if self.curriculum_randomize_max:
                    max_success = promote_mask & (cur_level >= max_level)
                    if max_success.any():
                        count = int(max_success.sum().item())
                        next_level[max_success] = torch.randint(
                            low=0,
                            high=self.curriculum_num_levels,
                            size=(count,),
                            dtype=gs.tc_int,
                            device=gs.device,
                        )
                        self.env_terrain_type[idx[max_success]] = torch.randint(
                            low=0,
                            high=self.curriculum_num_types,
                            size=(count,),
                            dtype=gs.tc_int,
                            device=gs.device,
                        )

                self.env_terrain_level[idx] = next_level
                self.extras["curriculum_progress"] = {
                    "forward_progress_mean": float(forward_progress.mean().item()),
                    "completed_track_count": int(completed_track.sum().item()),
                }

        spawn_idx = None
        spawn_x = None
        spawn_y = None

        if self.terrain_curriculum_enabled:
            # 根据每个 env 的(地形类型, 难度等级)决定 reset 出生点。
            if envs_idx is None:
                idx = torch.arange(self.num_envs, device=gs.device)
            else:
                idx = torch.where(envs_idx)[0]

            if idx.numel() > 0:
                spawn_idx = idx
                spawn_x, spawn_y = self._sample_curriculum_spawn_xy(idx)
                qpos_batch = self.init_qpos.repeat(idx.numel(), 1)
                qpos_batch[:, 0] = spawn_x
                qpos_batch[:, 1] = spawn_y
                qpos_batch[:, 2] = self.init_base_pos[2] + 0.5

                self.robot.set_qpos(
                    qpos_batch,
                    envs_idx=idx,
                    zero_velocity=True,
                    skip_forward=True,
                )
        else:
            self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            if self.terrain_curriculum_enabled:
                if spawn_x is None or spawn_y is None:
                    all_idx = torch.arange(self.num_envs, device=gs.device)
                    spawn_x, spawn_y = self._sample_curriculum_spawn_xy(all_idx)
                self.base_pos[:, 0] = spawn_x
                self.base_pos[:, 1] = spawn_y
                self.base_pos[:, 2] = self.init_base_pos[2]
            else:
                self.base_pos.copy_(self.init_base_pos)
            self.last_base_pos.copy_(self.base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            if self.env_traverse_dir is not None:
                self.env_traverse_dir.fill_(1.0)
        else:
            if self.terrain_curriculum_enabled:
                idx = torch.where(envs_idx)[0]
                if idx.numel() > 0:
                    if spawn_x is None or spawn_y is None:
                        spawn_x, spawn_y = self._sample_curriculum_spawn_xy(idx)
                    self.base_pos[idx, 0] = spawn_x
                    self.base_pos[idx, 1] = spawn_y
                    self.base_pos[idx, 2] = self.init_base_pos[2]
            else:
                torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.base_pos, self.last_base_pos, out=self.last_base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None],
                self.init_projected_gravity,
                self.projected_gravity,
                out=self.projected_gravity,
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            if self.env_traverse_dir is not None:
                self.env_traverse_dir.masked_fill_(envs_idx, 1.0)

        self._randomize_physics(envs_idx)

        if self.terrain_curriculum_enabled:
            # 记录新 episode 起点，用于下次 reset 时计算“穿行距离”并升降级。
            if envs_idx is None:
                self.env_start_pos.copy_(self.base_pos)
            else:
                if spawn_idx is not None and spawn_idx.numel() > 0:
                    self.env_start_pos[spawn_idx] = self.base_pos[spawn_idx]

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_orientation(self):
        # projected_gravity 在机器人坐标系下，xy 分量越大代表机身越倾斜。
        # 惩罚该项可直接鼓励“保持机身水平”，比 world-z 高度约束更适合 rough terrain。
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_forward_progress(self):
        if not self.terrain_curriculum_enabled or self.env_traverse_dir is None:
            return torch.clamp(self.base_lin_vel[:, 0], min=0.0)
        progress_delta = (self.base_pos[:, 0] - self.last_base_pos[:, 0]) * self.env_traverse_dir
        return torch.clamp(progress_delta / self.dt, min=0.0, max=1.0)

    def _reward_uphill_progress(self):
        # 上一轮在 12deg+ 主要是坡底 timeout；只奖励坡道前半段推进，
        # 让策略把学习压力放在“真正上坡”而不是坡前平地小步移动。
        if not self.terrain_curriculum_enabled or self.env_traverse_dir is None:
            return torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)

        forward_progress, _ = self._get_curriculum_progress()
        uphill_start = max(0.0, self.slope_start_offset_m - 0.25)
        uphill_end = self.slope_start_offset_m + self.slope_ramp_length_m + 0.25
        on_uphill = (forward_progress >= uphill_start) & (forward_progress <= uphill_end)
        progress_delta = (self.base_pos[:, 0] - self.last_base_pos[:, 0]) * self.env_traverse_dir
        return torch.clamp(progress_delta / self.dt, min=0.0, max=1.0) * on_uphill.to(gs.tc_float)

    def _reward_stall(self):
        forward_cmd = torch.abs(self.commands[:, 0])
        moving_too_slow = torch.clamp(0.6 * forward_cmd - self.base_lin_vel[:, 0], min=0.0)
        return moving_too_slow

    def _reward_completion(self):
        # 完整通过坡道要比只在坡底获得少量 tracking/progress 更划算。
        return self.current_completed_track.to(gs.tc_float)

    def _reward_timeout_incomplete(self):
        # 高坡失败多为 timeout，不加该项时策略可以在坡底拖到回合结束。
        return self.current_timeout_incomplete.to(gs.tc_float)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_alive(self):
        return torch.ones(self.num_envs, dtype=gs.tc_float, device=gs.device)
