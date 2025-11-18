import genesis as gs
import torch
import numpy as np
import mediapy
from scipy.spatial.transform import Rotation as R
import os
from collections import defaultdict
from mushroom_rl.core import MDPInfo
from mushroom_rl.rl_utils.spaces import Box

from gd.traj_optim_cmaes import (
    create_linear_array,
    create_exp_array,
    create_custom_array,
    TrajOptimCMAES
)


class Train_Env():
    def __init__(self, task, scene=None, GUI=False, camera=False, log_dir=None, n_envs=None, requires_grad=False, scene_version=None):
        self.task = task
        self.GUI = GUI
        self.n_envs = n_envs
        self.requires_grad = requires_grad
        print(f"GUI: {self.GUI}, n_envs: {self.n_envs}, requires_grad: {self.requires_grad}")

        if scene is None:
            # Initialize scene, if not provided
            gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)

            viewer_options = gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            )

            self.scene = gs.Scene(
                viewer_options=viewer_options,
                sim_options=gs.options.SimOptions(
                    dt=1e-3,
                    substeps=5,
                    requires_grad=requires_grad,
                ),
                rod_options=gs.options.RodOptions(
                    damping=15.0,
                    angular_damping=10.0,
                    n_pbd_iters=20,
                ),
                show_viewer=self.GUI,
            )
        else:
            # Use provided scene, this means genesis already initialized
            self.scene = scene

        self.create_log_dir(log_dir)

        self.cameras = list()
        self.frames = defaultdict(list)
        if scene_version is None:
            scene_version = 1
        if scene_version == 1:
            self.construct_scene(camera=camera)
        elif scene_version == 2:
            self.construct_scene_v2(camera=camera)
        else:
            raise ValueError(f"Unknown scene_version: {scene_version}")
        self.scene_version = scene_version
        print(f'Scene version set to: {scene_version}')
        self.control_dist_init = None
        self.debug_point_nodes = list()

    def construct_traj_optim(self, max_ddist=0.1, max_grad_norm=1000, debug=False):
        if not self.requires_grad:
            return

        self.c = TrajOptimCMAES(
            scene=self.scene,
            rod=self.rope,
            grasp_point_ids=self.control_idx,
            n_optim_dofs=3,
            max_ddist=max_ddist,
            max_grad_norm=max_grad_norm,
            debug=debug,
        )

    def construct_scale_array(self, scale_method, n_steps, exp_base=1.1):
        if scale_method is None:
            scale_array = torch.ones(n_steps, dtype=gs.tc_float)
            self.scale_array = scale_array / n_steps
            print(f'Using uniform scale array:\n{self.scale_array}')
        elif scale_method == 'linear':
            self.scale_array = create_linear_array(n_steps)
            print(f'Using linear scale array:\n{self.scale_array}')
        elif scale_method == 'exp':
            self.scale_array = create_exp_array(n_steps, base=exp_base)
            print(f'Using exponential scale array (base={exp_base}):\n{self.scale_array}')
        elif scale_method == 'custom':
            self.scale_array = create_custom_array(n_steps)
            print(f'Using custom scale array:\n{self.scale_array}')
        else:
            raise ValueError(f'Unknown scale method: {scale_method}')

    def create_log_dir(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)

    def init_mass(self, mass=0.015):
        for entity in self.scene.sim.rigid_solver.entities[2:]:
            for link in entity.links:
                link._inertial_mass = mass

    def construct_scene(self, camera=False):
        raise NotImplementedError()

    def construct_scene_v2(self, camera=False):
        raise NotImplementedError()

    def construct_cameras(self):
        raise NotImplementedError()

    def reward(self):
        raise NotImplementedError()

    def loss_criterion(self, state):
        raise NotImplementedError()

    def reset(self, debug=False, envs_idx=None):
        raise NotImplementedError()

    # # # # # # RL Utils # # # # # #
    def compute_observation(self):
        raise NotImplementedError()

    # # # # # # RL Utils # # # # # #
    def reset_all(self, env_mask, state=None):
        env_indices = torch.where(env_mask)[0]
        # Check if there are any environments to reset
        if len(env_indices) > 0:
            self.reset(envs_idx=env_indices)

        obs = self.compute_observation()
        return obs, [{}] * self.n_envs

    # # # # # # RL Utils # # # # # #
    def init_rl_env(
        self,
        n_steps = 10,
        pos_bound = 0.1,
        angle_bound = 5.0,
        n_rigid_obs=0,
        debug=False
    ):
        self._l2_limit = pos_bound
        # RL/vectorized env configuration
        self._backend = 'torch'
        if debug:
            # TODO: hack here
            if getattr(self, "c1", None) is not None:
                self.c1.debug = True
            if getattr(self, "c2", None) is not None:
                self.c2.debug = True

        # Observation / action specs
        self._obs_dim = (self.rope.n_vertices + n_rigid_obs) * 3    # (n_vertices + n_rigid_obs) * 3
        self._act_dim = len(self.control_idx) * 6                   # n_ctrl * 6
        self._horizon = n_steps                                     # 10
        self._steps_per_action = self.steps_interval

        # Observation/action spaces
        low_obs = torch.full((self._obs_dim,), -np.inf, dtype=torch.float32)
        high_obs = torch.full((self._obs_dim,), np.inf, dtype=torch.float32)
        observation_space = Box(low_obs, high_obs)

        act_limit = [pos_bound] * (self._act_dim // 2) + [angle_bound] * (self._act_dim // 2)
        act_limit = torch.tensor(act_limit, dtype=torch.float32)
        print(f'Bound: {act_limit}')
        low_act = -torch.ones((self._act_dim,), dtype=torch.float32) * act_limit
        high_act = torch.ones((self._act_dim,), dtype=torch.float32) * act_limit
        action_space = Box(low_act, high_act)

        # Control dt approximates sim dt * internal steps
        control_dt = self.scene.sim_options.dt * self._steps_per_action
        self._mdp_info = MDPInfo(observation_space, action_space, gamma=0.99, horizon=self._horizon, dt=control_dt, backend=self._backend)

        # Compatibility in `reset()`
        self.use_qpos = True

    @property
    def info(self):
        return self._mdp_info

    @property
    def number(self):
        return self.n_envs

    def save_animation(self, save_dir):
        if len(self.frames) == 0:
            return

        for cid in self.frames:
            video_path = os.path.join(save_dir, f"view_{cid}_best_debug.mp4")
            mediapy.write_video(video_path, self.frames[cid], fps=30, qp=18)
            print(f'Saved video to {video_path}')

    def loss_above_plane(self, state):
        # Required loss to make sure the vertices above the plane
        verts_batch = state.pos
        loss_abv_plane = torch.relu(
            self.rope.material.segment_radius - verts_batch[:, :, 2]
        ).sum(dim=1)                    # (n_envs,)

        return loss_abv_plane

    # # # # # # RL Utils # # # # # #
    def render_all(self, env_mask, record=False):
        if self.GUI:
            pass
        if record:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return None

    # # # # # # RL Utils # # # # # #
    def stop(self):
        pass

    def eval_traj(self, trajs, debug=False, **kwargs):
        if self.scene_version == 1:
            return self.eval_traj_v1(trajs, debug=debug)
        elif self.scene_version == 2:
            return self.eval_traj_v2(trajs, debug=debug, **kwargs)

    def eval_traj_v1(self, trajs, debug=False):
        """
        Evaluate trajectories.

        Rewards:
        - If an env survives all micro-steps: reward = self.reward()[env].
        - If an env COLLIDES or gets NaNs in verts: reward = survival_time / total_micro_steps.
        - If env reward is NaN at the end: reward = -100.

        Survival time counts micro-steps from 0..N, where N = n_steps * steps_interval.
        """
        import numpy as np

        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 3 == 0 and dof // 3 == n_ctrl, (
            f"dof must be 3 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        self.reset()

        steps_interval = self.steps_interval
        total_micro_steps = int(n_steps * steps_interval)
        if total_micro_steps <= 0:
            # Degenerate case: no steps → everyone "survives"; defer to env reward (or -100 if NaN)
            rewards = np.asarray(self.reward(), dtype=np.float32)
            rewards[np.isnan(rewards)] = -100.0
            return rewards.astype(np.float32)

        # Per-env status
        alive = np.ones((self.n_envs,), dtype=bool)              # True until first failure (collision or NaN)
        ever_nan = np.zeros((self.n_envs,), dtype=bool)          # True if verts ever became NaN
        ever_collided = np.zeros((self.n_envs,), dtype=bool)     # True if collision occurred
        ever_stretched = np.zeros((self.n_envs,), dtype=bool)    # True if stretching failure occurred
        first_fail_step = np.full((self.n_envs,), total_micro_steps, dtype=np.int32)  # micro-step index of first failure

        for i in range(n_steps):
            # Check NaNs BEFORE micro-stepping this macro-step
            verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now = np.isnan(verts_rope).any(axis=(1, 2))
            newly_nan = nan_now & alive
            if newly_nan.any():
                # Failure occurs before any micro-step of this macro-step
                # Use step = max(1, i*steps_interval) to keep survival count >= 1 if we want strictly positive
                step_at_nan = i * steps_interval
                step_at_nan = max(1, step_at_nan)
                first_fail_step[newly_nan] = step_at_nan
                ever_nan[newly_nan] = True
                alive[newly_nan] = False

            # Early exit if everyone is already NaN
            if ever_nan.all():
                break

            # If no env is alive anymore, we can stop
            if not alive.any():
                break

            # Prepare interpolation to targets for this macro-step
            current_pos = verts_rope[:, self.control_idx]              # (n_envs, n_ctrl, 3)
            delta = trajs[:, i].reshape(self.n_envs, -1, 3)            # (n_envs, n_ctrl, 3)

            if debug:
                debug_pos = current_pos + delta
                debug_pos = debug_pos.copy()
                for batch_idx in range(self.n_envs):
                    offset = self.scene.envs_offset[batch_idx]
                    for ii in self.debug_point_nodes:
                        self.scene.clear_debug_object(ii)
                    self.debug_point_nodes = list()
                    for ii in range(len(self.control_idx)):
                        if debug_pos[batch_idx, ii, 2] < self.rope.material.segment_radius:
                            color = (1.0, 1.0, 0.0, 0.6)
                            debug_pos[batch_idx, ii, 2] = self.rope.material.segment_radius
                        else:
                            color = (0.0, 1.0, 0.0, 0.6)
                        self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                            pos=debug_pos[batch_idx, ii] + offset,
                            radius=0.016,
                            color=color
                        ))

            for j in range(steps_interval):
                if not alive.any():
                    break

                # NOTE: Do not move already-failed envs
                delta[~alive, :, :] = 0.0

                alpha = (j + 1) / steps_interval
                target_pos = current_pos + delta * alpha               # (n_envs, n_ctrl, 3)

                # Apply target positions; if set_pos_single isn't batch-aware, loop envs instead.
                for k in range(n_ctrl):
                    self.rope.set_pos_single(target_pos[:, k], self.control_idx[k])

                self.scene.step()

                if j % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

                # Post-step: detect collisions
                collided = self.rope._solver.vertices_collision.collided.to_numpy()  # (n_verts, n_envs)
                collided = collided.T  # (n_envs, n_vertices)
                verts_to_check = np.array(self.control_idx) + self.rope._v_start
                collided_ctrl = collided[:, verts_to_check].any(axis=1)          # (n_envs,)

                newly_collided = collided_ctrl & alive
                if newly_collided.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_collided] = np.minimum(first_fail_step[newly_collided], global_step)
                    ever_collided[newly_collided] = True
                    alive[newly_collided] = False

                # Post-step: detect stretching failures
                if self.control_dist_init is not None:
                    # (n_envs,)
                    control_dist_now = self.rope.get_geodesic_distance(
                        self.control_idx[0], self.control_idx[1]
                    )
                    # 1% stretch allowed
                    stretched_between_ctrl = control_dist_now / self.control_dist_init > 1.01
                    newly_stretched = stretched_between_ctrl & alive
                    if newly_stretched.any():
                        global_step = i * steps_interval + (j + 1)
                        first_fail_step[newly_stretched] = np.minimum(first_fail_step[newly_stretched], global_step)
                        ever_stretched[newly_stretched] = True
                        alive[newly_stretched] = False

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_nan_after] = np.minimum(first_fail_step[newly_nan_after], global_step)
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

        # Compute base rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)

        # Compose final rewards
        final = np.empty((n_envs,), dtype=np.float32)

        failed = ~alive  # failed due to collision or NaN during rollout
        survived = alive

        # Failed: reward = survival_ratio (counts both collision and NaN cases)
        if failed.any():
            survival_ratio = first_fail_step.astype(np.float32) / float(total_micro_steps)
            final[failed] = survival_ratio[failed] - 100

        # Survived full rollout: take env reward; if it's NaN, clamp to -100
        final[survived] = env_rewards[survived]
        if env_rewards_nan.any():
            final[env_rewards_nan] = -100.0

        return final.astype(np.float32)

    def eval_traj_v2(self, trajs, debug=False, **kwargs):
        raise NotImplementedError()

    def adaptive_scale(self, trajs, deltas, ratio=0.1):
        norm_trajs = np.linalg.norm(trajs, axis=-1, keepdims=True)
        norm_deltas = np.linalg.norm(deltas, axis=-1, keepdims=True)

        norm_deltas = np.clip(norm_deltas, a_min=gs.EPS, a_max=None)
        scaling_factor = (ratio * norm_trajs) / norm_deltas

        deltas_scaled = deltas * scaling_factor

        return deltas_scaled

    def gd_one_step(self, trajs):
        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        trajs = torch.tensor(trajs, dtype=gs.tc_float)
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 3 == 0 and dof // 3 == n_ctrl, (
            f"dof must be 3 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        total_horizon = 0
        horizon_ids = list()

        self.reset()

        loss = 0.
        for i in range(n_steps):
            local_loss = 0.
            n_horizons = self.steps_interval
            # (n_envs, n_ctrl, 3)
            traj_i = trajs[:, i].reshape(self.n_envs, -1, 3)
            hpos, _ = self.c.pre_apply_grad(dpos=traj_i, num_horizons=n_horizons)
            for j in range(n_horizons):
                self.c.on_apply_grad(hpos[j])
                self.scene.step()

            state = self.rope.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            scale = self.scale_array[i]

            local_loss += self.loss_criterion(state).mean()
            local_loss += self.loss_above_plane(state).mean()

            loss += scale * local_loss

        loss.backward()

        deltas = list()
        for horizon_idx in horizon_ids:
            delta = self.c.gather_grad(horizon_idx=horizon_idx)
            deltas.append(delta)

        # (n_envs, n_steps, n_ctrl, 3)
        deltas = torch.stack(deltas, dim=1)
        deltas = deltas.reshape(self.n_envs, n_steps, -1)
        # (n_envs, n_steps, dof)
        deltas = deltas.detach().cpu().numpy()

        return deltas
