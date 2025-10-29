import genesis as gs
import torch
import numpy as np
from train_env import Train_Env
from gd.traj_optim_cmaes import TrajOptimCMAES

class Train_Env_Separation(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, requires_grad=False):
        super().__init__(task, GUI=GUI, camera=camera, n_envs=n_envs, log_dir=log_dir, requires_grad=requires_grad)
        self.steps_interval = 200
    
    def construct_traj_optim(self, max_ddist=0.1, max_grad_norm=1000, debug=False):
        if not self.requires_grad:
            return

        self.c1 = TrajOptimCMAES(
            scene=self.scene,
            rod=self.rope,
            grasp_point_ids=[self.control_idx[0]],
            n_optim_dofs=3,
            max_ddist=max_ddist,
            max_grad_norm=max_grad_norm,
            debug=debug,
        )

        self.c2 = TrajOptimCMAES(
            scene=self.scene,
            rod=self.rope2,
            grasp_point_ids=[self.control_idx[1]],
            n_optim_dofs=3,
            max_ddist=max_ddist,
            max_grad_norm=max_grad_norm,
            debug=debug,
        )

    def construct_scene(self, camera):
        plane = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

        segment_radius = 0.01
        self.rope = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                E=5e3,
                G=1e3,
            ),
            morph=gs.morphs.Rod(
                file="meshes/ropea.npy",
                rest_state="straight",
                pos=(0., 0., 0.012),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        self.rope2 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                E=5e3,
                G=1e3,
            ),
            morph=gs.morphs.Rod(
                file="meshes/ropeb.npy",
                rest_state="straight",
                pos=(0., 0., 0.012),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope02.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [27, 27]
        self.action_dim = len(self.control_idx) * 3

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(0.5, 1.5, 1.), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=30, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(0.5, -1.5, 1.), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=30, GUI=False
        ))

        self.cameras = cameras

    def reward(self):
        A = self.rope.get_all_verts()
        B = self.rope2.get_all_verts()

        # Pairwise distances between ropes for each env:
        # D shape: (n_envs, n_verts_A, n_verts_B)
        diff = A[:, :, None, :] - B[:, None, :, :]
        D = np.linalg.norm(diff, axis=-1)

        # For each vertex in A, distance to nearest in B; and vice versa
        a_to_b_min = D.min(axis=2)  # (n_envs, n_verts_A)
        b_to_a_min = D.min(axis=1)  # (n_envs, n_verts_B)

        # Symmetric NN distance (Chamfer-style), averaged per env
        rewards = a_to_b_min.mean(axis=1) + b_to_a_min.mean(axis=1)  # (n_envs,)

        # Larger reward -> ropes farther apart
        return rewards.tolist()

    def loss_criterion(self, state, state2):
        # (n_envs, n_verts, 3), torch tensor
        verts_batch = state.pos
        verts_batch_2 = state2.pos

        diff = verts_batch[:, :, None, :] - verts_batch_2[:, None, :, :]   # (n_envs, n_verts_A, n_verts_B, 3)
        D = torch.norm(diff, dim=-1)                                       # (n_envs, n

        a_to_b_min = D.min(dim=2).values  # (n_envs, n_verts_A)
        b_to_a_min = D.min(dim=1).values  # (n_envs, n_verts_B)

        loss_chamfer = a_to_b_min.mean(dim=1) + b_to_a_min.mean(dim=1)  # (n_envs,)
        loss_chamfer = - loss_chamfer      # want to maximize chamfer distance

        return loss_chamfer

    def reset(self):
        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx[0]] = True
        self.rope.set_fixed(0, fixed_np)

        fixed_np2 = np.zeros((self.n_envs, self.rope2.n_vertices), dtype=bool)
        fixed_np2[:, self.control_idx[1]] = True
        self.rope2.set_fixed(0, fixed_np2)

    def eval_traj(self, trajs, debug=False):
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
        assert n_ctrl == 2, f"Expected exactly 2 control points for Separation. Got {n_ctrl} control points."

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
        first_fail_step = np.full((self.n_envs,), total_micro_steps, dtype=np.int32)  # micro-step index of first failure

        for i in range(n_steps):
            # Check NaNs BEFORE micro-stepping this macro-step
            verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now = np.isnan(verts_rope).any(axis=(1, 2))
            verts_rope2 = self.rope2.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now2 = np.isnan(verts_rope2).any(axis=(1, 2))
            nan_now = nan_now | nan_now2  # Combine NaN info from both ropes
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
            current_pos = verts_rope[:, self.control_idx[0]]              # (n_envs, 3)
            current_pos2 = verts_rope2[:, self.control_idx[1]]            # (n_envs, 3)
            delta = trajs[:, i].reshape(self.n_envs, -1, 3)            # (n_envs, 2, 3)

            if debug:
                for batch_idx in range(self.n_envs):
                    offset = self.scene.envs_offset[batch_idx]
                    for ii in self.debug_point_nodes:
                        self.scene.clear_debug_object(ii)
                    self.debug_point_nodes = list()
                    for ii in range(len(self.control_idx)):
                        if ii == 0:
                            debug_pos = current_pos + delta[:, ii, :]
                        else:
                            debug_pos = current_pos2 + delta[:, ii, :]
                        debug_pos = debug_pos.copy()
                        if debug_pos[batch_idx, 2] < self.rope.material.segment_radius:
                            color = (1.0, 1.0, 0.0, 0.6)
                            debug_pos[batch_idx, 2] = self.rope.material.segment_radius
                        else:
                            color = (0.0, 1.0, 0.0, 0.6)
                        self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                            pos=debug_pos[batch_idx] + offset,
                            radius=0.016,
                            color=color
                        ))

            for j in range(steps_interval):
                if not alive.any():
                    break

                # NOTE: Do not move already-failed envs
                delta[~alive, :, :] = 0.0

                alpha = (j + 1) / steps_interval

                # Apply target positions; if set_pos_single isn't batch-aware, loop envs instead.
                for k in range(n_ctrl):
                    if k == 0:
                        target_pos = current_pos + alpha * delta[:, k, :]  # (n_envs, 3)
                        self.rope.set_pos_single(target_pos, self.control_idx[k])
                    else:
                        target_pos = current_pos2 + alpha * delta[:, k, :]  # (n_envs, 3)
                        self.rope2.set_pos_single(target_pos, self.control_idx[k])

                self.scene.step()

                if j % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

                # Post-step: detect collisions
                collided = self.rope._solver.vertices_collision.collided.to_numpy()  # (n_verts, n_envs)
                collided = collided.T  # (n_envs, n_vertices)
                verts_to_check = np.array(self.control_idx[0]) + self.rope._v_start
                collided_ctrl = collided[:, verts_to_check]           # (n_envs,)

                collided2 = self.rope2._solver.vertices_collision.collided.to_numpy()  # (n_verts, n_envs)
                collided2 = collided2.T  # (n_envs, n_vertices)
                verts_to_check2 = np.array(self.control_idx[1]) + self.rope2._v_start
                collided_ctrl2 = collided2[:, verts_to_check2]         # (n_envs,)

                collided_ctrl = collided_ctrl | collided_ctrl2  # Combine collision info from both ropes

                newly_collided = collided_ctrl & alive
                if newly_collided.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_collided] = np.minimum(first_fail_step[newly_collided], global_step)
                    ever_collided[newly_collided] = True
                    alive[newly_collided] = False

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                verts_rope2_post = self.rope2.get_all_verts()
                nan_after2 = np.isnan(verts_rope2_post).any(axis=(1, 2))
                nan_after = nan_after | nan_after2  # Combine NaN info from both ropes
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

    def gd_one_step(self, trajs, ratio):
        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        trajs_origin = trajs.copy()
        trajs = torch.tensor(trajs, dtype=gs.tc_float)
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 3 == 0 and dof // 3 == n_ctrl, (
            f"dof must be 3 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )
        assert n_ctrl == 2, f"Expected exactly 2 control points for Separation. Got {n_ctrl} control points."

        total_horizon = 0
        horizon_ids = list()

        self.reset()

        loss = 0.
        for i in range(n_steps):
            local_loss = 0.
            n_horizons = self.steps_interval
            # (n_envs, n_ctrl, 3)
            traj_i = trajs[:, i].reshape(self.n_envs, -1, 3)
            traj_i_1 = traj_i[:, 0, :].unsqueeze(1)  # (n_envs, 1, 3)
            traj_i_2 = traj_i[:, 1, :].unsqueeze(1)  # (n_envs, 1, 3)

            hpos_1, _ = self.c1.pre_apply_grad(dpos=traj_i_1, num_horizons=n_horizons)
            hpos_2, _ = self.c2.pre_apply_grad(dpos=traj_i_2, num_horizons=n_horizons)
            for j in range(n_horizons):
                self.c1.on_apply_grad(hpos_1[j])
                self.c2.on_apply_grad(hpos_2[j])
                self.scene.step()

            state = self.rope.get_state()
            state2 = self.rope2.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            scale = self.scale_array[i]

            local_loss += self.loss_criterion(state, state2).mean()
            local_loss += self.loss_above_plane(state).mean()
            local_loss += self.loss_above_plane(state2).mean()

            loss += scale * local_loss

        loss.backward()

        deltas_1 = list()
        deltas_2 = list()
        for horizon_idx in horizon_ids:
            delta_1 = self.c1.gather_grad(horizon_idx=horizon_idx)
            deltas_1.append(delta_1)
            delta_2 = self.c2.gather_grad(horizon_idx=horizon_idx)
            deltas_2.append(delta_2)

        # (n_envs, n_steps, 1, 3)
        deltas_1 = torch.stack(deltas_1, dim=1)
        # (n_envs, n_steps, 1, 3)
        deltas_2 = torch.stack(deltas_2, dim=1)
        # (n_envs, n_steps, 2, 3)
        deltas = torch.cat([deltas_1, deltas_2], dim=2)
        assert deltas.shape == (self.n_envs, n_steps, n_ctrl, 3)
        deltas = deltas.reshape(self.n_envs, n_steps, -1)
        deltas = deltas.detach().cpu().numpy()

        # ensure each delta is within ratio x trajs_origin
        deltas = self.adaptive_scale(trajs_origin, deltas, ratio=ratio)

        print(f'traj: {np.abs(trajs_origin).mean(0).mean(0)}')
        print(f'delta: {np.abs(deltas).mean(0).mean(0)}')

        # Update trajs
        return trajs_origin + deltas
