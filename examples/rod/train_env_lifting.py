import genesis as gs
import torch
import numpy as np
from train_env import Train_Env
from controller import (
    rod_vertex_attached_to_gripper,
    rod_vertex_detached_from_gripper,
    RobotController,
    RobotControllerPink,
)

class Train_Env_Lifting(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, requires_grad=False, scene_version=None):
        super().__init__(task, GUI=GUI, camera=camera, n_envs=n_envs, log_dir=log_dir, requires_grad=requires_grad, scene_version=scene_version)
        self.steps_interval = 200

        # initial distance between control points
        self.control_dist_init = self.rope.get_geodesic_distance(self.control_idx[0], self.control_idx[1])

        print(f'Initial distance between control points: {self.control_dist_init[0]:.4f}')

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
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=30,
                interval=0.02,
                axis="x",
                pos=(0.3, 0.12, 0.02),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        self.b1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/nut_open.glb",
                pos= (0.53, 0, 0.05),
                euler=(0, 180, 90),
                scale=(1, 1, 1),
            )
        )

        self.b2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/nut_open.glb",
                pos= (0.67, 0, 0.05),
                euler=(0, 180, 90),
                scale=(1, 1, 1),
            )
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [6, 23]
        self.action_dim = len(self.control_idx) * 3

    def construct_scene_v2(self, camera):
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
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=30,
                interval=0.02,
                axis="x",
                pos=(0.3, 0.12, 0.02),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        self.b1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/nut_open.glb",
                pos= (0.53, 0, 0.05),
                euler=(0, 180, 90),
                scale=(1, 1, 1),
            )
        )

        self.b2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.1,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/nut_open.glb",
                pos= (0.67, 0, 0.05),
                euler=(0, 180, 90),
                scale=(1, 1, 1),
            )
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(0.25, -0.4, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget'],
            ),
            surface=gs.surfaces.Smooth(),
        )

        self.franka2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(0.95, -0.4, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget'],
            ),
            surface=gs.surfaces.Smooth(),
        )

        if camera:
            self.construct_cameras()

        gripper_geom_indices = list()
        for gi in self.franka1.get_link("panda_leftfinger")._geoms:
            gripper_geom_indices.append(gi.idx)
        for gi in self.franka1.get_link("panda_rightfinger")._geoms:
            gripper_geom_indices.append(gi.idx)
        for gi in self.franka2.get_link("panda_leftfinger")._geoms:
            gripper_geom_indices.append(gi.idx)
        for gi in self.franka2.get_link("panda_rightfinger")._geoms:
            gripper_geom_indices.append(gi.idx)

        self.gripper_geom_indices = gripper_geom_indices
        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [4, 25]
        self.action_dim = len(self.control_idx) * 6

        # Construct controller
        for f in [self.franka1, self.franka2]:
            f.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 80, 80]),
            )
            f.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
            )
            f.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
                np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
            )
        self._ef1 = self.franka1.get_link("panda_grasptarget")
        self._ef2 = self.franka2.get_link("panda_grasptarget")

        # NOTE: use the first env to initalize gripper pos
        init_pos = self.rope.get_all_verts()
        init_pos_f1 = init_pos[0, self.control_idx[0], :]
        init_pos_f1[2] = 0.013       # a bit above the ground
        init_pos_f2 = init_pos[0, self.control_idx[1], :]
        init_pos_f2[2] = 0.013       # a bit above the ground
        open_gap = 0.01

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_pos=init_pos_f1.tolist(),
            initial_gripper_gap=open_gap,
        )

        self.c2 = RobotControllerPink(
            self.scene, self.franka2, self._ef2,
            initial_pos=init_pos_f2.tolist(),
            initial_gripper_gap=open_gap,
        )

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(1., 1.7, 1.), up=(0, 0, 1),
            lookat=(0.5, 0., 0), fov=24, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(0.2, 1.7, 0.6), up=(0, 0, 1),
            lookat=(0.6, 0., 0), fov=24, GUI=False
        ))

        self.cameras = cameras

    def reward(self):
        # [n_envs, 3]
        nut_a_pos_batch = self.b1.get_pos().cpu().numpy()
        nut_b_pos_batch = self.b2.get_pos().cpu().numpy()
        # # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()

        rewards = []
        for i in range(self.n_envs):
            verts = verts_batch[i]
            nut_a_pos = nut_a_pos_batch[i]
            nut_b_pos = nut_b_pos_batch[i]

            dist = np.linalg.norm(nut_a_pos - nut_b_pos)
            height = nut_a_pos[2] + nut_b_pos[2]

            dist_nut_a = np.linalg.norm(verts - nut_a_pos, axis=1)
            min_dists_nut_a = np.min(dist_nut_a)

            dist_nut_b = np.linalg.norm(verts - nut_b_pos, axis=1)
            min_dists_nut_b = np.min(dist_nut_b)

            # dist: we want nut a and nut b to be close
            # height: we want the nuts to be lifted up
            # min_dists_nut_a, min_dists_nut_b: we want the rope to be close to the nuts
            reward = - dist + 2 * height
            reward += - 0.5 * (min_dists_nut_a + min_dists_nut_b)

            rewards.append(reward)

        return rewards

    def reset(self, debug=False, envs_idx=None):
        self.scene.reset(envs_idx=envs_idx)

        if self.scene_version == 1:
            fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
            fixed_np[:, self.control_idx] = True
            self.rope.set_fixed(0, fixed_np)

        elif self.scene_version == 2:
            envs_idx_ = range(max(self.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]

            for f in [self.franka1, self.franka2]:
                f.set_qpos(
                    np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * len(envs_idx_)),
                    envs_idx=envs_idx_
                )

            for i in envs_idx_:
                rod_vertex_detached_from_gripper(self.rope, self.control_idx[0], envs_idx=i)
                rod_vertex_detached_from_gripper(self.rope, self.control_idx[1], envs_idx=i)

            qpos1 = self.c1.set_initial_position(envs_idx=envs_idx)
            qpos2 = self.c2.set_initial_position(envs_idx=envs_idx)
            if not self.use_qpos:
                qpos1 = qpos1.cpu().numpy()
                qpos2 = qpos2.cpu().numpy()
                qpos = np.concatenate([qpos1, qpos2], axis=-1)
                self.qpos_seq[0] = qpos

            for i in envs_idx_:
                rod_vertex_attached_to_gripper(self.rope, self.control_idx[0], self._ef1, envs_idx=i)
                rod_vertex_attached_to_gripper(self.rope, self.control_idx[1], self._ef2, envs_idx=i)

    def eval_traj_v2(self, trajs, debug=False, **kwargs):
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
        assert dof % 6 == 0 and dof // 6 == n_ctrl, (
            f"dof must be 6 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )
        
        n_steps_sub = 2
        if kwargs.get("qpos", None) is None:
            self.qpos_seq = np.zeros((n_steps * n_steps_sub + 1, self.n_envs, len(self.control_idx) * 9))
            self.use_qpos = False
        else:
            self.qpos_seq = kwargs["qpos"]
            self.use_qpos = True

        self.reset(debug=debug)

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
            delta = trajs[:, i].reshape(self.n_envs, 2 * 6)            # (n_envs, 2 * 6), n_ctrl == 2
            # first half: translation
            delta1_xyz = torch.tensor(delta[:, 0:3], dtype=gs.tc_float)
            delta2_xyz = torch.tensor(delta[:, 3:6], dtype=gs.tc_float)
            # second half: rotation
            delta1_rot = torch.tensor(delta[:, 6:9], dtype=gs.tc_float)
            delta2_rot = torch.tensor(delta[:, 9:12], dtype=gs.tc_float)

            n_intervals_per_substep = steps_interval // n_steps_sub

            for j in range(n_steps_sub):
                if not alive.any():
                    break

                # NOTE: Do not move already-failed envs
                delta1_xyz[~alive, :] = 0.0
                delta2_xyz[~alive, :] = 0.0
                delta1_rot[~alive, :] = 0.0
                delta2_rot[~alive, :] = 0.0

                alpha = 1 / n_steps_sub
                dxyz1 = alpha * delta1_xyz
                drot1 = alpha * delta1_rot
                dxyz2 = alpha * delta2_xyz
                drot2 = alpha * delta2_rot

                if self.use_qpos:
                    qpos = self.qpos_seq[i * n_steps_sub + j + 1]
                    qpos = torch.tensor(qpos, dtype=gs.tc_float)
                    qpos1, qpos2 = torch.split(qpos, qpos.shape[0] // 2)
                    self.c1.robot.control_dofs_position(qpos1[..., :-2], self.c1.motors_dof)
                    self.c2.robot.control_dofs_position(qpos2[..., :-2], self.c2.motors_dof)
                    self.c1.robot.control_dofs_position(qpos1[..., -2:], self.c1.fingers_dof)
                    self.c2.robot.control_dofs_position(qpos2[..., -2:], self.c2.fingers_dof)

                    self.c1.draw_debug_point(dxyz1, min_z=0.03)
                    self.c2.draw_debug_point(dxyz2, min_z=0.03)
                else:
                    qpos1 = self.c1.control_robot(
                        0, 0,
                        dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=0.03
                    )
                    qpos2 = self.c2.control_robot(
                        0, 0,
                        dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=0.03
                    )
                    qpos1 = qpos1.cpu().numpy()
                    qpos2 = qpos2.cpu().numpy()
                    # (n_envs, n_dofs * 2)
                    qpos = np.concatenate([qpos1, qpos2], axis=-1)
                    self.qpos_seq[i * n_steps_sub + j + 1] = qpos

                for k in range(n_intervals_per_substep):
                    self.scene.step()

                    if (k + j * n_intervals_per_substep) % 10 == 0:
                        for cid, cam in enumerate(self.cameras):
                            img = cam.render()[0]
                            self.frames[cid].append(img)

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

                # Post-step: detect ik convergence
                if hasattr(self.c1, 'convergence'):
                    newly_not_converged = ~self.c1.convergence & alive
                    if newly_not_converged.any():
                        global_step = i * steps_interval + (j + 1)
                        first_fail_step[newly_not_converged] = np.minimum(first_fail_step[newly_not_converged], global_step)
                        alive[newly_not_converged] = False

                if hasattr(self.c2, 'convergence'):
                    newly_not_converged2 = ~self.c2.convergence & alive
                    if newly_not_converged2.any():
                        global_step = i * steps_interval + (j + 1)
                        first_fail_step[newly_not_converged2] = np.minimum(first_fail_step[newly_not_converged2], global_step)
                        alive[newly_not_converged2] = False

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

        for i in range(self.n_envs):
            rod_vertex_detached_from_gripper(self.rope, self.control_idx[0], envs_idx=i)
            rod_vertex_detached_from_gripper(self.rope, self.control_idx[1], envs_idx=i)

        if not self.use_qpos:
            self.qpos_seq = self.qpos_seq.transpose(1, 0, 2)  # (n_envs, n_steps * n_steps_sub + 1, n_dofs)
            self.qpos_seq = self.qpos_seq.astype(np.float32)

        return final.astype(np.float32)

    def compute_observation(self):
        verts_rope = self.rope.get_all_verts_tc()                   # (n_envs, n_verts, 3)
        obs_rope = verts_rope.reshape(self.n_envs, -1).to(torch.float32)

        nut_a_pos = self.b1.get_pos()     # shape: (n_envs, 3)
        nut_a_pos = torch.tensor(nut_a_pos, dtype=torch.float32)
        nut_b_pos = self.b2.get_pos()  # shape: (n_envs, 3)
        nut_b_pos = torch.tensor(nut_b_pos, dtype=torch.float32)

        obs = torch.cat([obs_rope, nut_a_pos, nut_b_pos], dim=1)
        return obs

    def step_all(self, env_mask, action):
        """ Used in MushroomRL """
        # Accept torch or numpy; operate and return torch for torch backend
        if isinstance(action, np.ndarray):
            action = torch.tensor(action)
        else:
            action = torch.as_tensor(action)
        if action.ndim == 1:
            action = action.unsqueeze(0)

        if isinstance(env_mask, np.ndarray):
            env_mask_np = torch.tensor(env_mask, dtype=torch.bool)
        else:
            env_mask_np = torch.as_tensor(env_mask, dtype=torch.bool)

        assert action.shape == (self.n_envs, self._act_dim), \
            f"Expected action shape {(self.n_envs, self._act_dim)}, got {action.shape}"

        # Track failure states and absorbing flags (only track masked envs)
        absorbing = np.zeros((self.n_envs,), dtype=bool)
        tracked = env_mask_np.clone().cpu().numpy()
        alive = tracked.copy()

        action = action.to(torch.float32)
        action = torch.clamp(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)

        # Split action for two controllers: first half for controller 1, second half for controller 2
        action1_xyz = action[:, :self._act_dim // 4]
        action2_xyz = action[:, self._act_dim // 4:self._act_dim // 2]
        action1_rot = action[:, self._act_dim // 2:self._act_dim // 2 + self._act_dim // 4]
        action2_rot = action[:, self._act_dim // 2 + self._act_dim // 4:]

        # Apply L2 limit to translation actions
        action1_xyz_norm = torch.linalg.norm(action1_xyz, dim=1, keepdim=True)
        scale1 = torch.ones_like(action1_xyz_norm)
        over1 = action1_xyz_norm > self._l2_limit
        scale1[over1] = self._l2_limit / (action1_xyz_norm[over1] + gs.EPS)
        action1_xyz = action1_xyz * scale1

        action2_xyz_norm = torch.linalg.norm(action2_xyz, dim=1, keepdim=True)
        scale2 = torch.ones_like(action2_xyz_norm)
        over2 = action2_xyz_norm > self._l2_limit
        scale2[over2] = self._l2_limit / (action2_xyz_norm[over2] + gs.EPS)
        action2_xyz = action2_xyz * scale2

        # Check NaNs BEFORE micro-stepping this macro-step
        verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
        nan_now = np.isnan(verts_rope).any(axis=(1, 2))
        newly_nan = nan_now & alive
        if newly_nan.any():
            # Failure occurs before any micro-step of this macro-step
            absorbing[newly_nan] = True
            alive[newly_nan] = False

        n_steps_sub = 2
        n_intervals_per_substep = self._steps_per_action // n_steps_sub

        for j in range(n_steps_sub):
            if not (alive & tracked).any():
                break

            # NOTE: Do not move already-failed envs
            action1_xyz[~alive, :] = 0.0
            action1_rot[~alive, :] = 0.0
            action2_xyz[~alive, :] = 0.0
            action2_rot[~alive, :] = 0.0

            alpha = 1 / n_steps_sub
            dxyz1 = alpha * action1_xyz
            drot1 = alpha * action1_rot
            dxyz2 = alpha * action2_xyz
            drot2 = alpha * action2_rot

            qpos1 = self.c1.control_robot(
                0, 0,
                dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=0.03
            )
            qpos2 = self.c2.control_robot(
                0, 0,
                dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=0.03
            )

            for k in range(n_intervals_per_substep):
                self.scene.step()

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
                    absorbing[newly_stretched] = True
                    alive[newly_stretched] = False

            # Post-step: detect ik convergence for controller 1
            if hasattr(self.c1, 'convergence'):
                newly_not_converged = ~self.c1.convergence & alive
                if newly_not_converged.any():
                    absorbing[newly_not_converged] = True
                    alive[newly_not_converged] = False

            # Post-step: detect ik convergence for controller 2
            if hasattr(self.c2, 'convergence'):
                newly_not_converged2 = ~self.c2.convergence & alive
                if newly_not_converged2.any():
                    absorbing[newly_not_converged2] = True
                    alive[newly_not_converged2] = False

            # Post-step: detect NaNs that emerge during micro-stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive
            if newly_nan_after.any():
                absorbing[newly_nan_after] = True
                alive[newly_nan_after] = False

        # Compute base rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)

        # Compose final rewards
        rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
        failed = absorbing | env_rewards_nan
        rewards[failed] = 0.0
        rewards[~failed] = env_rewards[~failed] + 30.
        rewards = torch.as_tensor(rewards).reshape((self.n_envs,))
        absorbing = torch.as_tensor(absorbing).reshape((self.n_envs,))

        next_obs = self.compute_observation()

        return next_obs, rewards, absorbing, [{}] * self.n_envs
