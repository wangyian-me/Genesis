import genesis as gs
import torch
import numpy as np
from train_env import Train_Env
from controller import RobotController, RobotControllerPink

class Train_Env_Slingshot(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, requires_grad=False, scene_version=None):
        super().__init__(task, GUI=GUI, camera=camera, n_envs=n_envs, log_dir=log_dir, requires_grad=requires_grad, scene_version=scene_version)
        self.steps_interval = 200

        # NOTE: assume running from "examples/rod"
        initial_gripper_qpos = np.load("target_pos/slingshot_pregrasp_qpos.npy")
        self.initial_gripper_qpos = torch.tensor(initial_gripper_qpos, dtype=gs.tc_float)

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
                K=8e5,  # 5e5
                E=1e5,
                G=0,
                use_inextensible=False,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=12,
                interval=0.02,
                axis="x",
                pos=(0.0, 0.0, 0.21),
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
                needs_coup=False
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.3,
                pos=(0, 0, 0.15),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.b2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=False
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.3,
                pos=(0.24, 0, 0.15),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.sphere = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, rho=200, coup_friction=0.02,
            ),
            morph=gs.morphs.Sphere(
                radius=0.02,
                pos=(0.12, 0.06, 0.2),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 1.0)
            )
        )

        self.cube = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, rho=20, coup_friction=0.02,
            ),
            morph=gs.morphs.Box(
                pos=(0.12, 0.23, 0.22),
                size=(0.08, 0.08, 0.08),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.7, 0.7, 1.0)
            )
        )

        self.table = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.02,
            ),
            morph=gs.morphs.Box(
                pos=(0.12, 1.0, 0.09),
                size=(0.8, 1.9, 0.18),
                euler=(0, 0, 0),
                fixed=True,
            ),
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [6]
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
                K=8e5,  # 5e5
                E=1e5,
                G=0,
                use_inextensible=False,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=12,
                interval=0.02,
                axis="x",
                pos=(0.0, 0.0, 0.21),
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
                needs_coup=False
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.3,
                pos=(0, 0, 0.15),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.b2 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=False
            ),
            morph=gs.morphs.Cylinder(
                radius=0.015,
                height=0.3,
                pos=(0.24, 0, 0.15),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.sphere = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, rho=200, coup_friction=0.02,
            ),
            morph=gs.morphs.Sphere(
                radius=0.02,
                pos=(0.12, 0.06, 0.2),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 1.0)
            )
        )

        self.cube = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, rho=20, coup_friction=0.02,
            ),
            morph=gs.morphs.Box(
                pos=(0.12, 0.23, 0.22),
                size=(0.08, 0.08, 0.08),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.7, 0.7, 1.0)
            )
        )

        self.table = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.02,
            ),
            morph=gs.morphs.Box(
                pos=(0.12, 1.0, 0.09),
                size=(0.8, 1.9, 0.18),
                euler=(0, 0, 0),
                fixed=True,
            ),
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(-0.33, -0.65, 0),
                fixed=True,
                collision=True,
                links_to_keep=['panda_grasptarget'],
            ),
            surface=gs.surfaces.Smooth(),
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [6]
        self.action_dim = len(self.control_idx) * 6

        # Construct controller
        for f in [self.franka1]:
            f.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 30, 30]),
            )
            f.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
            )
            f.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
                np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
            )
        self._ef1 = self.franka1.get_link("panda_grasptarget")

        # move to pre-grasp pose
        x0 = 0.12
        y0 = -0.1
        z0 = 0.216
        open_gap = 0.03

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_pos=(x0, y0, z0),
            initial_gripper_gap=open_gap,
        )

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(2, -1.4, 1.5), up=(0, 0, 1),
            lookat=(0.12, 0.2, 0.18), fov=24, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(-0.15, 1.4, 1.2), up=(0, 0, 1),
            lookat=(0.12, 0.25, 0.35), fov=33, GUI=False
        ))

        self.cameras = cameras

    def reward(self):
        # [n_envs, 3]
        cube_pos = self.cube.get_pos().cpu().numpy()

        rewards = []
        for i in range(self.n_envs):
            cube_pos_y = cube_pos[i, 1]

            # NOTE: New: max y position is 5, min y position is 0
            cube_pos_y = min(max(cube_pos_y, 0), 5)

            # we want the cube to be as far as possible in +y direction
            rewards.append(cube_pos_y)

        return rewards

    def reset(self, debug=False, envs_idx=None):
        self.scene.reset(envs_idx=envs_idx)

        if self.scene_version == 1:
            fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
            fixed_np[:, self.control_idx] = True
            fixed_np[:, [0, 1, 10, 11]] = True  # also fix the two ends
            self.rope.set_fixed(0, fixed_np)

        elif self.scene_version == 2:
            envs_idx_ = range(max(self.n_envs, 1)) if envs_idx is None else [int(i) for i in envs_idx]

            for f in [self.franka1]:
                f.set_qpos(
                    np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * len(envs_idx_)),
                    envs_idx=envs_idx_
                )

            self.rope.set_fixed_states(fixed_ids=[0, 1, 10, 11])

            self.c1.set_initial_dofs_position(self.initial_gripper_qpos, False, envs_idx=envs_idx)

            force = -1.0
            self.c1.control_robot(force, force, g_dof_use_force=True)
            for i in range(100):
                self.scene.step()
                if i % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

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
        ever_stretched = np.zeros((self.n_envs,), dtype=bool)    # True if excessive stretching force occurred
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
                f_s = self.rope.get_all_stretching_force()[0][self.control_idx[0]]
                print(f"Step {i}, f_s {f_s}, mag {np.linalg.norm(f_s)}")
                debug_pos = current_pos + delta
                debug_pos = debug_pos.copy()
                for batch_idx in range(self.n_envs):
                    offset = self.scene.envs_offset[batch_idx]
                    for ii in self.debug_point_nodes:
                        self.scene.clear_debug_object(ii)
                    self.debug_point_nodes = list()
                    for ii in range(len(self.control_idx)):
                        self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                            pos=debug_pos[batch_idx, ii] + offset,
                            radius=0.016,
                            color=(0.0, 1.0, 0.0, 0.6)
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

                # Post-step: detect excessive stretching force
                stretched_force = self.rope.get_all_stretching_force()[:, self.control_idx[0]]       # (n_envs, 3)
                force_magnitudes = np.linalg.norm(stretched_force, axis=1)   # (n_envs,)
                newly_exceed_force = (force_magnitudes > 50) & alive
                if newly_exceed_force.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_exceed_force] = np.minimum(first_fail_step[newly_exceed_force], global_step)
                    ever_stretched[newly_exceed_force] = True
                    alive[newly_exceed_force] = False

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_nan_after] = np.minimum(first_fail_step[newly_nan_after], global_step)
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

        # now we only fix the two ends
        self.rope.set_fixed_states(
            fixed_ids=[0, 1, 10, 11]
        )

        for s in range(500):
            self.scene.step()
            if s % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

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
        """
        Evaluate trajectories.

        Rewards:
        - If an env survives all micro-steps: reward = self.reward()[env].
        - If an env COLLIDES or gets NaNs in verts: reward = survival_time / total_micro_steps.
        - If env reward is NaN at the end: reward = -100.

        Survival time counts micro-steps from 0..N, where N = n_steps * steps_interval.
        """
        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 6 == 0 and dof // 6 == n_ctrl, (
            f"dof must be 6 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        n_steps_sub = 2
        if kwargs.get("qpos", None) is None:
            self.qpos_seq = np.zeros((n_steps * n_steps_sub, self.n_envs, len(self.control_idx) * 9))
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
            delta = trajs[:, i].reshape(self.n_envs, 6)            # (n_envs, 6), n_ctrl == 1!
            delta = torch.tensor(delta, dtype=gs.tc_float)

            n_intervals_per_substep = steps_interval // n_steps_sub

            for j in range(n_steps_sub):
                if not alive.any():
                    break

                # NOTE: Do not move already-failed envs
                delta[~alive, :] = 0.0

                alpha = 1 / n_steps_sub
                dxyz = alpha * delta[:, :3]
                drot = alpha * delta[:, 3:]

                if self.use_qpos:
                    qpos = self.qpos_seq[i * n_steps_sub + j]
                    qpos = torch.tensor(qpos, dtype=gs.tc_float)
                    self.c1.robot.control_dofs_position(qpos[..., :-2], self.c1.motors_dof)
                    gripper_arg = torch.tensor([[-3, -3]] * self.scene.n_envs)
                    self.c1.robot.control_dofs_force(gripper_arg, self.c1.fingers_dof)

                    self.c1.draw_debug_point(dxyz, min_z=0.03)
                else:
                    qpos = self.c1.control_robot(
                        -3.0, -3.0, g_dof_use_force=True,
                        dx=dxyz[:, 0], dy=dxyz[:, 1], dz=dxyz[:, 2], di=drot[:, 0], dj=drot[:, 1], dk=drot[:, 2], min_z=0.03
                    )
                    self.qpos_seq[i * n_steps_sub + j] = qpos.cpu().numpy()

                for k in range(n_intervals_per_substep):
                    self.scene.step()

                    if (k + j * n_intervals_per_substep) % 10 == 0:
                        for cid, cam in enumerate(self.cameras):
                            img = cam.render()[0]
                            self.frames[cid].append(img)

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    global_step = i * steps_interval + (j + 1)
                    first_fail_step[newly_nan_after] = np.minimum(first_fail_step[newly_nan_after], global_step)
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

        # release gripper
        self.c1.control_robot(0.08, 0.08)
        for s in range(500):
            self.scene.step()
            if s % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

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

        if not self.use_qpos:
            self.qpos_seq = self.qpos_seq.transpose(1, 0, 2)  # (n_envs, n_steps * n_steps_sub, n_dofs)
            self.qpos_seq = self.qpos_seq.astype(np.float32)

        return final.astype(np.float32)

    def compute_observation(self):
        verts_rope = self.rope.get_all_verts_tc()                   # (n_envs, n_verts, 3)
        obs_rope = verts_rope.reshape(self.n_envs, -1).to(torch.float32)
        return obs_rope

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
        action = action * self._act_magnitude
        action = torch.clamp(action, self._mdp_info.action_space.low, self._mdp_info.action_space.high)
        action_xyz = action[:, :self._act_dim // 2]
        action_rot = action[:, self._act_dim // 2:]

        action_xyz_norm = torch.linalg.norm(action_xyz, dim=1, keepdim=True)
        scale = torch.ones_like(action_xyz_norm)
        over = action_xyz_norm > self._l2_limit
        scale[over] = self._l2_limit / (action_xyz_norm[over] + gs.EPS)
        action_xyz = action_xyz * scale

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
            action_xyz[~alive, :] = 0.0
            action_rot[~alive, :] = 0.0

            alpha = 1 / n_steps_sub
            dxyz = alpha * action_xyz
            drot = alpha * action_rot

            qpos = self.c1.control_robot(
                -3.0, -3.0, g_dof_use_force=True,
                dx=dxyz[:, 0], dy=dxyz[:, 1], dz=dxyz[:, 2], di=drot[:, 0], dj=drot[:, 1], dk=drot[:, 2], min_z=0.03
            )

            for k in range(n_intervals_per_substep):
                self.scene.step()

            # Post-step: detect NaNs that emerge during micro-stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive
            if newly_nan_after.any():
                absorbing[newly_nan_after] = True
                alive[newly_nan_after] = False

        # release gripper
        self.c1.control_robot(0.08, 0.08)
        for s in range(500):
            self.scene.step()

        # Compute base rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)

        # Compose final rewards
        rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
        failed = absorbing | env_rewards_nan
        rewards[failed] = 0.0
        rewards[~failed] = env_rewards[~failed]
        rewards = torch.as_tensor(rewards).reshape((self.n_envs,))
        absorbing = torch.as_tensor(absorbing).reshape((self.n_envs,))

        next_obs = self.compute_observation()

        return next_obs, rewards, absorbing, [{}] * self.n_envs
