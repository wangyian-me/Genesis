import genesis as gs
import time
import torch
import numpy as np
from train_env import Train_Env
from gd.traj_optim_cmaes import TrajOptimCMAES
from controller import RobotController, RobotControllerPink, TrajOptimController

class Train_Env_Wiring_ring(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, n_substeps_per_step=None, requires_grad=False, scene_version=None):
        gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        )

        scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(
                dt=1e-3,
                substeps=5,
                requires_grad=requires_grad,
            ),
            rod_options=gs.options.RodOptions(
                damping=15.0,
                angular_damping=10.0,
                adjacent_gap=3,
                n_pbd_iters=20,
            ),
            show_viewer=GUI,
        )
        init_gripper_qpos1 = np.load('target_pos/wiring_ring_pregrasp_qpos1.npy')
        self.init_gripper_qpos1 = torch.tensor(init_gripper_qpos1, dtype=gs.tc_float)
        init_gripper_qpos2 = np.load('target_pos/wiring_ring_pregrasp_qpos2.npy')
        self.init_gripper_qpos2 = torch.tensor(init_gripper_qpos2, dtype=gs.tc_float)
        super().__init__(task, scene=scene, GUI=GUI, camera=camera, n_envs=n_envs, n_substeps_per_step=n_substeps_per_step, log_dir=log_dir, requires_grad=requires_grad, scene_version=scene_version)

        self.ring1_center = np.array([0.27, 0.0, self.rope.material.segment_radius], dtype=gs.np_float)
        self.ring2_center = np.array([0.09, -0.27, self.rope.material.segment_radius], dtype=gs.np_float)
        self.ring1_normal = np.array([-1., 0., 0.], dtype=gs.np_float)
        self.ring2_normal = np.array([0., -1., 0.], dtype=gs.np_float)

        self.ring1_center_tc = torch.tensor([0.27, 0.0, self.rope.material.segment_radius], dtype=gs.tc_float)
        self.ring2_center_tc = torch.tensor([0.09, -0.27, self.rope.material.segment_radius], dtype=gs.tc_float)
        self.ring1_normal_tc = torch.tensor([-1., 0., 0.], dtype=gs.tc_float)
        self.ring2_normal_tc = torch.tensor([0., -1., 0.], dtype=gs.tc_float)

        # initial distance between control points
        self.control_dist_init = self.rope.get_geodesic_distance(self.control_idx[0], self.control_idx[1])

        # NOTE: assume running from "examples/rod"
        self.target_pos_sub1 = np.load("target_pos/wiring_ring_subtask1.npy")
        self.target_pos_sub1_tc = torch.tensor(self.target_pos_sub1, dtype=gs.tc_float)
        self.target_pos_sub2 = np.load("target_pos/wiring_ring_subtask2.npy")
        self.target_pos_sub2_tc = torch.tensor(self.target_pos_sub2, dtype=gs.tc_float)
        self.target_pos_sub3 = np.load("target_pos/wiring_ring_finalpos.npy")
        self.target_pos_sub3_tc = torch.tensor(self.target_pos_sub3, dtype=gs.tc_float)
        print(f'Loaded target pos from "wiring_ring_subtask1.npy", shape = {self.target_pos_sub1.shape}')
        print(f'Loaded target pos from "wiring_ring_subtask2.npy", shape = {self.target_pos_sub2.shape}')
        print(f'Loaded target pos from "wiring_ring_finalpos.npy", shape = {self.target_pos_sub3.shape}')
        print(f'Initial distance between control points: {self.control_dist_init[0]:.4f}')

        # Use the counter to track how many steps have been taken for each env
        self._env_step_counter = np.array([0] * self.n_envs)
        self._min_z = self.rope.material.segment_radius

    def construct_traj_optim(self, max_ddist=0.1, max_grad_norm=1000, controller=False, debug=False, **kwargs):
        if not self.requires_grad:
            return

        if controller:
            self.c = TrajOptimController(
                scene=self.scene,
                rod=self.rope,
                # NOTE: Grasp point ids changes throughout the task!
                grasp_point_ids=[self.control_idx[1]],
                n_stages=self._n_steps,
                n_optim_dofs=3,
                max_ddist=max_ddist,
                max_grad_norm=max_grad_norm,
                debug=debug,
                **kwargs
            )
        else:
            self.c = TrajOptimCMAES(
                scene=self.scene,
                rod=self.rope,
                grasp_point_ids=self.control_idx,
                n_optim_dofs=3,
                max_ddist=max_ddist,
                max_grad_norm=max_grad_norm,
                debug=debug,
            )

    def construct_scene(self, camera):
        plane = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.01,
            ),
            morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

        segment_radius = 0.01
        self.rope = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                # K=1e5,
                E=1e3,
                G=1e3,
                # use_inextensible=False
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=60,
                interval=0.01,
                axis="x",
                pos=(0.3, 0.0, 0.02),
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

        self.ring1 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.008,
                static_friction=0.1,
                kinetic_friction=0.08,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=24,
                radius=0.04,
                axis="y",
                pos=(0.27, 0.0, 0.008),
                euler=(-30, 0, 0),
                gap=1,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4),
                vis_mode='recon',
            )
        )

        self.ring2 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.008,
                static_friction=0.1,
                kinetic_friction=0.08,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=24,
                radius=0.04,
                axis="y",
                pos=(0.09, -0.27, 0.008),
                euler=(-30, 0, 90),
                gap=1,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4),
                vis_mode='recon',
            )
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [11, 30]
        self.action_dim = len(self.control_idx) * 3

    def construct_scene_v2(self, camera):
        plane = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.01,
            ),
            morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

        segment_radius = 0.01
        self.rope = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=segment_radius,
                segment_mass=0.001,
                # K=1e5,
                E=1e3,
                G=1e3,
                # use_inextensible=False
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=60,
                interval=0.01,
                axis="x",
                pos=(0.3, 0.0, 0.02),
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

        self.ring1 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.008,
                static_friction=0.1,
                kinetic_friction=0.08,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=24,
                radius=0.04,
                axis="y",
                pos=(0.27, 0.0, 0.008),
                euler=(-30, 0, 0),
                gap=1,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4),
                vis_mode='recon',
            )
        )

        self.ring2 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.008,
                static_friction=0.1,
                kinetic_friction=0.08,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=24,
                radius=0.04,
                axis="y",
                pos=(0.09, -0.27, 0.008),
                euler=(-30, 0, 90),
                gap=1,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4),
                vis_mode='recon',
            )
        )

        self.franka1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.9
            ),
            morph=gs.morphs.URDF(
                file='urdf/panda_bullet/panda.urdf',
                pos=(0.45, -0.6, 0),
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
                pos=(0.7, 0.25, 0),
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
        self.scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)
        print('gripper geom rigstered', self.scene.rod_solver._geom_indices)
        self.scene.build(n_envs=self.n_envs, env_spacing=(10, 10))

        self.control_idx = [3, 22]
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

        init_pos_f2 = self.rope.get_all_verts()[0, self.control_idx[1], :]
        init_pos_f2[2] = 0.013       # a bit above the ground
        self._open_gap = 0.03

        self.c1 = RobotControllerPink(
            self.scene, self.franka1, self._ef1,
            initial_pos=[0.1426, 0.0, 0.033],
            initial_gripper_gap=self._open_gap,
        )
        self.init_gripper_qpos1[self.c1.fingers_dof] = self._open_gap

        self.c2 = RobotControllerPink(
            self.scene, self.franka2, self._ef2,
            initial_pos=init_pos_f2.tolist(),
            initial_gripper_gap=self._open_gap,
        )
        self.init_gripper_qpos2[self.c2.fingers_dof] = 0.012

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(-1.5, 0.9, 0.5), up=(0, 0, 1),
            lookat=(0.3, 0., 0), fov=24, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(-1, -0.8, 1.4), up=(0, 0, 1),
            lookat=(0.2, 0., 0), fov=20, GUI=False
        ))

        self.cameras = cameras

    @staticmethod
    def sigmoid_func(x):
        # Sigmoid function to map values to the range (0, 1)
        return 1 / (1 + np.exp(-x))

    def _reward_sub1(self, verts) -> float:
        # 1. get close to the center of the rings
        dists_ring1 = np.linalg.norm(verts - self.ring1_center, axis=1)
        min_dists_ring1 = np.min(dists_ring1)  # minimum distance to ring1 center

        # 2. encourage the rope to point through the rings
        min_idx_ring1 = np.argmin(dists_ring1)
        # avoid index out of range when accessing verts[min_idx_ring1 + 1]
        min_idx_ring1 = np.minimum(min_idx_ring1, verts.shape[0] - 2)
        dir_rope_ring1 = verts[min_idx_ring1] - verts[min_idx_ring1 + 1]  # direction of the rope at the closest point to ring1
        dir_rope_ring1 = dir_rope_ring1 / (np.linalg.norm(dir_rope_ring1) + 1e-8)  # normalize
        dir_alignment_ring1 = np.dot(dir_rope_ring1, self.ring1_normal)
        score_dir_ring1 = self.sigmoid_func(dir_alignment_ring1 * 5)  # scale to make it more sensitive

        # 3. follow the curve
        # [n_verts, 3]
        target = self.target_pos_sub1
        dists_curve = np.linalg.norm(verts - target, axis=1)

        # combine the rewards
        reward = - np.mean(dists_curve) - 0.1 * np.std(dists_curve)
        reward += - min_dists_ring1 + 0.1 * score_dir_ring1
        return float(reward)

    def _reward_sub2(self, verts) -> float:
        # 1. get close to the center of the rings
        dists_ring1 = np.linalg.norm(verts - self.ring1_center, axis=1)
        min_dists_ring1 = np.min(dists_ring1)  # minimum distance to ring1 center

        dists_ring2 = np.linalg.norm(verts - self.ring2_center, axis=1)
        min_dists_ring2 = np.min(dists_ring2)  # minimum distance to ring2 center

        # 2. encourage the rope to point through the rings
        min_idx_ring1 = np.argmin(dists_ring1)
        # avoid index out of range when accessing verts[min_idx_ring1 + 1]
        min_idx_ring1 = np.minimum(min_idx_ring1, verts.shape[0] - 2)
        dir_rope_ring1 = verts[min_idx_ring1] - verts[min_idx_ring1 + 1]  # direction of the rope at the closest point to ring1
        dir_rope_ring1 = dir_rope_ring1 / (np.linalg.norm(dir_rope_ring1) + 1e-8)  # normalize
        dir_alignment_ring1 = np.dot(dir_rope_ring1, self.ring1_normal)
        score_dir_ring1 = self.sigmoid_func(dir_alignment_ring1 * 5)  # scale to make it more sensitive

        min_idx_ring2 = np.argmin(dists_ring2)
        # avoid index out of range when accessing verts[min_idx_ring2 + 1]
        min_idx_ring2 = np.minimum(min_idx_ring2, verts.shape[0] - 2)
        dir_rope_ring2 = verts[min_idx_ring2] - verts[min_idx_ring2 + 1]  # direction of the rope at the closest point to ring2
        dir_rope_ring2 = dir_rope_ring2 / (np.linalg.norm(dir_rope_ring2) + 1e-8)  # normalize
        dir_alignment_ring2 = np.dot(dir_rope_ring2, self.ring2_normal)
        score_dir_ring2 = self.sigmoid_func(dir_alignment_ring2 * 5)  # scale to make it more sensitive

        # 3. follow the curve
        # [n_verts, 3]
        target = self.target_pos_sub2
        dists_curve = np.linalg.norm(verts - target, axis=1)

        # combine the rewards
        reward = - np.mean(dists_curve) - 0.1 * np.std(dists_curve)
        reward += - min_dists_ring1 - min_dists_ring2 + 0.1 * score_dir_ring1 + 0.1 * score_dir_ring2
        return float(reward)

    def _reward_sub3(self, verts) -> float:
        # 1. get close to the center of the rings
        dists_ring1 = np.linalg.norm(verts - self.ring1_center, axis=1)
        min_dists_ring1 = np.min(dists_ring1)  # minimum distance to ring1 center

        dists_ring2 = np.linalg.norm(verts - self.ring2_center, axis=1)
        min_dists_ring2 = np.min(dists_ring2)  # minimum distance to ring2 center

        # 2. encourage the rope to point through the rings
        min_idx_ring1 = np.argmin(dists_ring1)
        # avoid index out of range when accessing verts[min_idx_ring1 + 1]
        min_idx_ring1 = np.minimum(min_idx_ring1, verts.shape[0] - 2)
        dir_rope_ring1 = verts[min_idx_ring1] - verts[min_idx_ring1 + 1]  # direction of the rope at the closest point to ring1
        dir_rope_ring1 = dir_rope_ring1 / (np.linalg.norm(dir_rope_ring1) + 1e-8)  # normalize
        dir_alignment_ring1 = np.dot(dir_rope_ring1, self.ring1_normal)
        score_dir_ring1 = self.sigmoid_func(dir_alignment_ring1 * 5)  # scale to make it more sensitive

        min_idx_ring2 = np.argmin(dists_ring2)
        # avoid index out of range when accessing verts[min_idx_ring2 + 1]
        min_idx_ring2 = np.minimum(min_idx_ring2, verts.shape[0] - 2)
        dir_rope_ring2 = verts[min_idx_ring2] - verts[min_idx_ring2 + 1]  # direction of the rope at the closest point to ring2
        dir_rope_ring2 = dir_rope_ring2 / (np.linalg.norm(dir_rope_ring2) + 1e-8)  # normalize
        dir_alignment_ring2 = np.dot(dir_rope_ring2, self.ring2_normal)
        score_dir_ring2 = self.sigmoid_func(dir_alignment_ring2 * 5)  # scale to make it more sensitive

        # 3. follow the curve
        # [n_verts, 3]
        target = self.target_pos_sub3
        dists_curve = np.linalg.norm(verts - target, axis=1)

        # combine the rewards
        reward = - np.mean(dists_curve) - 0.1 * np.std(dists_curve)
        reward += - min_dists_ring1 - min_dists_ring2 + 0.1 * score_dir_ring1 + 0.1 * score_dir_ring2
        return float(reward)

    def reward(self):
        # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()
        assert verts_batch.shape[1] == self.target_pos_sub1.shape[0]
        assert verts_batch.shape[1] == self.target_pos_sub2.shape[0]
        assert verts_batch.shape[1] == self.target_pos_sub3.shape[0]

        rewards = []
        for i in range(self.n_envs):
            # [n_verts, 3]
            verts = verts_batch[i]
            env_step = self._env_step_counter[i]

            # TODO: now hack absolute step to determine subtask
            if env_step < 40:
                reward = self._reward_sub1(verts)
            elif env_step < 80:
                reward = self._reward_sub2(verts)
            else:
                reward = self._reward_sub3(verts)
            rewards.append(reward)

        return rewards

    def _loss_sub1(self, verts):
        # verts: [n_verts, 3]
        target = self.target_pos_sub1_tc
        dists = torch.norm(verts - target, dim=1)
        loss = torch.mean(dists) + 0.1 * torch.std(dists)
        return loss

    def _loss_sub2(self, verts):
        # verts: [n_verts, 3]
        target = self.target_pos_sub2_tc
        dists = torch.norm(verts - target, dim=1)
        loss = torch.mean(dists) + 0.1 * torch.std(dists)
        return loss

    def _loss_sub3(self, verts):
        # verts: [n_verts, 3]
        target = self.target_pos_sub3_tc
        dists = torch.norm(verts - target, dim=1)
        loss = torch.mean(dists) + 0.1 * torch.std(dists)
        return loss

    def loss_criterion(self, state):
        # state.pos: [n_envs, n_verts, 3]
        assert state.pos.shape[1] == self.target_pos_sub1_tc.shape[0]
        assert state.pos.shape[1] == self.target_pos_sub2_tc.shape[0]
        assert state.pos.shape[1] == self.target_pos_sub3_tc.shape[0]

        losses = list()
        for i in range(self.n_envs):
            verts = state.pos[i]
            env_step = self._env_step_counter[i]

            if env_step < 40:
                loss = self._loss_sub1(verts)
            elif env_step < 80:
                loss = self._loss_sub2(verts)
            else:
                loss = self._loss_sub3(verts)
            losses.append(loss)

        losses = torch.stack(losses)
        return losses

    def reset(self, envs_idx=None):
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

            self.c1.set_initial_dofs_position(self.init_gripper_qpos1, False, envs_idx=envs_idx)
            self.c2.set_initial_dofs_position(self.init_gripper_qpos2, False, envs_idx=envs_idx)

            self.c1.control_robot(self._open_gap, self._open_gap, envs_idx=envs_idx)
            self.c2.control_robot(-1, -1, g_dof_use_force=True, envs_idx=envs_idx)
            for i in range(30):
                self.scene.step()
                if i % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            self._env_step_counter[np.asarray(envs_idx_)] = 0

        fixed_ring1_np = np.ones((self.n_envs, self.ring1.n_vertices), dtype=bool)
        self.ring1.set_fixed(0, fixed_ring1_np)
        fixed_ring2_np = np.ones((self.n_envs, self.ring2.n_vertices), dtype=bool)
        self.ring2.set_fixed(0, fixed_ring2_np)

    def _check_feasible_transition_sub1_to_sub2(self) -> np.ndarray:
        feasible = torch.ones((self.n_envs,), dtype=torch.bool)

        # check if the 4th vertex has x < ring1 center x
        vert_pos = self.rope.get_all_verts_tc()[:, 4, :]    # (n_envs, 3)
        feasible = feasible & (self.ring1_center_tc[0] - vert_pos[:, 0] > 0.05)

        # check if the 4th vertex is through ring1
        vec_center_to_vert = vert_pos - self.ring1_center_tc   # (n_envs, 3)
        vec_center_to_vert_ = vec_center_to_vert / (torch.norm(vec_center_to_vert, dim=1, keepdim=True) + 1e-8)    # check alignment with ring normal
        alignment = torch.einsum('ij, j -> i', vec_center_to_vert_, self.ring1_normal_tc)
        feasible = feasible & (alignment > 0)
        return feasible.cpu().numpy()
    
    def _check_feasible_transition_sub2_to_sub3(self) -> np.ndarray:
        feasible = torch.ones((self.n_envs,), dtype=torch.bool)

        # check if the 4th vertex has x < ring1 center x
        vert_pos = self.rope.get_all_verts_tc()[:, 4, :]    # (n_envs, 3)
        feasible = feasible & (self.ring1_center_tc[0] - vert_pos[:, 0] > 0.05)

        # check if the 4th vertex is through ring1
        vec_center_to_vert = vert_pos - self.ring1_center_tc   # (n_envs, 3)
        vec_center_to_vert_ = vec_center_to_vert / (torch.norm(vec_center_to_vert, dim=1, keepdim=True) + 1e-8)    # check alignment with ring normal
        alignment = torch.einsum('ij, j -> i', vec_center_to_vert_, self.ring1_normal_tc)
        feasible = feasible & (alignment > 0)

        # check if the 18th vertex has x < ring1 center x
        vert_pos = self.rope.get_all_verts_tc()[:, 18, :]    # (n_envs, 3)
        feasible = feasible & (self.ring1_center_tc[1] - vert_pos[:, 1] > 0.05)

        # check if the 18th vertex is through ring1
        vec_center_to_vert = vert_pos - self.ring1_center_tc   # (n_envs, 3)
        vec_center_to_vert_ = vec_center_to_vert / (torch.norm(vec_center_to_vert, dim=1, keepdim=True) + 1e-8)    # check alignment with ring normal
        alignment = torch.einsum('ij, j -> i', vec_center_to_vert_, self.ring2_normal_tc)
        feasible = feasible & (alignment > 0)
        return feasible.cpu().numpy()

    def _transition_sub1_to_sub2(self, envs_idx: torch.Tensor):
        target_pos = self.rope.get_all_verts_tc()[:, 4]         # fetch the 4th vertex
        target_pos[:, 2] = 0.013

        self.c1.set_robot(
            0.012, 0.012, pos=target_pos, envs_idx=envs_idx, min_z=self._min_z
        )
        self.c2.set_robot(
            self._open_gap, self._open_gap, envs_idx=envs_idx, min_z=self._min_z
        )
        for i in range(10):
            self.scene.step()
            if i % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

        self.c1.control_robot(0, 0, envs_idx=envs_idx, min_z=self._min_z)
        self.c2.control_robot(self._open_gap * 2, self._open_gap * 2, envs_idx=envs_idx, min_z=self._min_z)
        for i in range(80):
            self.scene.step()
            if i % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

        if self.gd_initialized and isinstance(self.c, TrajOptimController):
            # control point changed from 22 to 4
            self.c.grasp_point_ids = [4]

    def _transition_sub2_to_sub3(self, envs_idx: torch.Tensor):
        target_pos = self.rope.get_all_verts_tc()[:, 18]         # fetch the 18th vertex
        target_pos[:, 2] = 0.013

        self.c1.set_robot(
            self._open_gap, self._open_gap, envs_idx=envs_idx, min_z=self._min_z
        )
        self.c2.set_robot(
            0.012, 0.012, pos=target_pos, envs_idx=envs_idx, min_z=self._min_z
        )
        for i in range(10):
            self.scene.step()
            if i % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

        self.c1.control_robot(self._open_gap * 2, self._open_gap * 2, envs_idx=envs_idx, min_z=self._min_z)
        self.c2.control_robot(0, 0, envs_idx=envs_idx, min_z=self._min_z)
        for i in range(80):
            self.scene.step()
            if i % 10 == 0:
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

        if self.gd_initialized and isinstance(self.c, TrajOptimController):
            # control point changed from 4 to 18
            self.c.grasp_point_ids = [18]

    def eval_traj_v3(self, trajs, **kwargs):
        """
        Evaluate trajectories using cumulative reward.
        """
        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 6 == 0 and dof // 6 == n_ctrl, (
            f"dof must be 6 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        n_steps_sub = self._cmaes_n_steps_sub
        if kwargs.get("qpos", None) is None:
            self.qpos_seq = np.zeros((n_steps * n_steps_sub + 1, self.n_envs, len(self.control_idx) * 9))
            self.use_qpos = False
        else:
            self.qpos_seq = kwargs["qpos"]
            self.use_qpos = True

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

        reward_accum = np.zeros((self.n_envs,), dtype=np.float32)

        for i in range(n_steps):
            # Check NaNs BEFORE micro-stepping this macro-step
            verts_rope = self.rope.get_all_verts()  # (n_envs, n_vertices, 3)
            nan_now = np.isnan(verts_rope).any(axis=(1, 2))
            newly_nan = nan_now & alive
            if newly_nan.any():
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

                cur_substep = i * n_steps_sub + j
                c1_open = (cur_substep < 40) | (cur_substep >= 80)

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
                    if c1_open:
                        qpos1 = self.c1.control_robot(
                            self._open_gap, self._open_gap, min_z=self._min_z
                        )
                        qpos2 = self.c2.control_robot(
                            0, 0,
                            dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=self._min_z
                        )
                    else:
                        qpos1 = self.c1.control_robot(
                            0, 0,
                            dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=self._min_z
                        )
                        qpos2 = self.c2.control_robot(
                            self._open_gap, self._open_gap, min_z=self._min_z
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

                # Post-step: detect whether gripper lost the rod
                lost = np.ones((self.n_envs,), dtype=bool)
                for i_b in range(self.n_envs):
                    grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                    c1_retained = False
                    c2_retained = False
                    for k, v in grasp_info.items():
                        if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                            c1_retained = True
                        if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                            c2_retained = True
                    if c1_open:
                        c1_retained = True  # ignore gripper 1 when it is open
                    else:
                        c2_retained = True  # ignore gripper 2 when it is open
                    # lost either gripper
                    lost[i_b] = not (c1_retained and c2_retained)
                newly_lost = lost & alive
                if newly_lost.any():
                    alive[newly_lost] = False

                # Post-step: detect ik convergence
                if hasattr(self.c1, 'convergence'):
                    newly_not_converged = ~self.c1.convergence & alive
                    if newly_not_converged.any():
                        alive[newly_not_converged] = False

                if hasattr(self.c2, 'convergence'):
                    newly_not_converged2 = ~self.c2.convergence & alive
                    if newly_not_converged2.any():
                        alive[newly_not_converged2] = False

                # Post-step: detect NaNs that emerge during micro-stepping
                verts_rope_post = self.rope.get_all_verts()
                nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
                newly_nan_after = nan_after & alive
                if newly_nan_after.any():
                    ever_nan[newly_nan_after] = True
                    alive[newly_nan_after] = False

                self._env_step_counter[alive] += 1

                # Check and perform subtask transitions
                # if not feasible, mark as failed
                not_able_to_trans12 = (cur_substep == 39) & alive & (~self._check_feasible_transition_sub1_to_sub2())
                if not_able_to_trans12.any():
                    alive[not_able_to_trans12] = False

                needs_trans_12 = (cur_substep == 39) & alive
                if needs_trans_12.any():
                    trans_12_idx = np.where(needs_trans_12)[0]
                    trans_12_idx = torch.as_tensor(trans_12_idx)
                    self._transition_sub1_to_sub2(trans_12_idx)

                    # Check whether gripper 1 holds after transition
                    lost = np.ones((self.n_envs,), dtype=bool)
                    for i_b in range(self.n_envs):
                        grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                        c1_retained = False
                        for k, v in grasp_info.items():
                            if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                                c1_retained = True
                        lost[i_b] = not c1_retained
                    newly_lost = lost & alive
                    if newly_lost.any():
                        alive[newly_lost] = False

                # Check and perform subtask transitions
                # if not feasible, mark as failed
                not_able_to_trans23 = (cur_substep == 79) & alive & (~self._check_feasible_transition_sub2_to_sub3())
                if not_able_to_trans23.any():
                    alive[not_able_to_trans23] = False

                needs_trans_23 = (cur_substep == 79) & alive
                if needs_trans_23.any():
                    trans_23_idx = np.where(needs_trans_23)[0]
                    trans_23_idx = torch.as_tensor(trans_23_idx)
                    self._transition_sub2_to_sub3(trans_23_idx)

                    # Check whether gripper 2 holds after transition
                    lost = np.ones((self.n_envs,), dtype=bool)
                    for i_b in range(self.n_envs):
                        grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                        c2_retained = False
                        for k, v in grasp_info.items():
                            if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                                c2_retained = True
                        lost[i_b] = not c2_retained
                    newly_lost = lost & alive
                    if newly_lost.any():
                        alive[newly_lost] = False

                # Collect reward here
                substep_rewards_pre = np.asarray(self.reward(), dtype=np.float32)
                substep_rewards_nan = np.isnan(substep_rewards_pre)

                substep_rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
                failed = ~alive | substep_rewards_nan
                substep_rewards[failed] = 0.0
                substep_rewards[~failed] = np.clip(substep_rewards_pre[~failed] + 1.0, a_min=0.0, a_max=np.inf)
                reward_accum += substep_rewards

        if not self.use_qpos:
            self.qpos_seq = self.qpos_seq.transpose(1, 0, 2)  # (n_envs, n_steps * n_steps_sub + 1, n_dofs)
            self.qpos_seq = self.qpos_seq.astype(np.float32)

        return reward_accum.astype(np.float32)

    def compute_observation(self):
        verts_rope = self.rope.get_all_verts_tc()                   # (n_envs, n_verts, 3)
        obs_rope_pos = verts_rope.reshape(self.n_envs, -1).to(torch.float32)

        vels_rope = self.rope.get_all_vels_tc()                     # (n_envs, n_verts, 3)
        obs_rope_vel = vels_rope.reshape(self.n_envs, -1).to(torch.float32)

        obs_rope = torch.cat([obs_rope_pos, obs_rope_vel], dim=1)

        ef1_pos = self.c1.ef.get_pos().to(torch.float32)
        ef1_quat = self.c1.ef.get_quat().to(torch.float32)
        joint1_qpos = self.c1.robot.get_dofs_position(self.c1.motors_dof).to(torch.float32)
        c1_obs = torch.cat([ef1_pos, ef1_quat, joint1_qpos], dim=1)

        ef2_pos = self.c2.ef.get_pos().to(torch.float32)
        ef2_quat = self.c2.ef.get_quat().to(torch.float32)
        joint2_qpos = self.c2.robot.get_dofs_position(self.c2.motors_dof).to(torch.float32)
        c2_obs = torch.cat([ef2_pos, ef2_quat, joint2_qpos], dim=1)

        obs = torch.cat([obs_rope, c1_obs, c2_obs], dim=1)

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
        action = action * self._act_magnitude
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

        n_steps_sub = self._steps_interval_split
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

            c1_open = (self._env_step_counter < 40) | (self._env_step_counter >= 80)

            convergence_c1 = None
            convergence_c2 = None

            if c1_open.all():
                qpos1 = self.c1.control_robot(
                    self._open_gap, self._open_gap, min_z=self._min_z
                )
                convergence_c1 = self.c1.convergence
                qpos2 = self.c2.control_robot(
                    0, 0,
                    dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=self._min_z
                )
                convergence_c2 = self.c2.convergence
            elif not c1_open.any():
                qpos1 = self.c1.control_robot(
                    0, 0,
                    dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=self._min_z
                )
                convergence_c1 = self.c1.convergence
                qpos2 = self.c2.control_robot(
                    self._open_gap, self._open_gap, min_z=self._min_z
                )
                convergence_c2 = self.c2.convergence
            else:
                c1_open_ids = np.where(c1_open)[0]
                c1_close_ids = np.where(~c1_open)[0]
                qpos1 = torch.empty((self.n_envs, len(self.c1.motors_dof) + len(self.c1.fingers_dof)), dtype=gs.tc_float)
                qpos2 = torch.empty((self.n_envs, len(self.c2.motors_dof) + len(self.c2.fingers_dof)), dtype=gs.tc_float)
                convergence_c1 = np.empty((self.n_envs,), dtype=bool)
                convergence_c2 = np.empty((self.n_envs,), dtype=bool)

                qpos1_open = self.c1.control_robot(
                    self._open_gap, self._open_gap, min_z=self._min_z, envs_idx=c1_open_ids
                )
                convergence_c1[c1_open_ids] = self.c1.convergence[c1_open_ids]
                qpos1_close = self.c1.control_robot(
                    0, 0,
                    dx=dxyz1[:, 0], dy=dxyz1[:, 1], dz=dxyz1[:, 2], di=drot1[:, 0], dj=drot1[:, 1], dk=drot1[:, 2], min_z=self._min_z,
                    envs_idx=c1_close_ids
                )
                convergence_c1[c1_close_ids] = self.c1.convergence[c1_close_ids]

                qpos1[c1_open_ids, :] = qpos1_open
                qpos1[c1_close_ids, :] = qpos1_close

                qpos2_open = self.c2.control_robot(
                    self._open_gap, self._open_gap, min_z=self._min_z, envs_idx=c1_close_ids
                )
                convergence_c2[c1_close_ids] = self.c2.convergence[c1_close_ids]
                qpos2_close = self.c2.control_robot(
                    0, 0,
                    dx=dxyz2[:, 0], dy=dxyz2[:, 1], dz=dxyz2[:, 2], di=drot2[:, 0], dj=drot2[:, 1], dk=drot2[:, 2], min_z=self._min_z,
                    envs_idx=c1_open_ids
                )
                convergence_c2[c1_open_ids] = self.c2.convergence[c1_open_ids]

                qpos2[c1_close_ids, :] = qpos2_open
                qpos2[c1_open_ids, :] = qpos2_close

            for k in range(n_intervals_per_substep):
                self.scene.step()
                if (k + j * n_intervals_per_substep) % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            # Post-step: detect whether gripper lost the rod
            lost = np.ones((self.n_envs,), dtype=bool)
            for i_b in range(self.n_envs):
                grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                c1_retained = False
                c2_retained = False
                for k, v in grasp_info.items():
                    if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                        c1_retained = True
                    if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                        c2_retained = True
                if c1_open[i_b]:
                    c1_retained = True  # ignore gripper 1 when it is open
                else:
                    c2_retained = True  # ignore gripper 2 when it is open
                # lost either gripper
                lost[i_b] = not (c1_retained and c2_retained)
            newly_lost = lost & alive
            if newly_lost.any():
                absorbing[newly_lost] = True
                alive[newly_lost] = False

            # Post-step: detect ik convergence for controller 1
            if convergence_c1 is not None:
                newly_not_converged = ~convergence_c1 & alive
                if newly_not_converged.any():
                    absorbing[newly_not_converged] = True
                    alive[newly_not_converged] = False

            # Post-step: detect ik convergence for controller 2
            if convergence_c2 is not None:
                newly_not_converged2 = ~convergence_c2 & alive
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

        self._env_step_counter[alive] += 1

        # Check and perform subtask transitions
        # if not feasible, mark as failed
        not_able_to_trans12 = (self._env_step_counter == 40) & alive & (~self._check_feasible_transition_sub1_to_sub2())
        if not_able_to_trans12.any():
            absorbing[not_able_to_trans12] = True
            alive[not_able_to_trans12] = False

        needs_trans_12 = (self._env_step_counter == 40) & alive
        if needs_trans_12.any():
            trans_12_idx = np.where(needs_trans_12)[0]
            trans_12_idx = torch.as_tensor(trans_12_idx)
            self._transition_sub1_to_sub2(trans_12_idx)

            # Check whether gripper 1 holds after transition
            lost = np.ones((self.n_envs,), dtype=bool)
            for i_b in range(self.n_envs):
                grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                c1_retained = False
                for k, v in grasp_info.items():
                    if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                        c1_retained = True
                lost[i_b] = not c1_retained
            newly_lost = lost & alive
            if newly_lost.any():
                absorbing[newly_lost] = True
                alive[newly_lost] = False

        # Check and perform subtask transitions
        # if not feasible, mark as failed
        not_able_to_trans23 = (self._env_step_counter == 80) & alive & (~self._check_feasible_transition_sub2_to_sub3())
        if not_able_to_trans23.any():
            absorbing[not_able_to_trans23] = True
            alive[not_able_to_trans23] = False

        needs_trans_23 = (self._env_step_counter == 80) & alive
        if needs_trans_23.any():
            trans_23_idx = np.where(needs_trans_23)[0]
            trans_23_idx = torch.as_tensor(trans_23_idx)
            self._transition_sub2_to_sub3(trans_23_idx)

            # Check whether gripper 2 holds after transition
            lost = np.ones((self.n_envs,), dtype=bool)
            for i_b in range(self.n_envs):
                grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                c2_retained = False
                for k, v in grasp_info.items():
                    if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                        c2_retained = True
                lost[i_b] = not c2_retained
            newly_lost = lost & alive
            if newly_lost.any():
                absorbing[newly_lost] = True
                alive[newly_lost] = False

        # Compute base rewards
        env_rewards = np.asarray(self.reward(), dtype=np.float32)
        env_rewards_nan = np.isnan(env_rewards)

        # Compose final rewards
        rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
        failed = absorbing | env_rewards_nan
        rewards[failed] = 0.0
        rewards[~failed] = np.clip(env_rewards[~failed] + 1.0, a_min=0.0, a_max=np.inf)
        rewards = torch.as_tensor(rewards).reshape((self.n_envs,))
        absorbing = torch.as_tensor(absorbing).reshape((self.n_envs,))

        next_obs = self.compute_observation()

        return next_obs, rewards, absorbing, [{}] * self.n_envs

    def train_one_iter_gd(self, it=None, max_it=None, skip_backward=False):
        self.qpos_seq = np.zeros((self._n_steps + 1, self.n_envs, len(self.control_idx) * 9))
        self.use_qpos = False

        self.reset()

        loss = 0.
        total_horizon = 0
        horizon_ids = list()

        alive = np.ones((self.n_envs,), dtype=bool)
        reward_accum = np.zeros((self.n_envs,), dtype=np.float32)
        
        forward_elapsed = 0.0

        for i in range(self._n_steps):
            local_loss = 0.
            n_horizons = self.steps_interval
            # Do not move already-failed envs
            delta_pos = self.c.pre_apply_grad(stage_idx=i)

            c1_open = (i < 40) | (i >= 80)

            step_qpos = list()
            if c1_open:
                qpos1 = self.c1.control_robot(
                    self._open_gap, self._open_gap, min_z=self._min_z
                )
                qpos2 = self.c2.control_robot(
                    0, 0,
                    dx=delta_pos[:, 0, 0], dy=delta_pos[:, 0, 1], dz=delta_pos[:, 0, 2], min_z=self._min_z
                )
            else:
                qpos1 = self.c1.control_robot(
                    0, 0,
                    dx=delta_pos[:, 0, 0], dy=delta_pos[:, 0, 1], dz=delta_pos[:, 0, 2], min_z=self._min_z
                )
                qpos2 = self.c2.control_robot(
                    self._open_gap, self._open_gap, min_z=self._min_z
                )
            qpos1 = qpos1.cpu().numpy()
            qpos2 = qpos2.cpu().numpy()
            # (n_envs, n_dofs * 2)
            qpos = np.concatenate([qpos1, qpos2], axis=-1)
            step_qpos.append(qpos)
            step_qpos = np.concatenate(step_qpos, axis=-1) # (n_envs, n_ctrl * 9)
            self.qpos_seq[i + 1] = step_qpos

            forward_start_time = time.time()
            for j in range(n_horizons):
                self.scene.step()
                if j % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)
            forward_elapsed += time.time() - forward_start_time

            # Post-step: detect whether gripper lost the rod
            lost = np.ones((self.n_envs,), dtype=bool)
            for i_b in range(self.n_envs):
                grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                c1_retained = False
                c2_retained = False
                for k, v in grasp_info.items():
                    if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                        c1_retained = True
                    if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                        c2_retained = True
                if c1_open:
                    c1_retained = True  # ignore gripper 1 when it is open
                else:
                    c2_retained = True  # ignore gripper 2 when it is open
                # lost either gripper
                lost[i_b] = not (c1_retained and c2_retained)
            newly_lost = lost & alive
            if newly_lost.any():
                alive[newly_lost] = False

            # Post-step: detect NaNs that emerge during micro-stepping
            verts_rope_post = self.rope.get_all_verts()
            nan_after = np.isnan(verts_rope_post).any(axis=(1, 2))
            newly_nan_after = nan_after & alive
            if newly_nan_after.any():
                alive[newly_nan_after] = False

            self._env_step_counter[alive] += 1

            # Check and perform subtask transitions
            # if not feasible, mark as failed
            not_able_to_trans12 = (i == 39) & alive & (~self._check_feasible_transition_sub1_to_sub2())
            if not_able_to_trans12.any():
                alive[not_able_to_trans12] = False

            needs_trans_12 = (i == 39) & alive
            if needs_trans_12.any():
                trans_12_idx = np.where(needs_trans_12)[0]
                trans_12_idx = torch.as_tensor(trans_12_idx)
                self._transition_sub1_to_sub2(trans_12_idx)

                # Check whether gripper 1 holds after transition
                lost = np.ones((self.n_envs,), dtype=bool)
                for i_b in range(self.n_envs):
                    grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                    c1_retained = False
                    for k, v in grasp_info.items():
                        if v == self.gripper_geom_indices[0] or v == self.gripper_geom_indices[1]:
                            c1_retained = True
                    lost[i_b] = not c1_retained
                newly_lost = lost & alive
                if newly_lost.any():
                    alive[newly_lost] = False

            # Check and perform subtask transitions
            # if not feasible, mark as failed
            not_able_to_trans23 = (i == 79) & alive & (~self._check_feasible_transition_sub2_to_sub3())
            if not_able_to_trans23.any():
                alive[not_able_to_trans23] = False

            needs_trans_23 = (i == 79) & alive
            if needs_trans_23.any():
                trans_23_idx = np.where(needs_trans_23)[0]
                trans_23_idx = torch.as_tensor(trans_23_idx)
                self._transition_sub2_to_sub3(trans_23_idx)

                # Check whether gripper 2 holds after transition
                lost = np.ones((self.n_envs,), dtype=bool)
                for i_b in range(self.n_envs):
                    grasp_info = self.scene.sim.coupler.get_rod_rigid_gripper_contact_info(envs_idx=i_b)
                    c2_retained = False
                    for k, v in grasp_info.items():
                        if v == self.gripper_geom_indices[2] or v == self.gripper_geom_indices[3]:
                            c2_retained = True
                    lost[i_b] = not c2_retained
                newly_lost = lost & alive
                if newly_lost.any():
                    alive[newly_lost] = False

            # Compute loss
            state = self.rope.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            loss_c = self.loss_criterion(state) + self.loss_above_plane(state)
            local_loss += loss_c.mean()

            scale = self.scale_array[i]
            loss += scale * local_loss

            self.c.post_check(stage_idx=i, alive=torch.as_tensor(alive))

            # Collect reward here
            substep_rewards_pre = np.asarray(self.reward(), dtype=np.float32)
            substep_rewards_nan = np.isnan(substep_rewards_pre)

            substep_rewards = np.full((self.n_envs,), 0.0, dtype=np.float32)
            failed = ~alive | substep_rewards_nan
            substep_rewards[failed] = 0.0
            substep_rewards[~failed] = np.clip(substep_rewards_pre[~failed] + 1.0, a_min=0.0, a_max=np.inf)
            reward_accum += substep_rewards

        out = dict()
        out['loss'] = loss.item()
        out['reward'] = reward_accum

        backward_elapsed = 0.0

        if not skip_backward:

            backward_start_time = time.time()
            loss.backward()
            backward_elapsed = time.time() - backward_start_time

            for stage_idx, horizon_idx in enumerate(horizon_ids):
                self.c.gather_grad(
                    stage_idx=stage_idx,
                    horizon_idx=horizon_idx + 30,
                    cur_step=it,
                    max_step=max_it,
                    lr=self.lr,
                    lr_min=self.lr_min,
                )
        
        out['forward_time'] = forward_elapsed
        out['backward_time'] = backward_elapsed
        out['lr'] = self.c._lr

        out['qpos_seq'] = self.qpos_seq  # (n_steps, n_envs, n_ctrl * 9)
        return out
