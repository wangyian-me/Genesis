# NOTE: assume runs from "examples/rod"

import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('.')
from gd.train_env import Train_Env_GD
from gd.traj_optim import TrajOptim
    

class Train_GD_Wiring_Ring(Train_Env_GD):
    def __init__(self, args):
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
                requires_grad=True,
            ),
            rod_options=gs.options.RodOptions(
                damping=15.0,
                angular_damping=10.0,
                adjacent_gap=3,
                n_pbd_iters=20,
            ),
            show_viewer=args.show_gui,
        )
        super().__init__(args, scene=scene)

        self.ring1_center = np.array([0.27, 0.0, self.rope.material.segment_radius], dtype=gs.np_float)
        self.ring2_center = np.array([0.09, -0.27, self.rope.material.segment_radius], dtype=gs.np_float)
        self.ring1_normal = np.array([-1., 0., 0.], dtype=gs.np_float)
        self.ring2_normal = np.array([0., -1., 0.], dtype=gs.np_float)

        self.ring1_center_tc = torch.tensor([0.27, 0.0, self.rope.material.segment_radius], dtype=gs.tc_float)
        self.ring2_center_tc = torch.tensor([0.09, -0.27, self.rope.material.segment_radius], dtype=gs.tc_float)
        self.ring1_normal_tc = torch.tensor([-1., 0., 0.], dtype=gs.tc_float)
        self.ring2_normal_tc = torch.tensor([0., -1., 0.], dtype=gs.tc_float)

        # NOTE: assume running from "examples/rod"
        self.target_pos = np.load("target_pos/wiring_ring_finalpos.npy")
        print(f'Loaded target pos from "wiring_ring_finalpos.npy", shape = {self.target_pos.shape}')

    def construct_traj_optim(self):
        self.control_idx = [11, 30]

        self.c = TrajOptim(
            self.scene, self.rope,
            grasp_point_ids=self.control_idx,
            n_stages=self.args.n_steps,
            n_optim_dofs=3,
            max_ddist=self.args.max_ddist,
            use_adam=self.args.use_adam,
            debug=self.args.debug,
            lr_scheduler=self.args.lr_scheduler,
        )

    def loss_criterion(self, state):
        # (n_envs, n_verts, 3), torch tensor
        verts_batch = state.pos

        target = torch.tensor(self.target_pos, dtype=verts_batch.dtype, device=verts_batch.device)

        ring1_center = self.ring1_center_tc
        ring2_center = self.ring2_center_tc
        ring1_normal = self.ring1_normal_tc
        ring2_normal = self.ring2_normal_tc

        # 1. get close to the center of the rings
        dists_ring1 = torch.norm(verts_batch - ring1_center[None, None, :], dim=2)
        min_dists_ring1 = torch.min(dists_ring1, dim=1)[0]
        
        dists_ring2 = torch.norm(verts_batch - ring2_center[None, None, :], dim=2)
        min_dists_ring2 = torch.min(dists_ring2, dim=1)[0]

        # 2. encourage the rope to point through the rings
        min_idx_ring1 = torch.argmin(dists_ring1, dim=1)
        # avoid index out of range when accessing verts_batch[:, min_idx_ring1 + 1]
        min_idx_ring1 = torch.minimum(min_idx_ring1, torch.tensor(verts_batch.shape[1] - 2, device=verts_batch.device))
        # direction of the rope at the closest point to ring1
        dir_rope_ring1 = (
            verts_batch[torch.arange(verts_batch.shape[0]), min_idx_ring1] - 
            verts_batch[torch.arange(verts_batch.shape[0]), min_idx_ring1 + 1]
        )
        dir_rope_ring1 = dir_rope_ring1 / (torch.norm(dir_rope_ring1, dim=1, keepdim=True) + 1e-8)  # normalize
        dir_alignment_ring1 = torch.einsum('ij, j -> i', dir_rope_ring1, ring1_normal)
        score_dir_ring1 = torch.sigmoid(dir_alignment_ring1 * 5)

        min_idx_ring2 = torch.argmin(dists_ring2, dim=1)
        # avoid index out of range when accessing verts_batch[:, min_idx_ring2 + 1]
        min_idx_ring2 = torch.minimum(min_idx_ring2, torch.tensor(verts_batch.shape[1] - 2, device=verts_batch.device))
        # direction of the rope at the closest point to ring2
        dir_rope_ring2 = (
            verts_batch[torch.arange(verts_batch.shape[0]), min_idx_ring2] - 
            verts_batch[torch.arange(verts_batch.shape[0]), min_idx_ring2 + 1]
        )
        dir_rope_ring2 = dir_rope_ring2 / (torch.norm(dir_rope_ring2, dim=1, keepdim=True) + 1e-8)
        dir_alignment_ring2 = torch.einsum('ij, j -> i', dir_rope_ring2, ring2_normal)
        score_dir_ring2 = torch.sigmoid(dir_alignment_ring2 * 5)

        # 3. follow the curve
        # (n_envs, n_verts)
        dists_curve = torch.norm(verts_batch - target[None, :, :], dim=2)

        # combine the losses
        loss = 0.5 * torch.mean(dists_curve, dim=1) + 0.05 * torch.std(dists_curve, dim=1)   # (n_envs,)
        loss += min_dists_ring1 + min_dists_ring2 - score_dir_ring1 - score_dir_ring2

        return loss

    @staticmethod
    def sigmoid_func(x):
        # Sigmoid function to map values to the range (0, 1)
        return 1 / (1 + np.exp(-x))

    def reward(self):
        # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()
        assert verts_batch.shape[1] == self.target_pos.shape[0]

        ring1_center = self.ring1_center
        ring2_center = self.ring2_center
        ring1_normal = self.ring1_normal
        ring2_normal = self.ring2_normal

        rewards = []
        for i in range(self.args.n_envs):
            # [n_verts, 3]
            verts = verts_batch[i]

            # 1. get close to the center of the rings
            dists_ring1 = np.linalg.norm(verts - ring1_center, axis=1)
            min_dists_ring1 = np.min(dists_ring1)  # minimum distance to ring1 center

            dists_ring2 = np.linalg.norm(verts - ring2_center, axis=1)
            min_dists_ring2 = np.min(dists_ring2)  # minimum distance to ring2 center

            # 2. encourage the rope to point through the rings
            min_idx_ring1 = np.argmin(dists_ring1)
            # avoid index out of range when accessing verts[min_idx_ring1 + 1]
            min_idx_ring1 = np.minimum(min_idx_ring1, verts.shape[0] - 2)
            dir_rope_ring1 = verts[min_idx_ring1] - verts[min_idx_ring1 + 1]  # direction of the rope at the closest point to ring1
            dir_rope_ring1 = dir_rope_ring1 / (np.linalg.norm(dir_rope_ring1) + 1e-8)  # normalize
            dir_alignment_ring1 = np.dot(dir_rope_ring1, ring1_normal)
            score_dir_ring1 = self.sigmoid_func(dir_alignment_ring1 * 5)  # scale to make it more sensitive

            min_idx_ring2 = np.argmin(dists_ring2)
            # avoid index out of range when accessing verts[min_idx_ring2 + 1]
            min_idx_ring2 = np.minimum(min_idx_ring2, verts.shape[0] - 2)
            dir_rope_ring2 = verts[min_idx_ring2] - verts[min_idx_ring2 + 1]  # direction of the rope at the closest point to ring2
            dir_rope_ring2 = dir_rope_ring2 / (np.linalg.norm(dir_rope_ring2) + 1e-8)  # normalize
            dir_alignment_ring2 = np.dot(dir_rope_ring2, ring2_normal)
            score_dir_ring2 = self.sigmoid_func(dir_alignment_ring2 * 5)  # scale to make it more sensitive

            # 3. follow the curve
            # [n_verts, 3]
            target = self.target_pos
            dists_curve = np.linalg.norm(verts - target, axis=1)

            # combine the rewards
            reward = - np.mean(dists_curve) - 0.1 * np.std(dists_curve)
            reward += - min_dists_ring1 - min_dists_ring2 + score_dir_ring1 + score_dir_ring2
            rewards.append(reward)

        return rewards

    def reset(self):
        self.scene.reset()

        fixed_np = np.zeros((self.args.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        fixed_ring1_np = np.ones((self.args.n_envs, self.ring1.n_vertices), dtype=bool)
        self.ring1.set_fixed(0, fixed_ring1_np)
        fixed_ring2_np = np.ones((self.args.n_envs, self.ring2.n_vertices), dtype=bool)
        self.ring2.set_fixed(0, fixed_ring2_np)

    def construct_scene(self):
        ########################## entities ##########################
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

        self.construct_cameras()

        self.scene.build(n_envs=self.args.n_envs, env_spacing=(1, 1))

    def construct_cameras(self):
        cameras = list()
        if self.args.vis_path is not None:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(-1.6, 1.0, 1.4), up=(0, 0, 1),
                lookat=(0.3, 0., 0), fov=24, GUI=False
            ))
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(-1, -0.8, 1.4), up=(0, 0, 1),
                lookat=(0.2, 0., 0), fov=20, GUI=False
            ))

        self.cameras = cameras
