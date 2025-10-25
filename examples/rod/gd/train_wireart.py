# NOTE: assume runs from "examples/rod"

import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('.')
from gd.train_env import Train_Env_GD
from gd.traj_optim import TrajOptim


class Train_GD_Wire_Art(Train_Env_GD):
    def __init__(self, args):
        super().__init__(args)

        # NOTE: assume running from "examples/rod"
        self.target_pos = np.load("target_pos/plasticity_finalpos.npy")
        print(f'Loaded target pos from "plasticity_finalpos.npy", shape = {self.target_pos.shape}')

    def construct_traj_optim(self):
        self.control_idx = [1, 43]

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

        # Euclidean distance from each vertex to the target point
        # (n_envs, n_verts)
        dists = torch.norm(verts_batch - target[None, :, :], dim=2)

        # Loss per env
        loss_dist = torch.mean(dists, dim=1) + 0.1 * torch.std(dists, dim=1)   # (n_envs,)

        return loss_dist

    def reward(self):
        # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()
        assert verts_batch.shape[1] == self.target_pos.shape[0]

        rewards = []
        for i in range(self.args.n_envs):
            # [n_verts, 3]
            target = self.target_pos
            # [n_verts, 3]
            verts = verts_batch[i]
            # [n_verts]
            dists = np.linalg.norm(verts - target, axis=1)

            reward = - np.mean(dists) - 0.1 * np.std(dists)

            rewards.append(reward)

        return rewards

    def reset(self):
        self.scene.reset()

        fixed_np = np.zeros((self.args.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        # Also fix all vertices of the rings
        fixed_b1_np = np.ones((self.args.n_envs, self.b1.n_vertices), dtype=bool)
        self.b1.set_fixed(0, fixed_b1_np)
        fixed_b2_np = np.ones((self.args.n_envs, self.b2.n_vertices), dtype=bool)
        self.b2.set_fixed(0, fixed_b2_np)

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
                E=1e4,
                G=0,
                plastic_yield=0.2,
                plastic_creep=0.9,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=45,
                interval=0.02,
                axis="x",
                pos=(-0.04, 0.0, 0.02),
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
            material=gs.materials.ROD.Base(
                segment_radius=0.006,
                static_friction=0.1,
                kinetic_friction=0.08,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=16,
                radius=0.032,
                axis="y",
                pos=(0.28, 0.0, 0.006),
                euler=(-15, 0, 0),
                gap=1,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4),
                vis_mode='recon',
            )
        )

        self.b2 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.006,
                static_friction=0.1,
                kinetic_friction=0.08,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=16,
                radius=0.032,
                axis="y",
                pos=(0.56, 0.0, 0.006),
                euler=(-15, 0, 0),
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
                res=(1200, 900), pos=(-1.2, 0.8, 1.0), up=(0, 0, 1),
                lookat=(0.6, 0.3, 0), fov=30, GUI=False
            ))
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(-0.2, -1.5, 0.6), up=(0, 0, 1),
                lookat=(0.45, 0.3, 0), fov=30, GUI=False
            ))

        self.cameras = cameras
