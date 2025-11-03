# NOTE: assume runs from "examples/rod"

import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('.')
from gd.train_env import Train_Env_GD
from gd.traj_optim import TrajOptim


class Train_GD_Coiling(Train_Env_GD):
    def __init__(self, args):
        super().__init__(args)

    def construct_traj_optim(self):
        self.control_idx = [1]

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
        target = torch.tensor([0.0, 0.0, 0.15], dtype=verts_batch.dtype, device=verts_batch.device)

        # Euclidean distance from each vertex to the target point
        # (n_envs, n_verts)
        dists = torch.norm(verts_batch - target[None, None, :], dim=-1)

        # Loss per env
        loss_dist = dists.sum(dim=1)  # (n_envs,)

        return loss_dist

    def reward(self):
        verts_batch = self.rope.get_all_verts()   # shape: (n_envs, n_verts, 3), NumPy array
        point = np.array([0.0, 0.0, 0.15], dtype=verts_batch.dtype)

        # Euclidean distance from each vertex to the point
        dists = np.linalg.norm(verts_batch - point, axis=-1)  # (n_envs, n_verts)

        # Reward per env: negative sum of distances across all verts
        rewards = -dists.sum(axis=1)  # (n_envs,)

        return rewards.tolist()

    def reset(self):
        self.scene.reset()

        fixed_np = np.zeros((self.args.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

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
                E=1e3,
                G=1e3,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=60,
                interval=0.02,
                axis="x",
                pos=(-0.6, 0.265, 0.01),
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

        self.cone1 = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.7,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/cone.obj",
                pos=(0, 0, 0.15),
                scale=(5, 5, 3),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0),
            ),
        )

        self.construct_cameras()

        self.scene.build(n_envs=self.args.n_envs, env_spacing=(1, 1))

    def construct_cameras(self):
        cameras = list()
        if self.args.vis_path is not None:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(0.5, 2.7, 1.), up=(0, 0, 1),
                lookat=(0., 0., 0.1), fov=30, GUI=False
            ))
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(-1.8, -0.5, 1.8), up=(0, 0, 1),
                lookat=(0., 0., 0.1), fov=30, GUI=False
            ))

        self.cameras = cameras
