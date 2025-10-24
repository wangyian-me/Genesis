import genesis as gs
import torch
import numpy as np
from train_env import Train_Env

class Train_Env_Separation(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, requires_grad=False):
        super().__init__(task, GUI=GUI, camera=camera, n_envs=n_envs, log_dir=log_dir, requires_grad=requires_grad)
        self.steps_interval = 200

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

        self.control_idx = [27]
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

    def loss_criterion(self, state):
        # (n_envs, n_verts, 3), torch tensor
        verts_batch = state.pos
        verts_batch_2 = self.rope2.get_all_verts_tc()

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
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)
