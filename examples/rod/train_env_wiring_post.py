import genesis as gs
import torch
import numpy as np
from train_env import Train_Env

class Train_Env_Wiring_post(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, requires_grad=False):
        super().__init__(task, GUI=GUI, camera=camera, n_envs=n_envs, log_dir=log_dir, requires_grad=requires_grad)
        self.steps_interval = 200

        # NOTE: assume running from "examples/rod"
        self.target_pos = np.load("target_pos/wiring_post_finalpos.npy")
        print(f'Loaded target pos from "wiring_post_finalpos.npy", shape = {self.target_pos.shape}')

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

        self.stick1 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.02,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=4,
                interval=0.02,
                axis="z",
                pos=(0.245, 0.14, -0.02),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        self.stick2 = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=0.02,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=4,
                interval=0.02,
                axis="z",
                pos=(0.1, 0.275, -0.02),
                euler=(0, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 0.4, 0.4)
            )
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [5, 35]
        self.action_dim = len(self.control_idx) * 3

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(0.2, 1.2, 1.5), up=(0, 0, 1),
            lookat=(0.3, 0.2, 0), fov=30, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(-1.6, -1.2, 1.5), up=(0, 0, 1),
            lookat=(0.3, 0.2, 0), fov=30, GUI=False
        ))

        self.cameras = cameras

    def reward(self):
        # [n_envs, n_verts, 3]
        verts_batch = self.rope.get_all_verts()
        assert verts_batch.shape[1] == self.target_pos.shape[0]

        rewards = []
        for i in range(self.n_envs):
            # [n_verts, 3]
            target = self.target_pos
            # [n_verts, 3]
            verts = verts_batch[i]
            # [n_verts]
            dists = np.linalg.norm(verts - target, axis=1)

            reward = - np.mean(dists) - 0.1 * np.std(dists)

            rewards.append(reward)

        return rewards

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

    def reset(self):
        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        fixed_stick1_np = np.ones((self.n_envs, self.stick1.n_vertices), dtype=bool)
        self.stick1.set_fixed(0, fixed_stick1_np)
        fixed_stick2_np = np.ones((self.n_envs, self.stick2.n_vertices), dtype=bool)
        self.stick2.set_fixed(0, fixed_stick2_np)
