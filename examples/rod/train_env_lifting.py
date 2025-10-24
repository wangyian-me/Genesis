import genesis as gs
import torch
import numpy as np
from train_env import Train_Env

class Train_Env_Lifting(Train_Env):
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

        self.control_idx = [7, 23]
        self.action_dim = len(self.control_idx) * 3

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
        # rope_verts_batch = self.rope.get_all_verts()

        rewards = []
        for i in range(self.n_envs):
            nut_a_pos = nut_a_pos_batch[i]
            nut_b_pos = nut_b_pos_batch[i]

            dist = np.linalg.norm(nut_a_pos - nut_b_pos)
            height = (nut_a_pos[2] + nut_b_pos[2]) / 2.0

            # # [n_verts, 3]
            # rope_verts = rope_verts_batch[i]
            # # check the dist to y axis
            # rope_y = np.mean(np.abs(rope_verts[:, 1]))

            # dist: we want nut a and nut b to be close
            # height: we want the nuts to be lifted up
            # rope_y: we want the rope to be as centered as possible
            reward = - dist + height

            rewards.append(reward)

        return rewards

    def reset(self):
        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)
