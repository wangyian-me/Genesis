import genesis as gs
import torch
import numpy as np
from train_env import Train_Env

class Train_Env_Gathering(Train_Env):
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
                K=1e5,
                E=1e3,
                G=1e3,
                use_inextensible=False
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=45,
                interval=0.02,
                axis="x",
                pos=(-0.15, 0.1, 0.02),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.4, 1.0, 0.4),
                vis_mode='recon',
            )
        )

        self.sphere = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.3,
            ),
            morph=gs.morphs.Sphere(
                radius=0.05,
                pos=(0.3, -0.07, 0.05),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.51, 0.77, 0.75)
            )
        )

        self.bunny = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.3,
            ),
            morph=gs.morphs.Mesh(
                file="meshes/bunny.obj",
                scale=0.2,
                pos=(0.6, -0.3, 0.1),
            ),
            surface=gs.surfaces.Default(
                color=(0., 0.42, 0.47)
            )
        )

        self.cylinder = self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True, coup_friction=0.3,
            ),
            morph=gs.morphs.Cylinder(
                radius=0.08,
                height=0.12,
                pos=(0.2, -0.4, 0.06),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                color=(0.93, 0.96, 0.98)
            )
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [1, 43]
        self.action_dim = len(self.control_idx) * 3

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(3, -1, 1.5), up=(0, 0, 1),
            lookat=(0.3, -0.1, 0), fov=30, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(-1.5, -1, 1.4), up=(0, 0, 1),
            lookat=(0.3, -0.1, 0.), fov=30, GUI=False
        ))

        self.cameras = cameras

    def reward(self):


        pos1 = self.sphere.get_pos()    # shape: (n_envs, 3)
        pos2 = self.bunny.get_pos()     # shape: (n_envs, 3)
        pos3 = self.cylinder.get_pos()  # shape: (n_envs, 3)

        d12 = torch.norm(pos1 - pos2, dim=1)  # ||p1 - p2||
        d23 = torch.norm(pos2 - pos3, dim=1)  # ||p2 - p3||
        d13 = torch.norm(pos1 - pos3, dim=1)  # ||p1 - p3||

        rewards = -(d12 + d23 + d13)          # negative sum of pairwise distances
        return rewards.detach().cpu().tolist()                # list[float] of length n_envs

    def reset(self):
        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)
