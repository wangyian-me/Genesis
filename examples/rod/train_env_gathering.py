import genesis as gs
import torch
import numpy as np
from train_env import Train_Env

class Train_Env_Gathering(Train_Env):
    def __init__(self, task='wiring', GUI=False, camera=False, log_dir="xxx/wiring", n_envs=5, requires_grad=False):
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
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.2, -0.5, -0.1),
                upper_bound=(0.8, 0.5, 0.9),
                grid_density=100,
            ),
            rod_options=gs.options.RodOptions(
                damping=15.0,
                angular_damping=10.0,
                n_pbd_iters=20,
            ),
            show_viewer=GUI,
        )
        super().__init__(task, scene=scene, GUI=GUI, camera=camera, n_envs=n_envs, log_dir=log_dir, requires_grad=requires_grad)
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
                segment_mass=0.01,
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
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        self.sphere = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
                E=1e5,
                nu=0.3,
                von_mises_yield_stress=1e3,
            ),
            morph=gs.morphs.Sphere(
                radius=0.05,
                pos=(0.25, 0.02, 0.07),
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
                scale=0.1,
                pos=(0.5, -0.05, 0.07),
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
                radius=0.04,
                height=0.12,
                pos=(0.08, -0.08, 0.1),
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
            res=(1200, 900), pos=(0.4, 2., 0.7), up=(0, 0, 1),
            lookat=(0.25, 0., 0), fov=30, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(1.5, 0.75, 1.5), up=(0, 0, 1),
            lookat=(0.25, -0.1, 0.), fov=30, GUI=False
        ))

        self.cameras = cameras

    def reward(self):

        verts_batch = self.rope.get_all_verts_tc()  # shape: (n_envs, n_vertices, 3)

        pos1 = self.sphere.get_particles()    # shape: (n_envs, n_particles, 3)
        pos1 = np.mean(pos1, axis=1)          # shape: (n_envs, 3)
        pos1 = torch.tensor(pos1, dtype=verts_batch.dtype)
        pos2 = self.bunny.get_pos()     # shape: (n_envs, 3)
        pos3 = self.cylinder.get_pos()  # shape: (n_envs, 3)

        d12 = torch.norm(pos1 - pos2, dim=1)  # ||p1 - p2||
        d23 = torch.norm(pos2 - pos3, dim=1)  # ||p2 - p3||
        d13 = torch.norm(pos1 - pos3, dim=1)  # ||p1 - p3||

        # rewards = -(d12 + d23 + d13)          # negative sum of pairwise distances

        rewards = []

        for i in range(self.n_envs):
            verts = verts_batch[i]  # (n_vertices, 3)
            verts_to_pos1 = torch.norm(verts - pos1[i], dim=1)  # (n_vertices,)
            verts_to_pos2 = torch.norm(verts - pos2[i], dim=1)
            verts_to_pos3 = torch.norm(verts - pos3[i], dim=1)

            d12_i = d12[i]
            d23_i = d23[i]
            d13_i = d13[i]

            reward = -(d12_i + d23_i + d13_i)          # negative sum of pairwise distances
            reward -= torch.min(verts_to_pos1) * 0.1   # negative min distance from rope to pos1
            reward -= torch.min(verts_to_pos2) * 0.1   # negative min distance from rope to pos2
            reward -= torch.min(verts_to_pos3) * 0.1   # negative min distance from rope to pos3

            rewards.append(reward.item())

        return rewards             # list[float] of length n_envs

    def reset(self):
        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)
