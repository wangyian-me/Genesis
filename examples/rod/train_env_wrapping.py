import genesis as gs
import torch
import numpy as np
from train_env import Train_Env

class Train_Env_Wrapping(Train_Env):
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
                E=1e4,
                G=0,
                use_inextensible=False,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="circle",
                n_vertices=50,
                radius=0.14,
                axis="x",
                pos=(0.65, 0, 0.012),
                euler=(0.0, 0.0, 0.0),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope01.png",
                ),
                vis_mode='recon',
                normal_diff_clamp=1,
            )
        )

        friction_rigid = gs.materials.Rigid(
            needs_coup=True, coup_friction=1.0
        )

        self.c1 = self.scene.add_entity(
            material=friction_rigid,
            morph=gs.morphs.Cylinder(
                radius=0.143,
                height=0.2,
                pos=(0.1, 0., 0.1),
                fixed=True,
            ),
        )

        if camera:
            self.construct_cameras()

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

        self.control_idx = [12, 38]
        self.action_dim = len(self.control_idx) * 3

    def construct_cameras(self):
        cameras = list()
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(3, -1, 1.5), up=(0, 0, 1),
            lookat=(0.65, 0., 0), fov=24, GUI=False
        ))
        cameras.append(self.scene.add_camera(
            res=(1200, 900), pos=(-1, -0.8, 1.4), up=(0, 0, 1),
            lookat=(0.2, 0., 0), fov=30, GUI=False
        ))

        self.cameras = cameras

    def reward(self):
        """
        Encourage the rope to lie on a target circle.

        Args:
            n_samples: number of points to sample on the target circle
            radius:    target circle radius
            center:    (cx, cy, cz_guess) center; z can be overridden via `z`
            z:         plane height for the target circle; defaults to center[2]
            tau:       temperature for soft-min/soft-max (None => exact min/max).
                    Smaller tau => sharper; e.g., tau=0.01 for gentle smoothing.
        Returns:
            list of rewards (length = n_envs), larger is better.
        """

        V = self.rope.get_all_verts()  # (E, N, 3) NumPy
        E, N, _ = V.shape

        cx, cy, z = (0.1, 0, 0)

        S = 20
        angles = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
        circle_pts = np.stack(
            [cx + 0.143 * np.cos(angles),
            cy + 0.143 * np.sin(angles),
            np.full(S, z, dtype=V.dtype)],
            axis=1,  # (S, 3)
        ).astype(V.dtype, copy=False)

        # Distances from each sampled circle point to all rope verts
        # Shapes: circle (E, S, 1, 3), rope (E, 1, N, 3) -> D: (E, S, N)
        circle_b = np.broadcast_to(circle_pts[None, :, None, :], (E, S, 1, 3))
        verts_b  = V[:, None, :, :]
        D = np.linalg.norm(circle_b - verts_b, axis=-1)  # (E, S, N)

        nearest = D.min(axis=2)          # (E, S)
        worst_gap = nearest.max(axis=1)  # (E,)

        rewards = -worst_gap  # minimize worst gap -> maximize reward
        return rewards.tolist()

    def loss_criterion(self, state):
        # (n_envs, n_verts, 3), torch tensor
        verts_batch = state.pos
        E, N, _ = verts_batch.shape

        cx, cy, z = (0.1, 0, 0)

        S = 20
        angles = np.linspace(0.0, 2.0 * np.pi, S, endpoint=False)
        angles = torch.tensor(angles, dtype=verts_batch.dtype, device=verts_batch.device)
        circle_pts = torch.stack(
            [cx + 0.143 * torch.cos(angles),
            cy + 0.143 * torch.sin(angles),
            torch.full((S,), z, dtype=verts_batch.dtype, device=verts_batch.device)],
            dim=1,  # (S, 3)
        )  # (S, 3)

        # Distances from each sampled circle point to all rope verts
        # Shapes: circle (E, S, 1, 3), rope (E, 1, N, 3) -> D: (E, S, N)
        circle_b = circle_pts[None, :, None, :].expand(E, S, 1, 3)
        verts_b  = verts_batch[:, None, :, :].expand(E, S, N, 3)
        D = torch.linalg.norm(circle_b - verts_b, dim=-1)  # (E, S, N)

        nearest = D.min(dim=1).values          # (E, N)
        gap = nearest.mean(dim=1)              # (E,)

        return gap

    def reset(self):
        self.scene.reset()
        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)
