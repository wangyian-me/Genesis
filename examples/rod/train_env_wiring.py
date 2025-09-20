import genesis as gs
import imageio
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os 
import json
import matplotlib.pyplot as plt
from train_env import Train_Env
from ring_crossing_helper import ring_crossing_count_axis_aligned, ring_center_from_axis_aligned_vertices, closest_distance_rope_to_point

class Train_Env_Wiring(Train_Env):
    def __init__(self, task='wiring', log_dir="xxx/wiring", n_envs=5):
        super().__init__(task, n_envs=n_envs, log_dir=log_dir)

    def construct_scene(self):
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
                K=1e5,
                E=1e3,
                G=1e3,
                use_inextensible=False
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=60,
                interval=0.01,
                axis="x",
                pos=(0.3, 0.0, 0.02),
                euler=(0, 0, 0),
            ),
            surface=gs.surfaces.Default(
                # color=(0.4, 1.0, 0.4),
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="textures/rope01.png",
                ),
                vis_mode='recon',
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

        self.scene.rod_solver.register_gripper_geom_indices([])

        self.scene.build(n_envs=self.n_envs, env_spacing=(1, 1))

    def reward(self):
        verts_rode_batch = self.rope.get_all_verts()
        verts_ring1_batch = self.ring1.get_all_verts()
        verts_ring2_batch = self.ring2.get_all_verts()

        rewards = []
        for i in range(self.n_envs):
            verts_rode = verts_rode_batch[i]
            verts_ring1 = verts_ring1_batch
            verts_ring2 = verts_ring2_batch
            c1, hits = ring_crossing_count_axis_aligned(verts_rode, verts_ring1, 1e-6)
            c2, hits = ring_crossing_count_axis_aligned(verts_rode, verts_ring2, 1e-6)

            C1 = ring_center_from_axis_aligned_vertices(verts_ring1)
            C2 = ring_center_from_axis_aligned_vertices(verts_ring2)
            min_dist1, seg_idx, t, closest_pt = closest_distance_rope_to_point(verts_rode, C1)
            min_dist2, seg_idx, t, closest_pt = closest_distance_rope_to_point(verts_rode, C2)

            reward = 0
            if c1 % 2 == 1:
                reward += 1
            if c2 % 2 == 1:
                reward += 1

            reward -= min_dist1 + min_dist2

            rewards.append(reward)

        return rewards
    
    def step(self, actions):
        raise NotImplementedError()
        # to be done

    def eval_traj(self, trajs):
        # trajs should be (n_envs, n_steps, x)
        self.scene.reset()
        fixed_np = np.zeros((self._n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, 0] = True
        self.rope.set_fixed(fixed_np)
        steps_interval = 50

        for i in range(trajs.shape[1]):
            verts_rode = self.rope.get_all_verts()
            delta = trajs[:, i]
            current_pos = verts_rode[:, 0]
            for j in range(steps_interval):
                target_pos = current_pos + delta * (j + 1) / steps_interval
                self.rope.set_pos_single(target_pos)
                self.scene.step()
