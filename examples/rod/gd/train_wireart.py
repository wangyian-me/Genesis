# NOTE: assume runs from "examples/rod"

import argparse
import mediapy
import os
import json
import torch
import numpy as np
import genesis as gs
import sys
sys.path.append('.')
from gd.traj_optim import (
    TrajOptim,
    create_linear_array,
    create_exp_array,
    create_custom_array,
)
from collections import defaultdict


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--steps_interval', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_ddist', type=float, default=0.002)
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--exp_base', type=float, default=1.1)
    parser.add_argument('--scale_method', type=str, default=None,
                        choices=[None, 'linear', 'exp', 'custom'])
    parser.add_argument('--show_gui', action='store_true')
    parser.add_argument('--vis_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs/wireart_gd')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


class Train_GD_Wire_Art:
    def __init__(self, args):
        self.args = args

        ########################## init ##########################
        gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)

        ########################## create a scene ##########################
        viewer_options = gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        )

        self.scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(
                dt=1e-3,
                substeps=5,
                requires_grad=True,
                # gravity=(0.,0.,0.)
            ),
            rod_options=gs.options.RodOptions(
                damping=15.0,
                angular_damping=10.0,
                n_pbd_iters=20,
            ),
            show_viewer=args.show_gui,
        )

        self.construct_scene()
        self.construct_cameras()

        self.control_idx = [1, 43]

        self.c = TrajOptim(
            self.scene, self.rope,
            grasp_point_ids=self.control_idx,
            n_stages=args.n_steps,
            n_optim_dofs=3,
            max_ddist=args.max_ddist,
            use_adam=args.use_adam,
            debug=args.debug,
        )

        scale_method = args.scale_method
        if scale_method is None:
            scale_array = torch.ones(args.n_steps, dtype=gs.tc_float)
            self.scale_array = scale_array / args.n_steps
            print(f'Using uniform scale array:\n{self.scale_array}')
        elif scale_method == 'linear':
            self.scale_array = create_linear_array(args.n_steps)
            print(f'Using linear scale array:\n{self.scale_array}')
        elif scale_method == 'exp':
            self.scale_array = create_exp_array(args.n_steps, base=args.exp_base)
            print(f'Using exponential scale array (base={args.exp_base}):\n{self.scale_array}')
        elif scale_method == 'custom':
            self.scale_array = create_custom_array(args.n_steps)
            print(f'Using custom scale array:\n{self.scale_array}')
        else:
            raise ValueError(f'Unknown scale method: {scale_method}')

        # log
        self.log_dir = args.log_dir
        if os.path.exists(self.log_dir):
            latest_iter = -1
            for i in os.listdir(self.log_dir):
                # find the latest traj
                if i.endswith('_traj.pt') and i != 'best_traj.pt':
                    iter_id = int(i.split('_')[0])
                    if iter_id > latest_iter:
                        latest_iter = iter_id
            if latest_iter >= 0:
                self.iter_start = latest_iter + 1
                traj_path = os.path.join(self.log_dir, f'{latest_iter:03d}_traj.pt')
                self.c.traj = torch.load(traj_path).to(self.c.traj.device)
                print(f'Loaded existing traj from {traj_path}')
                if args.use_adam:
                    adam_path = os.path.join(self.log_dir, 'adam_state.pt')
                    if os.path.exists(adam_path):
                        adam_state = torch.load(adam_path)
                        self.c.m_buffer = adam_state['m_buffer'].to(self.c.traj.device)
                        self.c.v_buffer = adam_state['v_buffer'].to(self.c.traj.device)
                        self.c.iter = adam_state['iter']
                        print(f'Loaded existing Adam state from {adam_path}, iter={self.c.iter}')
                    else:
                        print(f'No existing Adam state found at {adam_path}. Starting Adam fresh.')
                
                print(f'Resumed from existing log dir: {self.log_dir}, will start from iter {self.iter_start}')
            else:
                self.iter_start = 0
                print(f'No existing traj found in {self.log_dir}. Starting from scratch.')

        else:
            os.makedirs(self.log_dir, exist_ok=True)
            self.iter_start = 0
            print(f'No existing log dir found. Created new: {self.log_dir}')
        with open(os.path.join(self.log_dir, 'args.txt'), 'w') as f:
            f.write(str(args) + '\n')

        print(args)
        print(f'Iter from {self.iter_start} to {self.args.n_iters-1}, each iter has {self.args.n_steps}x{self.args.steps_interval}={self.args.n_steps * self.args.steps_interval} steps')
        print(f'Max moving distance {self.args.max_ddist}x{self.args.n_steps}={self.args.max_ddist * self.args.n_steps} m for each control point')

        # NOTE: assume running from "examples/rod"
        self.target_pos = np.load("target_pos/plasticity_finalpos.npy")
        print(f'Loaded target pos from "plasticity_finalpos.npy", shape = {self.target_pos.shape}')

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

    def loss_above_plane(self, state):
        # Required loss to make sure the vertices above the plane
        verts_batch = state.pos
        loss_abv_plane = torch.relu(
            self.rope.material.segment_radius - verts_batch[:, :, 2]
        ).sum(dim=1)                    # (n_envs,)

        return loss_abv_plane

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

    def train_one_iter(self):
        total_horizon = 0
        horizon_ids = list()
        self.scene.reset()

        fixed_np = np.zeros((self.args.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        # Also fix all vertices of the rings
        fixed_b1_np = np.ones((self.args.n_envs, self.b1.n_vertices), dtype=bool)
        self.b1.set_fixed(0, fixed_b1_np)
        fixed_b2_np = np.ones((self.args.n_envs, self.b2.n_vertices), dtype=bool)
        self.b2.set_fixed(0, fixed_b2_np)

        loss = 0.

        for i in range(self.args.n_steps):
            local_loss = 0.
            n_horizons = self.args.steps_interval
            hpos, _ = self.c.pre_apply_grad(stage_idx=i, num_horizons=n_horizons)
            for j in range(n_horizons):
                self.c.on_apply_grad(hpos[j])
                self.scene.step()
                for cid, cam in enumerate(self.cameras):
                    img = cam.render()[0]
                    self.frames[cid].append(img)

            state = self.rope.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            scale = self.scale_array[i]

            local_loss += self.loss_criterion(state).mean()
            local_loss += self.loss_above_plane(state).mean()

            loss += scale * local_loss

        loss.backward()

        for stage_idx, horizon_idx in enumerate(horizon_ids):
            self.c.gather_grad(
                stage_idx=stage_idx,
                horizon_idx=horizon_idx,
                lr=self.args.lr,
            )

        out = dict()
        out['loss'] = loss.item()
        out['reward'] = self.reward()

        return out

    def train(self):
        info = dict()
        max_reward = -float('inf')

        for it in range(self.iter_start, self.args.n_iters):
            out = self.train_one_iter()
            print(f'Iter {it}: loss={out["loss"]:.6f}')
            info[it] = out

            iter_max_reward = max(out['reward'])
            if iter_max_reward > max_reward:
                max_reward = iter_max_reward
                torch.save(self.c.traj.clone().cpu(), os.path.join(self.log_dir, 'best_traj.pt'))
                print(f'  New best traj saved with reward {max_reward:.6f}')

            torch.save(self.c.traj.clone().cpu(), os.path.join(self.log_dir, f'{it:03d}_traj.pt'))

            if self.args.use_adam:
                torch.save({
                    'm_buffer': self.c.m_buffer,
                    'v_buffer': self.c.v_buffer,
                    'iter': self.c.iter,
                }, os.path.join(self.log_dir, 'adam_state.pt'))

            # log
            # write csv
            with open(os.path.join(self.log_dir, 'train_log.csv'), 'a') as f:
                if it == 0 and self.iter_start == 0:
                    f.write('iter,loss,reward\n')
                f.write(f'{it},{info[it]["loss"]:.6f},{max(info[it]["reward"]):.6f}\n')

        if self.args.vis_path is not None:
            for cid in self.frames:
                mediapy.write_video(
                    self.args.vis_path + f'_cam{cid}.mp4',
                    self.frames[cid],
                    fps=60,
                )

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
            surface=gs.surfaces.Default(
                color=(0.4, 1.0, 0.4),
                vis_mode='recon',
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

        self.scene.rod_solver.register_gripper_geom_indices([])

        self.scene.build(n_envs=self.args.n_envs, env_spacing=(1, 1))

    def construct_cameras(self):
        cameras = list()
        if self.args.vis_path is not None:
            cameras.append(self.scene.add_camera(
                res=(600, 450), pos=(-1.2, 0.8, 1.0), up=(0, 0, 1),
                lookat=(0.6, 0.3, 0), fov=30, GUI=False
            ))
            cameras.append(self.scene.add_camera(
                res=(600, 450), pos=(-0.2, -1.5, 0.6), up=(0, 0, 1),
                lookat=(0.45, 0.3, 0), fov=30, GUI=False
            ))

        self.cameras = cameras
        self.frames = defaultdict(list)

def main():
    args = arg_parser()

    trainer = Train_GD_Wire_Art(args)
    trainer.train()


if __name__ == '__main__':
    main()
