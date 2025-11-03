# NOTE: assume runs from "examples/rod"

import os
import time
import torch
import mediapy
import numpy as np
import genesis as gs
import sys
sys.path.append('.')
from gd.train_env import Train_Env_GD
from gd.traj_optim import (
    TrajOptim,
    create_linear_array,
    create_exp_array,
    create_custom_array,
)
from collections import defaultdict


class Train_GD_Separation(Train_Env_GD):
    def __init__(self, args):
        self.args = args

        ########################## init ##########################
        gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

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

        self.cameras = list()
        self.frames = defaultdict(list)
        self.construct_scene()

        # define 1. control_idx, 2. traj_optim
        self.construct_traj_optim()

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
        self.ckpt_dir = os.path.join(args.log_dir, "ckpts")
        if os.path.exists(self.ckpt_dir):
            latest_iter = -1
            for i in os.listdir(self.ckpt_dir):
                # find the latest traj
                if i.endswith('_traj.pt') and i != 'best_traj.pt':
                    iter_id = int(i.split('_')[0])
                    if iter_id > latest_iter:
                        latest_iter = iter_id
            if latest_iter >= 0:
                self.iter_start = latest_iter + 1
                traj_path = os.path.join(self.ckpt_dir, f'{latest_iter:03d}_traj.pt')
                traj = torch.load(traj_path)
                # split traj into c1 and c2
                traj_c1 = traj[:, :, :1, :].clone()
                traj_c2 = traj[:, :, 1:, :].clone()
                self.c1.traj = traj_c1.to(self.c1.traj.device)
                self.c2.traj = traj_c2.to(self.c2.traj.device)
                print(f'Loaded existing traj from {traj_path}')
                if args.use_adam:
                    adam_path = os.path.join(self.ckpt_dir, 'adam_state_1.pt')
                    if os.path.exists(adam_path):
                        adam_state = torch.load(adam_path)
                        self.c1.m_buffer = adam_state['m_buffer'].to(self.c1.traj.device)
                        self.c1.v_buffer = adam_state['v_buffer'].to(self.c1.traj.device)
                        print(f'Loaded existing Adam state from {adam_path}. Previous ends at iter {adam_state["cur_iter"]}.')
                    else:
                        print(f'No existing Adam state found at {adam_path}. Starting Adam fresh.')

                    adam_path = os.path.join(self.ckpt_dir, 'adam_state_2.pt')
                    if os.path.exists(adam_path):
                        adam_state = torch.load(adam_path)
                        self.c2.m_buffer = adam_state['m_buffer'].to(self.c2.traj.device)
                        self.c2.v_buffer = adam_state['v_buffer'].to(self.c2.traj.device)
                        print(f'Loaded existing Adam state from {adam_path}. Previous ends at iter {adam_state["cur_iter"]}.')
                    else:
                        print(f'No existing Adam state found at {adam_path}. Starting Adam fresh.')

                print(f'Resumed from existing ckpt dir: {self.ckpt_dir}, will start from iter {self.iter_start}')
            else:
                self.iter_start = 0
                print(f'No existing traj found in {self.ckpt_dir}. Starting from scratch.')

        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.iter_start = 0
            print(f'No existing ckpt dir found. Created new: {self.ckpt_dir}')
        with open(os.path.join(self.log_dir, 'args.txt'), 'w') as f:
            f.write(str(args) + '\n')

        print(args)
        print(f'Iter from {self.iter_start} to {self.args.n_iters-1}, each iter has {self.args.n_steps}x{self.args.steps_interval}={self.args.n_steps * self.args.steps_interval} steps')
        print(f'Max moving distance {self.args.max_ddist}x{self.args.n_steps}={self.args.max_ddist * self.args.n_steps} m for each control point')

    def construct_traj_optim(self):
        self.control_idx = [27, 2]

        self.c1 = TrajOptim(
            self.scene, self.rope,
            grasp_point_ids=[self.control_idx[0]],
            n_stages=self.args.n_steps,
            n_optim_dofs=3,
            max_ddist=self.args.max_ddist,
            use_adam=self.args.use_adam,
            debug=self.args.debug,
            lr_scheduler=self.args.lr_scheduler,
        )

        self.c2 = TrajOptim(
            self.scene, self.rope2,
            grasp_point_ids=[self.control_idx[1]],
            n_stages=self.args.n_steps,
            n_optim_dofs=3,
            max_ddist=self.args.max_ddist,
            use_adam=self.args.use_adam,
            debug=self.args.debug,
            lr_scheduler=self.args.lr_scheduler,
        )

    def loss_criterion(self, state, state2):
        # (n_envs, n_verts, 3), torch tensor
        verts_batch = state.pos
        verts_batch_2 = state2.pos

        diff = verts_batch[:, :, None, :] - verts_batch_2[:, None, :, :]   # (n_envs, n_verts_A, n_verts_B, 3)
        D = torch.norm(diff, dim=-1)                                       # (n_envs, n

        a_to_b_min = D.min(dim=2).values  # (n_envs, n_verts_A)
        b_to_a_min = D.min(dim=1).values  # (n_envs, n_verts_B)

        loss_chamfer = a_to_b_min.mean(dim=1) + b_to_a_min.mean(dim=1)  # (n_envs,)
        loss_chamfer = - loss_chamfer      # want to maximize chamfer distance

        return loss_chamfer

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

    def reset(self):
        self.scene.reset()

        fixed_np = np.zeros((self.args.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx[0]] = True
        self.rope.set_fixed(0, fixed_np)

        fixed_np2 = np.zeros((self.args.n_envs, self.rope2.n_vertices), dtype=bool)
        fixed_np2[:, self.control_idx[1]] = True
        self.rope2.set_fixed(0, fixed_np2)

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

        self.construct_cameras()

        self.scene.build(n_envs=self.args.n_envs, env_spacing=(1, 1))

    def construct_cameras(self):
        cameras = list()
        if self.args.vis_path is not None:
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(0.5, 1.5, 1.), up=(0, 0, 1),
                lookat=(0.3, 0., 0), fov=30, GUI=False
            ))
            cameras.append(self.scene.add_camera(
                res=(1200, 900), pos=(0.5, -1.5, 1.), up=(0, 0, 1),
                lookat=(0.3, 0., 0), fov=30, GUI=False
            ))

        self.cameras = cameras

    def train_one_iter(self, it=None, max_it=None):
        start_time = time.time()
        total_horizon = 0
        horizon_ids = list()

        # reset
        self.reset()

        loss = 0.
        local_best_reward = -float('inf')
        local_best_step = -1

        for i in range(self.args.n_steps):
            local_loss = 0.
            n_horizons = self.args.steps_interval
            hpos1, _ = self.c1.pre_apply_grad(stage_idx=i, num_horizons=n_horizons)
            hpos2, _ = self.c2.pre_apply_grad(stage_idx=i, num_horizons=n_horizons)

            for j in range(n_horizons):
                self.c1.on_apply_grad(hpos1[j])
                self.c2.on_apply_grad(hpos2[j])
                self.scene.step()
                if j % 10 == 0:
                    for cid, cam in enumerate(self.cameras):
                        img = cam.render()[0]
                        self.frames[cid].append(img)

            state = self.rope.get_state()
            state2 = self.rope2.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            scale = self.scale_array[i]

            local_loss += self.loss_criterion(state, state2).mean()
            local_loss += self.loss_above_plane(state).mean()
            local_loss += self.loss_above_plane(state2).mean()

            loss += scale * local_loss

            r_ = self.reward()
            if max(r_) > local_best_reward:
                local_best_reward = max(r_)
                local_best_step = i

        out = dict()
        out['loss'] = loss.item()
        out['reward'] = self.reward()
        out['local_best_reward'] = local_best_reward
        out['local_best_step'] = local_best_step

        loss.backward()

        for stage_idx, horizon_idx in enumerate(horizon_ids):
            self.c1.gather_grad(
                stage_idx=stage_idx,
                horizon_idx=horizon_idx,
                cur_step=it,
                max_step=max_it,
                lr=self.args.lr,
                lr_min=self.args.lr_min,
            )
            self.c2.gather_grad(
                stage_idx=stage_idx,
                horizon_idx=horizon_idx,
                cur_step=it,
                max_step=max_it,
                lr=self.args.lr,
                lr_min=self.args.lr_min,
            )

        out['iter_time'] = time.time() - start_time

        return out

    def train(self):
        info = dict()
        max_reward = -float('inf')

        for it in range(self.iter_start, self.args.n_iters):
            out = self.train_one_iter(it=it, max_it=self.args.n_iters)
            print(f'Iter {it}: loss={out["loss"]:.6f} elapsed={out["iter_time"]:.1f}s reward={max(out["reward"]):.6f}')
            info[it] = out

            iter_max_reward = max(out['reward'])
            c1 = self.c1.traj.clone().cpu()
            c2 = self.c2.traj.clone().cpu()
            # (n_envs, n_steps, 2, 3)
            c_ = torch.cat([c1, c2], dim=2)
            if iter_max_reward > max_reward:
                max_reward = iter_max_reward
                torch.save(c_, os.path.join(self.log_dir, 'best_traj.pt'))
                c_np = c_[0].reshape(self.args.n_steps, len(self.control_idx) * 3).numpy()
                np.save(os.path.join(self.log_dir, 'best_traj.npy'), c_np)
                print(f'  New best traj saved with final reward {max_reward:.6f}')

            torch.save(c_, os.path.join(self.ckpt_dir, f'{it:03d}_traj.pt'))

            if self.args.use_adam:
                torch.save({
                    'm_buffer': self.c1.m_buffer,
                    'v_buffer': self.c1.v_buffer,
                    'cur_iter': it,
                }, os.path.join(self.ckpt_dir, 'adam_state_1.pt'))

                torch.save({
                    'm_buffer': self.c2.m_buffer,
                    'v_buffer': self.c2.v_buffer,
                    'cur_iter': it,
                }, os.path.join(self.ckpt_dir, 'adam_state_2.pt'))

            # log
            # write csv
            with open(os.path.join(self.log_dir, 'train_log.csv'), 'a') as f:
                if it == 0 and self.iter_start == 0:
                    f.write('iter,loss,reward,best_reward_could_achieve,best_step,iter_time\n')
                f.write(f'{it},{info[it]["loss"]:.6f},{max(info[it]["reward"]):.6f},{info[it]["local_best_reward"]:.6f},{info[it]["local_best_step"]},{info[it]["iter_time"]:.1f}\n')

        if self.args.vis_path is not None:
            for cid in self.frames:
                mediapy.write_video(
                    self.args.vis_path + f'_cam{cid}.mp4',
                    self.frames[cid],
                    fps=30, qp=18
                )