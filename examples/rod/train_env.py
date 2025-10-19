import genesis as gs
import imageio
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import os 
import json
import matplotlib.pyplot as plt
from gd.traj_optim_cmaes import (
    create_linear_array,
    create_exp_array,
    create_custom_array,
    TrajOptimCMAES
)


class Train_Env():
    def __init__(self, task, scene=None, GUI=False, log_dir=None, n_envs=None, requires_grad=False):
        self.task = task
        self.GUI = GUI
        self.n_envs = n_envs
        self.requires_grad = requires_grad
        print(f"GUI: {self.GUI}, n_envs: {self.n_envs}, requires_grad: {self.requires_grad}")
        gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)
        if scene is None:
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
                    requires_grad=requires_grad,
                    # gravity=(0.,0.,0.)
                ),
                rod_options=gs.options.RodOptions(
                    damping=15.0,
                    angular_damping=10.0,
                    n_pbd_iters=20,
                ),
                show_viewer=self.GUI,
            )
        else:
            self.scene = scene

        self.scene_built_for_training = False
        self.scene_built_for_evaluation = False
        self.img_save_dir = None
        self.img_steps = 0

        self.cmaes_optimizer_created = False
        self.iter = 0

        self.create_log_dir(log_dir)

        self.construct_scene()

    def construct_traj_optim(self, max_ddist=0.1, max_grad_norm=1000, debug=False):
        if not self.requires_grad:
            return

        self.c = TrajOptimCMAES(
            scene=self.scene,
            rod=self.rope,
            grasp_point_ids=self.control_idx,
            n_optim_dofs=3,
            max_ddist=max_ddist,
            max_grad_norm=max_grad_norm,
            debug=debug,
        )

    def construct_scale_array(self, scale_method, n_steps, exp_base=1.1):
        if scale_method is None:
            scale_array = torch.ones(n_steps, dtype=gs.tc_float)
            self.scale_array = scale_array / n_steps
            print(f'Using uniform scale array:\n{self.scale_array}')
        elif scale_method == 'linear':
            self.scale_array = create_linear_array(n_steps)
            print(f'Using linear scale array:\n{self.scale_array}')
        elif scale_method == 'exp':
            self.scale_array = create_exp_array(n_steps, base=exp_base)
            print(f'Using exponential scale array (base={exp_base}):\n{self.scale_array}')
        elif scale_method == 'custom':
            self.scale_array = create_custom_array(n_steps)
            print(f'Using custom scale array:\n{self.scale_array}')
        else:
            raise ValueError(f'Unknown scale method: {scale_method}')

    def create_log_dir(self, log_dir):
        log_dir = os.path.join(log_dir, 'try')
        os.makedirs(log_dir, exist_ok=True)
        # n_tries = len([fil for fil in os.listdir(log_dir) if not '.' in fil])
        # self.img_save_dir = os.path.join(log_dir, f"{n_tries:03d}")
        # os.makedirs(self.img_save_dir, exist_ok=True)
        # os.makedirs(os.path.join(self.img_save_dir, "opt_log"), exist_ok=True)

    def init_mass(self, mass=0.015):
        for entity in self.scene.sim.rigid_solver.entities[2:]:
            for link in entity.links:
                link._inertial_mass = mass

    def construct_scene(self):
        raise NotImplementedError()

    def reward(self):
        raise NotImplementedError()
        self.img_steps += 1

    def save_gif(self, save_dir):
        images = []
        file_list = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        file_list.sort()
        for f in file_list:
            images.append(imageio.imread(os.path.join(save_dir, f)))
        imageio.mimsave(os.path.join(save_dir, 'movie.gif'), images)
    
    def reset(self):
        self.scene.reset()

    def loss_above_plane(self, state):
        # Required loss to make sure the vertices above the plane
        verts_batch = state.pos
        loss_abv_plane = torch.relu(
            self.rope.material.segment_radius - verts_batch[:, :, 2]
        ).sum(dim=1)                    # (n_envs,)

        return loss_abv_plane

    def gd_one_step(self, trajs, lr):
        assert trajs.ndim == 3, f"trajs must be (n_envs, n_steps, dof), got {trajs.shape}"
        n_envs, n_steps, dof = trajs.shape
        trajs_origin = trajs.copy()
        trajs = torch.tensor(trajs, dtype=gs.tc_float)
        assert n_envs == self.n_envs, f"n_envs mismatch: trajs has {n_envs}, self.n_envs is {self.n_envs}"
        n_ctrl = len(self.control_idx)
        assert dof % 3 == 0 and dof // 3 == n_ctrl, (
            f"dof must be 3 * len(control_idx). Got dof={dof}, len(control_idx)={n_ctrl}"
        )

        total_horizon = 0
        horizon_ids = list()
        self.scene.reset()

        fixed_np = np.zeros((self.n_envs, self.rope.n_vertices), dtype=bool)
        fixed_np[:, self.control_idx] = True
        self.rope.set_fixed(0, fixed_np)

        loss = 0.
        for i in range(n_steps):
            local_loss = 0.
            n_horizons = self.steps_interval
            # (n_envs, n_ctrl, 3)
            traj_i = trajs[:, i].reshape(self.n_envs, -1, 3)
            hpos, _ = self.c.pre_apply_grad(dpos=traj_i, num_horizons=n_horizons)
            for j in range(n_horizons):
                self.c.on_apply_grad(hpos[j])
                self.scene.step()

            state = self.rope.get_state()
            total_horizon += n_horizons
            horizon_ids.append(total_horizon)

            scale = self.scale_array[i]

            local_loss += self.loss_criterion(state).mean()
            local_loss += self.loss_above_plane(state).mean()

            loss += scale * local_loss

        loss.backward()

        deltas = list()
        for horizon_idx in horizon_ids:
            delta = self.c.gather_grad(horizon_idx=horizon_idx, lr=lr)
            deltas.append(delta)

        # (n_envs, n_steps, n_ctrl, 3)
        deltas = torch.stack(deltas, dim=1)
        deltas = deltas.reshape(self.n_envs, n_steps, -1)
        deltas = deltas.detach().cpu().numpy()

        print(f'traj: {np.abs(trajs_origin).mean(0).mean(0)}')
        print(f'delta: {np.abs(deltas).mean(0).mean(0)}')

        # Update trajs
        return trajs_origin + deltas
