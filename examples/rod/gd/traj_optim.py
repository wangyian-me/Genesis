import torch
import numpy as np
import genesis as gs

from genesis.engine.entities import RodEntity


def create_linear_array(N):
    base_seq = torch.arange(1, N + 1, dtype=gs.tc_float)
    out = base_seq / base_seq.sum()
    return out

def create_exp_array(N, base=1.1):
    exponents = torch.arange(N, dtype=gs.tc_float)
    base_seq = base ** exponents
    out = base_seq / base_seq.sum()
    return out

def create_custom_array(N):
    base_seq = torch.ones(N, dtype=gs.tc_float)
    base_seq[:N-1] = 0.5 / (N - 1)
    base_seq[N-1] = 0.5
    # assert base_seq.sum() == 1.0
    base_seq = base_seq / base_seq.sum() # ensure sum to 1
    return base_seq

def cosine_learning_rate_scheduler(base_lr, cur_iter, max_iter, min_lr=1e-6):
    if cur_iter >= max_iter:
        return min_lr
    cosine_decay = 0.5 * (1 + np.cos(np.pi * cur_iter / max_iter))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    return lr

class TrajOptim:
    def __init__(
        self,
        scene,
        rod: RodEntity,
        grasp_point_ids,
        n_stages=10,
        n_optim_dofs=3,
        max_ddist=0.05,
        max_grad_norm=1000.,
        use_adam=False,
        adam_config=None,
        debug=False,
        # lr scheduler
        lr_scheduler=None,
    ):
        self.scene = scene
        self.rod = rod
        self.grasp_point_ids = grasp_point_ids
        self.n_grasp_points = len(grasp_point_ids)

        self.traj = torch.zeros(
            size=(self.scene.n_envs, n_stages, self.n_grasp_points, n_optim_dofs), dtype=gs.tc_float
        )

        # for Adam optimizer
        self.use_adam = use_adam
        if self.use_adam:
            self.m_buffer = torch.zeros_like(self.traj)
            self.v_buffer = torch.zeros_like(self.traj)
            if adam_config is None:
                adam_config = {
                    "beta1": 0.9,
                    "beta2": 0.99,
                    "eps": 1e-8,
                }
            else:
                if "beta1" not in adam_config:
                    adam_config["beta1"] = 0.9
                if "beta2" not in adam_config:
                    adam_config["beta2"] = 0.99
                if "eps" not in adam_config:
                    adam_config["eps"] = 1e-8
            self.adam_config = adam_config

        if lr_scheduler is None:
            self.lr_scheduler = None
            print('No learning rate scheduler used.')
        elif lr_scheduler == 'cosine':
            self.lr_scheduler = cosine_learning_rate_scheduler
            print('Using cosine learning rate scheduler.')
        else:
            raise ValueError(f'Unknown learning rate scheduler: {lr_scheduler}')

        self.n_stages = n_stages
        self.n_optim_dofs = n_optim_dofs
        self.max_ddist = max_ddist
        self.max_grad_norm = max_grad_norm

        self.debug = debug
        self.debug_point_nodes = list()

    def pre_apply_grad(self, stage_idx, num_horizons=1):
        # compute delta pos for this stage, and then distribute to each horizon step within this stage
        dpos = self.traj[:, stage_idx, :, :]
        # ddist = torch.linalg.norm(dpos, dim=-1)
        # weight = self.max_ddist / (ddist + gs.EPS)
        # dpos_ = dpos * torch.minimum(weight, torch.ones_like(weight))[:, :, None]
        dpos_ = dpos

        expected_pos = self.rod.get_all_verts_tc()

        if self.debug:
            debug_pos = expected_pos[:, self.grasp_point_ids, :] + dpos_
            debug_pos = debug_pos.clone().detach().cpu().numpy()
            for batch_idx in range(self.scene.n_envs):
                offset = self.scene.envs_offset[batch_idx]
                for i in self.debug_point_nodes:
                    self.scene.clear_debug_object(i)
                self.debug_point_nodes = list()
                for i in range(self.n_grasp_points):
                    if debug_pos[batch_idx, i, 2] < self.rod.material.segment_radius:
                        color = (1.0, 1.0, 0.0, 0.6)
                        debug_pos[batch_idx, i, 2] = self.rod.material.segment_radius
                    else:
                        color = (0.0, 1.0, 0.0, 0.6)
                    self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                        pos=debug_pos[batch_idx, i] + offset,
                        radius=0.016,
                        color=color
                    ))

        target_pos_list = list()
        current_pos = expected_pos.clone()
        for i in range(num_horizons):
            alpha = (i + 1) / num_horizons
            expected_pos[:, self.grasp_point_ids, :] = current_pos[:, self.grasp_point_ids, :] + alpha * dpos_
            target_pos_list.append(expected_pos)

        return target_pos_list, dpos_

    def on_apply_grad(self, target_pos):
        # clamp the z-axis to be at least rod radius
        rod_radius = torch.tensor(self.rod.material.segment_radius, dtype=gs.tc_float, device=target_pos.device)
        target_pos[:, :, 2] = torch.maximum(target_pos[:, :, 2], rod_radius)
        self.rod.set_position(target_pos)

    def gather_grad(self, stage_idx, horizon_idx, cur_step=None, max_step=None, lr=0.01, lr_min=1e-6):
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(base_lr=lr, cur_iter=cur_step, max_iter=max_step, min_lr=lr_min)

        grad = self.rod._queried_states[horizon_idx][0].pos.grad

        # [n_envs, n_grasp_points, 3]
        contact_grad = grad[:, self.grasp_point_ids, :]
        # replace NaN or Inf with 0
        contact_grad = torch.where(torch.isnan(contact_grad), torch.zeros_like(contact_grad), contact_grad)
        contact_grad = torch.where(torch.isinf(contact_grad), torch.zeros_like(contact_grad), contact_grad)

        # clip gradient
        grad_norm = torch.linalg.norm(contact_grad, dim=-1)
        weight = self.max_grad_norm / (grad_norm + gs.EPS)
        contact_grad = contact_grad * torch.minimum(weight, torch.ones_like(weight))[:, :, None]

        if self.use_adam:
            # Adam
            beta1 = self.adam_config["beta1"]
            beta2 = self.adam_config["beta2"]
            eps = self.adam_config["eps"]

            m_t = beta1 * self.m_buffer[:, stage_idx, :, :] + (1 - beta1) * contact_grad
            v_t = beta2 * self.v_buffer[:, stage_idx, :, :] + (1 - beta2) * (contact_grad ** 2)
            self.m_buffer[:, stage_idx, :, :] = m_t
            self.v_buffer[:, stage_idx, :, :] = v_t

            m_cap = m_t / (1 - beta1 ** (cur_step + 1))
            v_cap = v_t / (1 - beta2 ** (cur_step + 1))

            d_pos = -lr * m_cap / (torch.sqrt(v_cap) + eps)
        else:
            # SGD
            d_pos = -lr * contact_grad

        # Post-step: detect collisions and correct the trajectory if necessary.
        collided = self.rod._queried_states[horizon_idx][0].collided
        # [n_envs, n_grasp_points]
        contact_collided = collided[:, self.grasp_point_ids]
        if contact_collided.any():
            collision_normal = self.rod._queried_states[horizon_idx][0].collision_normal
            collision_penetration = self.rod._queried_states[horizon_idx][0].collision_penetration

            contact_col_normal = collision_normal[:, self.grasp_point_ids, :]
            contact_col_pen = collision_penetration[:, self.grasp_point_ids]

            # For each collided point, we will push it out along the collision normal by the penetration depth.
            # This is a simple way to correct for collisions in the trajectory optimization.
            correction = contact_col_normal * contact_col_pen[:, :, None]
            # We will apply this correction to the trajectory if the point is collided.
            d_pos = torch.where(
                contact_collided[:, :, None], correction, d_pos
            )

        self.traj[:, stage_idx, :, :] += d_pos

        # ensure the max step distance constraint
        delta_dis = self.traj[:, stage_idx, :, :]
        ddist = torch.linalg.norm(delta_dis, dim=-1)
        weight = self.max_ddist / (ddist + gs.EPS)
        self.traj[:, stage_idx, :, :] = delta_dis * torch.minimum(weight, torch.ones_like(weight))[:, :, None]
