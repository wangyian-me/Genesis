import torch
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

class TrajOptimCMAES:
    def __init__(
        self,
        scene,
        rod: RodEntity,
        grasp_point_ids,
        n_optim_dofs=3,
        max_ddist=0.05,
        max_grad_norm=1000.,
        debug=False
    ):
        self.scene = scene
        self.rod = rod
        self.grasp_point_ids = grasp_point_ids
        self.n_grasp_points = len(grasp_point_ids)

        self.n_optim_dofs = n_optim_dofs
        self.max_ddist = max_ddist
        self.max_grad_norm = max_grad_norm

        self.debug = debug
        self.debug_point_nodes = list()

    def pre_apply_grad(self, dpos, num_horizons=1):
        # compute delta pos from cmaes sampling results
        # dpos: (n_envs, n_grasp_points, 3)
        assert dpos.shape == (self.scene.n_envs, self.n_grasp_points, self.n_optim_dofs)

        ddist = torch.linalg.norm(dpos, dim=-1)
        weight = self.max_ddist / (ddist + gs.EPS)
        dpos_ = dpos * torch.minimum(weight, torch.ones_like(weight))[:, :, None]

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
                    self.debug_point_nodes.append(self.scene.draw_debug_sphere(
                        pos=debug_pos[batch_idx, i] + offset,
                        radius=0.016,
                        color=(1.0, 0.0, 0.0, 0.6)
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

    def gather_grad(self, horizon_idx):
        grad = self.rod._queried_states[horizon_idx][0].pos.grad

        # (n_envs, n_grasp_points, 3)
        contact_grad = grad[:, self.grasp_point_ids, :]
        # replace NaN or Inf with 0
        contact_grad = torch.where(torch.isnan(contact_grad), torch.zeros_like(contact_grad), contact_grad)
        contact_grad = torch.where(torch.isinf(contact_grad), torch.zeros_like(contact_grad), contact_grad)

        # clip gradient
        grad_norm = torch.linalg.norm(contact_grad, dim=-1)
        weight = self.max_grad_norm / (grad_norm + gs.EPS)
        contact_grad = contact_grad * torch.minimum(weight, torch.ones_like(weight))[:, :, None]

        delta = -contact_grad # assume lr=1.0 here

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
            delta = correction

        return delta
