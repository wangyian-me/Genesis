# pylint: disable=no-value-for-parameter

import numpy as np
from math import pi
import gstaichi as ti
import gstaichi.math as tm
import torch

import genesis as gs
from genesis.engine.entities.rod_entity import RodEntity
from genesis.engine.states.solvers import RODSolverState
from genesis.utils.misc import ti_field_to_torch

from .base_solver import Solver

EPS = 1e-14


@ti.func
def get_perpendicular_vector(vector):
    """
    Returns a *unit* vector perpendicular to the input vector.
    """
    # Pick axis least aligned with vector
    abs_vector = ti.abs(vector)

    a = ti.Vector([0.0, 0.0, 0.0])
    if abs_vector.x <= abs_vector.y and abs_vector.x <= abs_vector.z:
        a = ti.Vector([1.0, 0.0, 0.0])
    elif abs_vector.y <= abs_vector.z:
        a = ti.Vector([0.0, 1.0, 0.0])
    else:
        a = ti.Vector([0.0, 0.0, 1.0])
    return tm.cross(vector, a).normalized()

@ti.func
def parallel_transport_normalized(t0, t1, v):
    """
    Transport vector :math:`v` from edge with tangent vector :math:`e0` to edge with tangent
    vector :math:`e1` (edge tangent vectors are normalized)
    """
    sin_theta_axis = tm.cross(t0, t1)
    cos_theta = tm.dot(t0, t1)
    den = 1 + cos_theta # denominator

    vprime = ti.Vector([0.0, 0.0, 0.0])
    if ti.abs(den) < EPS:
        vprime = v

    elif ti.abs(t0.x - t1.x) < EPS and ti.abs(t0.y - t1.y) < EPS and ti.abs(t0.z - t1.z) < EPS:
        vprime = v

    else:
        vprime = cos_theta * v + tm.cross(sin_theta_axis, v) + (tm.dot(sin_theta_axis, v) / den) * sin_theta_axis
    return vprime

@ti.func
def curvature_binormal(e0, e1):
    """
    Compute the curvature binormal for a vertex between two edges with tangents
    :math:`e0` and :math:`e1`, respectively (edge tangent vectors *not* necessarily normalized)
    """
    return 2.0 * tm.cross(e0, e1) / (tm.length(e0) * tm.length(e1) + tm.dot(e0, e1))

@ti.func
def get_updated_material_frame(prev_d3, d3, ref_d1, ref_d2, theta):
    """
    Parallel transport the reference frame vectors :math:`ref_d1` and :math:`ref_d2` from
    the previous edge to the new tangent vector :math:`d3` to get the updated reference frame.
    Then, rotate them by the twist angle :math:`theta` to get the updated material frame.
    """
    ref_d1 = parallel_transport_normalized(prev_d3, d3, ref_d1)
    ref_d2 = parallel_transport_normalized(prev_d3, d3, ref_d2)
    d1 = ti.cos(theta) * ref_d1 + ti.sin(theta) * ref_d2
    d2 = -ti.sin(theta) * ref_d1 + ti.cos(theta) * ref_d2
    return d1, d2, ref_d1, ref_d2

@ti.func
def get_angle(a, vec1, vec2):
    """
    Get the signed angle from :math:`vec1` to :math:`vec2` around axis :math:`a`; 
    ccw angles are positive. Assumes all vectors are *normalized* and *perpendicular* to :math:`a`
    Output in the range :math:`[-pi, pi]`
    """
    s = ti.max(-1.0, ti.min(1.0, tm.cross(vec1, vec2).dot(a)))
    c = ti.max(-1.0, ti.min(1.0, tm.dot(vec1, vec2)))
    return tm.atan2(s, c)

@ti.func
def get_updated_reference_twist(ref_d1_im1, ref_d1, d3_im1, d3):
    """
    Get the reference twist angle for the current edge based on the previous edge's
    reference director :math:`ref_d1_im1`, the current edge's reference director 
    :math:`ref_d1`, and the previous and current edge's tangent vectors :math:`d3_im1` 
    and :math:`d3`. Assumes all vectors are *normalized*.
    """
    # Finite rotation angle needed to take the parallel transported copy
    # of the previous edge's reference director to the current edge's
    # reference director.
    vec1 = parallel_transport_normalized(d3_im1, d3, ref_d1_im1)
    vec2 = ref_d1
    reference_twist = get_angle(d3, vec1, vec2)
    return reference_twist

@ti.func
def quat_rotate(q: tm.vec4, v: tm.vec3) -> tm.vec3:
    """
    Rotate vector `v` by quaternion `q`.
    """
    qvec = ti.Vector([q[1], q[2], q[3]])
    uv = tm.cross(qvec, v)
    uuv = tm.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


@ti.data_oriented
class RodSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._floor_height = options.floor_height
        self._floor_normal = options.floor_normal
        self._adjacent_gap = options.adjacent_gap
        self._damping = options.damping
        self._n_pbd_iters = options.n_pbd_iters

        # no boundary

        # lazy initialization
        self._constraints_initialized = False

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif isinstance(shape, (list, tuple)):
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)

    def init_rod_fields(self):
        # rod information (static)
        struct_rod_info = ti.types.struct(
            # material properties
            use_inextensible=gs.ti_bool,
            stretching_stiffness=gs.ti_float,
            bending_stiffness=gs.ti_float,
            twisting_stiffness=gs.ti_float,
            plastic_yield=gs.ti_float,
            plastic_creep=gs.ti_float,

            # indices
            first_vert_idx=gs.ti_int,           # index of the first vertex of this rod
            first_edge_idx=gs.ti_int,           # index of the first edge of this rod
            first_internal_vert_idx=gs.ti_int,  # index of the first internal vertex of this rod
            n_verts=gs.ti_int,                  # number of vertices in this rod

            # is loop
            is_loop=gs.ti_bool,
        )

        # rod energy (dynamic)
        struct_rod_energy = ti.types.struct(
            stretching_energy=gs.ti_float,
            bending_energy=gs.ti_float,
            twisting_energy=gs.ti_float,
        )

        self.rods_info = struct_rod_info.field(
            shape=self._n_rods, layout=ti.Layout.SOA
        )

        self.rods_energy = struct_rod_energy.field(
            shape=self._batch_shape(self._n_rods),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

        # keep track of gradients for time-stepping
        self.gradients = ti.field(
            dtype=gs.ti_float, needs_grad=False,
            shape=self._batch_shape(self._n_dofs)
        )

    def init_vertex_fields(self):
        # vertex information (static)
        struct_vertex_info = ti.types.struct(
            mass=gs.ti_float,
            radius=gs.ti_float,
            vert_rest=gs.ti_vec3,
            mu_s=gs.ti_float,
            mu_k=gs.ti_float,
            rod_idx=gs.ti_int,      # index of the rod this vertex belongs to
        )

        # vertex state (dynamic)
        struct_vertex_state = ti.types.struct(
            vert=gs.ti_vec3,        # current position
            vert_prev=gs.ti_vec3,   # previous position # TODO: do we need grad for this?
            vel=gs.ti_vec3,
        )

        struct_vertex_state_ng = ti.types.struct(
            fixed=gs.ti_bool,       # is the vertex fixed
            f_s=gs.ti_vec3,         # stretching force
            f_b=gs.ti_vec3,         # bending force
            f_t=gs.ti_vec3,         # twisting force
        )

        self.vertices_info = struct_vertex_info.field(
            shape=self._n_vertices, layout=ti.Layout.SOA
        )

        self.vertices = struct_vertex_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_vertices)),
            needs_grad=True,
            layout=ti.Layout.SOA
        )

        self.vertices_ng = struct_vertex_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_vertices)),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

    def init_edge_fields(self):
        # edge information (static)
        struct_edge_info = ti.types.struct(
            edge_rest=gs.ti_vec3,
            length_rest=gs.ti_float,
            d1_rest=gs.ti_vec3,         # material frame direction 1 in rest state
            d2_rest=gs.ti_vec3,         # material frame direction 2 in rest state
            d3_rest=gs.ti_vec3,         # material frame direction 3 in rest state (tangent)
            vert_idx=gs.ti_int,         # index of the starting vertex of this edge
        )

        # edge state (dynamic)
        struct_edge_state = ti.types.struct(
            edge=gs.ti_vec3,        # current edge vector
            length=gs.ti_float,     # current edge length
            theta=gs.ti_float,      # twist angle
            d1=gs.ti_vec3,          # material frame direction 1
            d2=gs.ti_vec3,          # material frame direction 2
            d3=gs.ti_vec3,          # material frame direction 3 (tangent)
            d1_ref=gs.ti_vec3,      # reference material frame direction 1
            d2_ref=gs.ti_vec3,      # reference material frame direction 2
            d3_prev=gs.ti_vec3,     # previous tangent
        )

        self.edges_info = struct_edge_info.field(
            shape=self._n_edges, layout=ti.Layout.SOA
        )

        self.edges = struct_edge_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_edges)),
            needs_grad=True,
            layout=ti.Layout.SOA
        )

    def init_internal_vertex_fields(self):
        # internal vertex information (static)
        struct_internal_vertex_info = ti.types.struct(
            twist_rest=gs.ti_float,     # rest twist
            edge_idx=gs.ti_int,         # index of the starting edge of this internal vertex
        )

        # internal vertex state (dynamic)
        struct_internal_vertex_state = ti.types.struct(
            kb=gs.ti_vec3,          # current curvature binormal
            twist=gs.ti_float,      # current twist
        )

        struct_internal_vertex_state_ng = ti.types.struct(
            kappa_rest=gs.ti_vec2,      # rest curvature,
        )

        self.internal_vertices_info = struct_internal_vertex_info.field(
            shape=self._n_internal_vertices, layout=ti.Layout.SOA
        )

        self.internal_vertices = struct_internal_vertex_state.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_internal_vertices)),
            needs_grad=True,
            layout=ti.Layout.SOA
        )

        self.internal_vertices_ng = struct_internal_vertex_state_ng.field(
            shape=self._batch_shape((self.sim.substeps_local + 1, self._n_internal_vertices)),
            needs_grad=False,
            layout=ti.Layout.SOA
        )

    def init_valid_edge_pairs_for_constraints(self):
        # NOTE: call this after call `_kernel_add_rods`
        valid_edge_pairs = list()
        for i in range(self._n_vertices):
            for j in range(i + 1, self._n_vertices):
                rod_id_i = self.vertices_info[i].rod_idx
                local_id_i = i - self.rods_info[rod_id_i].first_vert_idx
                rod_id_j = self.vertices_info[j].rod_idx
                local_id_j = j - self.rods_info[rod_id_j].first_vert_idx

                # filtering
                # 1. ensure i and j can actually start an edge
                is_loop_i = self.rods_info[rod_id_i].is_loop
                is_loop_j = self.rods_info[rod_id_j].is_loop

                if not is_loop_i and local_id_i >= self.rods_info[rod_id_i].n_verts - 1:
                    continue
                if not is_loop_j and local_id_j >= self.rods_info[rod_id_j].n_verts - 1:
                    continue

                # 2. ignore adjacent edges on the same rod
                if rod_id_i == rod_id_j:
                    if is_loop_i:
                        n_verts_in_rod = self.rods_info[rod_id_i].n_verts
                        dist_forward = local_id_j - local_id_i
                        dist_backward = (local_id_i + n_verts_in_rod) - local_id_j

                        if dist_forward < self._adjacent_gap + 1 or dist_backward < self._adjacent_gap + 1:
                            continue # Skip if adjacent on the loop.
                    else:
                        if abs(local_id_j - local_id_i) < self._adjacent_gap + 1:
                            continue # Skip if adjacent on the chain.

                valid_edge_pairs.append((i, j))
        
        valid_edge_pairs = np.array(valid_edge_pairs, dtype=gs.np_int)
        self._n_valid_edge_pairs = valid_edge_pairs.shape[0]

        # constraint for rod-rod collision
        struct_rr_info = ti.types.struct(
            valid_pair=ti.types.vector(2, gs.ti_int),
        )

        struct_rr_state = ti.types.struct(
            normal=gs.ti_vec3,
            penetration=gs.ti_float,
        )

        self.rr_constraint_info = struct_rr_info.field(
            shape=self._n_valid_edge_pairs, layout=ti.Layout.SOA
        )

        self.rr_constraints = struct_rr_state.field(
            shape=self._batch_shape(self._n_valid_edge_pairs),
            needs_grad=False,
            layout=ti.Layout.AOS
        )

    def init_constraints(self):
        struct_plane_info = ti.types.struct(
            point=gs.ti_vec3,
            normal=gs.ti_vec3,
            mu_s=gs.ti_float,
            mu_k=gs.ti_float,
        )

        # constraint for rod-plane collision
        struct_rp_state = ti.types.struct(
            normal=gs.ti_vec3,
            penetration=gs.ti_float,
            plane_idx=gs.ti_int
        )

        self.planes_info = struct_plane_info.field(
            shape=self._n_planes, layout=ti.Layout.SOA
        )

        self.rp_constraints = struct_rp_state.field(
            shape=self._batch_shape(self._n_vertices),
            needs_grad=False,
            layout=ti.Layout.AOS
        )

        self.init_valid_edge_pairs_for_constraints()

        self._constraints_initialized = True

    def init_ckpt(self):
        self._ckpt = dict()

    def reset_grad(self):
        self.elements_v.grad.fill(0)
        self.elements_el.grad.fill(0)

        for entity in self._entities:
            entity.reset_grad()

    def build(self):
        super().build()
        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._n_rods = len(self._entities)
        self._n_vertices = self.n_vertices
        self._n_edges = self.n_edges
        self._n_internal_vertices = self.n_internal_vertices
        self._n_dofs = self.n_dofs
        self._n_planes = 1  # NOTE: only floor for now

        if self.is_active():
            self.init_rod_fields()
            self.init_vertex_fields()
            self.init_edge_fields()
            self.init_internal_vertex_fields()

            for entity in self._entities:
                entity._add_to_solver()

            self.init_constraints()

        # default plane collider
        point = np.asarray(self._floor_normal) * self._floor_height
        for j in range(3):
            self.planes_info[0].point[j] = point[j]
            self.planes_info[0].normal[j] = self._floor_normal[j]    
        self.planes_info[0].mu_s = 0.3
        self.planes_info[0].mu_k = 0.25

    def add_entity(self, idx, material, morph, surface):

        # create entity
        entity = RodEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            v_start=self.n_vertices,
            e_start=self.n_edges,
            iv_start=self.n_internal_vertices,
        )

        self._entities.append(entity)
        return entity

    def is_active(self):
        return self._n_vertices > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ logging --------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.kernel
    def get_rod_length(self, f: ti.i32, i_r: ti.i32, length: ti.types.ndarray()):
        n_verts = self.rods_info[i_r].n_verts
        first_edge_idx = self.rods_info[i_r].first_edge_idx
        for i_e, i_b in ti.ndrange(n_verts - 1, self._B):
            edge_idx = first_edge_idx + i_e
            length[i_b] += self.edges[f, edge_idx, i_b].length

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def _func_clear_energy(self):
        for i_r, i_b in ti.ndrange(self._n_rods, self._B):
            self.rods_energy[i_r, i_b].stretching_energy = 0.0
            self.rods_energy[i_r, i_b].bending_energy = 0.0
            self.rods_energy[i_r, i_b].twisting_energy = 0.0

    @ti.func
    def _func_clear_gradients(self):
        self.gradients.fill(0.0)

    @ti.kernel
    def update_centerline_positions(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            if not self.vertices_ng[f, i_v, i_b].fixed:
                self.vertices[f, i_v, i_b].vert += self.vertices[f, i_v, i_b].vel * self.substep_dt

    @ti.kernel
    def update_centerline_velocities(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            mass = self.vertices_info[i_v].mass
            if not self.vertices_ng[f, i_v, i_b].fixed:
                gradient = ti.Vector([
                    self.gradients[3 * i_v + 0, i_b],
                    self.gradients[3 * i_v + 1, i_b],
                    self.gradients[3 * i_v + 2, i_b],
                ])
                self.vertices[f, i_v, i_b].vel -= gradient / mass * self.substep_dt

                # apply damping if enabled
                self.vertices[f, i_v, i_b].vel *= ti.exp(-self.substep_dt * self.damping)
                # add gravity (avoiding damping on gravity)
                self.vertices[f, i_v, i_b].vel += self.substep_dt * self._gravity[i_b]

    @ti.kernel
    def update_centerline_edges(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            self.edges[f, i_e, i_b].edge = self.vertices[f, v_e, i_b].vert - self.vertices[f, v_s, i_b].vert
            self.edges[f, i_e, i_b].length = tm.length(self.edges[f, i_e, i_b].edge)

    @ti.kernel
    def update_frame_thetas(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            if not self.vertices_ng[f, v_s, i_b].fixed or not self.vertices_ng[f, v_e, i_b].fixed:
                self.edges[f, i_e, i_b].theta -= self.gradients[3 * self._n_vertices + i_e, i_b] * self.substep_dt

    @ti.kernel
    def update_material_states(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges[f, i_e, i_b].d3 = self.edges[f, i_e, i_b].edge.normalized()

            d1, d2, d1_ref, d2_ref = get_updated_material_frame(
                self.edges[f, i_e, i_b].d3_prev,
                self.edges[f, i_e, i_b].d3,
                self.edges[f, i_e, i_b].d1_ref,
                self.edges[f, i_e, i_b].d2_ref,
                self.edges[f, i_e, i_b].theta,
            )
            self.edges[f, i_e, i_b].d1 = d1
            self.edges[f, i_e, i_b].d2 = d2
            self.edges[f, i_e, i_b].d1_ref = d1_ref
            self.edges[f, i_e, i_b].d2_ref = d2_ref
        
        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)

            self.internal_vertices[f, i_iv, i_b].kb = curvature_binormal(
                self.edges[f, e_s, i_b].d3, self.edges[f, e_e, i_b].d3
            )
            twist_ref = get_updated_reference_twist(
                self.edges[f, e_s, i_b].d1_ref, self.edges[f, e_e, i_b].d1_ref,
                self.edges[f, e_s, i_b].d3, self.edges[f, e_e, i_b].d3
            )
            self.internal_vertices[f, i_iv, i_b].twist = self.edges[f, e_e, i_b].theta - self.edges[f, e_s, i_b].theta + twist_ref

    @ti.kernel
    def update_velocities_after_projection(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            if not self.vertices_ng[f, i_v, i_b].fixed:
                self.vertices[f, i_v, i_b].vel = (self.vertices[f, i_v, i_b].vert - self.vertices[f, i_v, i_b].vert_prev) / self.substep_dt

    @ti.kernel
    def record_previous_positions(self, f: ti.i32):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            self.vertices[f, i_v, i_b].vert_prev = self.vertices[f, i_v, i_b].vert

    @ti.kernel
    def record_previous_tangents(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            self.edges[f, i_e, i_b].d3_prev = self.edges[f, i_e, i_b].d3

    @ti.func
    def _func_compute_stretching_energy(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            # check stretching enabled
            K = self.rods_info[rod_id].stretching_stiffness
            if K <= 0.:
                continue
            
            r = (self.vertices_info[v_s].radius + self.vertices_info[v_e].radius) * 0.5
            a, b = r, r
            A = pi * a * b  # cross-sectional area

            strain_i = (self.edges[f, i_e, i_b].length / self.edges_info[i_e].length_rest) - 1.0

            self.rods_energy[rod_id, i_b].stretching_energy += 0.5 * K * A * ti.pow(strain_i, 2) * self.edges_info[i_e].length_rest

            # -------------------------------- gradients --------------------------------

            gradient_magnitude = K * A * strain_i

            gradient_dx_i   = - gradient_magnitude * self.edges[f, i_e, i_b].d3
            gradient_dx_ip1 =   gradient_magnitude * self.edges[f, i_e, i_b].d3

            for k in range(3):
                ti.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                ti.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_ip1[k])

                ti.atomic_add(self.vertices_ng[f, v_s, i_b].f_s[k], -gradient_dx_i[k])
                ti.atomic_add(self.vertices_ng[f, v_e, i_b].f_s[k], -gradient_dx_ip1[k])
    
    @ti.func
    def _func_compute_bending_energy(self, f: ti.i32):
        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)
            v_s, v_m, v_e = self.get_hinge_vertices(e_s)
            rod_id = self.vertices_info[v_s].rod_idx

            # check bending enabled
            E = self.rods_info[rod_id].bending_stiffness
            if E <= 0.:
                continue
            
            r = self.vertices_info[v_m].radius
            a, b = r, r
            A = pi * a * b  # cross-sectional area
            B11 = E * A * ti.pow(a, 2) / 4.0
            B22 = E * A * ti.pow(b, 2) / 4.0

            kb = self.internal_vertices[f, i_iv, i_b].kb
            l_i = (self.edges_info[e_s].length_rest + self.edges_info[e_e].length_rest) * 0.5

            kappa1_i =   0.5 * tm.dot(kb, self.edges[f, e_s, i_b].d2 + self.edges[f, e_e, i_b].d2)
            kappa2_i = - 0.5 * tm.dot(kb, self.edges[f, e_s, i_b].d1 + self.edges[f, e_e, i_b].d1)

            # bending plasticity
            kappa1_rest_i = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[0]
            kappa2_rest_i = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[1]
            curr_kappa = ti.Vector([kappa1_i, kappa2_i])
            rest_kappa = ti.Vector([kappa1_rest_i, kappa2_rest_i])

            elastic_kappa = curr_kappa - rest_kappa
            elastic_kappa_norm = tm.length(elastic_kappa)

            yield_thres = self.rods_info[rod_id].plastic_yield
            creep_rate = self.rods_info[rod_id].plastic_creep

            yield_amount = elastic_kappa_norm - yield_thres
            if yield_amount > 0.:
                delta_rest_kappa = creep_rate * (yield_amount / elastic_kappa_norm) * elastic_kappa
                self.internal_vertices_ng[f, i_iv, i_b].kappa_rest += delta_rest_kappa

            kappa1_rest_i = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[0]
            kappa2_rest_i = self.internal_vertices_ng[f, i_iv, i_b].kappa_rest[1]

            self.rods_energy[rod_id, i_b].bending_energy += 0.5 * (
                B11 * ti.pow(kappa1_i - kappa1_rest_i, 2) +
                B22 * ti.pow(kappa2_i - kappa2_rest_i, 2)
            ) / l_i

            # -------------------------------- gradients --------------------------------

            gradient_kappa1_i_x_i = ti.Vector.zero(dt=gs.ti_float, n=9)
            gradient_kappa2_i_x_i = ti.Vector.zero(dt=gs.ti_float, n=9)

            chi = 1. + tm.dot(self.edges[f, e_s, i_b].d3, self.edges[f, e_e, i_b].d3)
            d1_tilde = (self.edges[f, e_s, i_b].d1 + self.edges[f, e_e, i_b].d1) / chi
            d2_tilde = (self.edges[f, e_s, i_b].d2 + self.edges[f, e_e, i_b].d2) / chi
            d3_tilde = (self.edges[f, e_s, i_b].d3 + self.edges[f, e_e, i_b].d3) / chi

            dkappa1_i_de_im1 = tm.cross(d2_tilde, -self.edges[f, e_e, i_b].d3 / self.edges_info[e_s].length_rest) - \
                kappa1_i * d3_tilde / self.edges_info[e_s].length_rest
            dkappa1_i_de_i = tm.cross(d2_tilde, self.edges[f, e_s, i_b].d3 / self.edges_info[e_e].length_rest) - \
                kappa1_i * d3_tilde / self.edges_info[e_e].length_rest
            dkappa2_i_de_im1 = tm.cross(d1_tilde, self.edges[f, e_e, i_b].d3 / self.edges_info[e_s].length_rest) - \
                kappa2_i * d3_tilde / self.edges_info[e_s].length_rest
            dkappa2_i_de_i = tm.cross(d1_tilde, -self.edges[f, e_s, i_b].d3 / self.edges_info[e_e].length_rest) - \
                kappa2_i * d3_tilde / self.edges_info[e_e].length_rest
            
            gradient_kappa1_i_x_i[0:3] = dkappa1_i_de_im1 * (- 1.0)
            gradient_kappa1_i_x_i[3:6] = dkappa1_i_de_im1 * (  1.0) + dkappa1_i_de_i * (- 1.0)
            gradient_kappa1_i_x_i[6:9] = dkappa1_i_de_i   * (  1.0)
            gradient_kappa2_i_x_i[0:3] = dkappa2_i_de_im1 * (- 1.0)
            gradient_kappa2_i_x_i[3:6] = dkappa2_i_de_im1 * (  1.0) + dkappa2_i_de_i * (- 1.0)
            gradient_kappa2_i_x_i[6:9] = dkappa2_i_de_i   * (  1.0)

            gradient_dx_i = (
                B11 * (kappa1_i - kappa1_rest_i) * gradient_kappa1_i_x_i + \
                B22 * (kappa2_i - kappa2_rest_i) * gradient_kappa2_i_x_i
            ) / l_i
            for k in range(3):
                ti.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                ti.atomic_add(self.gradients[3 * v_m + k, i_b], gradient_dx_i[k + 3])
                ti.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_i[k + 6])

                ti.atomic_add(self.vertices_ng[f, v_s, i_b].f_b[k], -gradient_dx_i[k])
                ti.atomic_add(self.vertices_ng[f, v_m, i_b].f_b[k], -gradient_dx_i[k + 3])
                ti.atomic_add(self.vertices_ng[f, v_e, i_b].f_b[k], -gradient_dx_i[k + 6])

            gradient_kappa1_i_theta_i = - ti.Vector([
                tm.dot(kb, self.edges[f, e_s, i_b].d1) * 0.5,
                tm.dot(kb, self.edges[f, e_e, i_b].d1) * 0.5
            ])
            gradient_kappa2_i_theta_i = - ti.Vector([
                tm.dot(kb, self.edges[f, e_s, i_b].d2) * 0.5,
                tm.dot(kb, self.edges[f, e_e, i_b].d2) * 0.5
            ])

            gradient_dtheta_i = (
                B11 * (kappa1_i - kappa1_rest_i) * gradient_kappa1_i_theta_i + \
                B22 * (kappa2_i - kappa2_rest_i) * gradient_kappa2_i_theta_i
            ) / l_i
            theta_dof_s_idx = 3 * self._n_vertices + e_s
            theta_dof_e_idx = 3 * self._n_vertices + e_e
            ti.atomic_add(self.gradients[theta_dof_s_idx, i_b], gradient_dtheta_i[0])
            ti.atomic_add(self.gradients[theta_dof_e_idx, i_b], gradient_dtheta_i[1])

    @ti.func
    def _func_compute_twisting_energy(self, f: ti.i32):
        for i_iv, i_b in ti.ndrange(self.n_internal_vertices, self._B):
            e_s, e_e = self.get_hinge_edges(i_iv)
            v_s, v_m, v_e = self.get_hinge_vertices(e_s)
            rod_id = self.vertices_info[v_s].rod_idx

            # check twisting enabled
            G = self.rods_info[rod_id].twisting_stiffness
            if G <= 0.:
                continue

            r = self.vertices_info[v_m].radius
            a, b = r, r
            A = pi * a * b  # cross-sectional area
            beta = G * A * (ti.pow(a, 2) + ti.pow(b, 2)) / 4.0

            kb = self.internal_vertices[f, i_iv, i_b].kb
            l_i = (self.edges_info[e_s].length_rest + self.edges_info[e_e].length_rest) * 0.5
            m_i = self.internal_vertices[f, i_iv, i_b].twist
            m_i_rest = self.internal_vertices_info[i_iv].twist_rest

            self.rods_energy[rod_id, i_b].twisting_energy += 0.5 * beta * ti.pow(m_i - m_i_rest, 2) / l_i

            # -------------------------------- gradients --------------------------------

            gradient_m_i_dx_i = ti.Vector.zero(dt=gs.ti_float, n=9)
            gradient_m_i_dx_i[0:3] = - kb / (2.0 * self.edges[f, e_s, i_b].length)
            gradient_m_i_dx_i[3:6] =   kb / (2.0 * self.edges[f, e_s, i_b].length) - kb / (2.0 * self.edges[f, e_e, i_b].length)
            gradient_m_i_dx_i[6:9] =   kb / (2.0 * self.edges[f, e_e, i_b].length)
            gradient_dx_i = beta / l_i * (m_i - m_i_rest) * gradient_m_i_dx_i
            for k in range(3):
                ti.atomic_add(self.gradients[3 * v_s + k, i_b], gradient_dx_i[k])
                ti.atomic_add(self.gradients[3 * v_m + k, i_b], gradient_dx_i[k + 3])
                ti.atomic_add(self.gradients[3 * v_e + k, i_b], gradient_dx_i[k + 6])

                ti.atomic_add(self.vertices_ng[f, v_s, i_b].f_t[k], -gradient_dx_i[k])
                ti.atomic_add(self.vertices_ng[f, v_m, i_b].f_t[k], -gradient_dx_i[k + 3])
                ti.atomic_add(self.vertices_ng[f, v_e, i_b].f_t[k], -gradient_dx_i[k + 6])

            gradient_m_i_dtheta_i = ti.Vector([-1.0, 1.0])
            gradient_dtheta_i = beta / l_i * (m_i - m_i_rest) * gradient_m_i_dtheta_i
            theta_dof_s_idx = 3 * self._n_vertices + e_s
            theta_dof_e_idx = 3 * self._n_vertices + e_e
            ti.atomic_add(self.gradients[theta_dof_s_idx, i_b], gradient_dtheta_i[0])
            ti.atomic_add(self.gradients[theta_dof_e_idx, i_b], gradient_dtheta_i[1])

    @ti.kernel
    def compute_energy_and_gradients(self, f: ti.i32):
        # clear energy and gradients
        self._func_clear_energy()
        self._func_clear_gradients()

        self._func_compute_stretching_energy(f)
        self._func_compute_bending_energy(f)
        self._func_compute_twisting_energy(f)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        pass

    def substep_pre_coupling(self, f):
        if self.is_active():
            self.record_previous_positions(f)
            self.record_previous_tangents(f)
            self.compute_energy_and_gradients(f)
            self.update_centerline_velocities(f)
            self.update_frame_thetas(f)
            self.update_centerline_positions(f)

            self.clear_contact_states()
            for i in range(self._n_pbd_iters):
                self._kernel_apply_inextensibility_constraints(f)
                self._kernel_apply_plane_collision_constraints(f, i)
                self._kernel_apply_rod_collision_constraints(f, i)

    def substep_pre_coupling_grad(self, f):
        if self.is_active():
            pass

    def substep_post_coupling(self, f):
        if self.is_active():
            self.update_centerline_edges(f)
            self.update_material_states(f)

            self.update_velocities_after_projection(f)
            self._kernel_apply_plane_friction(f)
            self._kernel_apply_rod_friction(f)

    def substep_post_coupling_grad(self, f):
        if self.is_active():
            pass

    def reset_grad(self):
        pass

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    # TODO: implement gradient functions
    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def add_grad_from_state(self, state):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def set_state(self, f, state, envs_idx=None):
        if self.is_active():
            self._kernel_set_state(f, state.pos, state.vel, state.fixed)

    def get_state(self, f):
        if self.is_active():
            state = RODSolverState(self._scene)
            self._kernel_get_state(f, state.pos, state.vel, state.fixed)
        else:
            state = None
        return state

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        vertices = self.surface_render_v.vertices
        indices = self.surface_render_f.indices

        return vertices, indices

    def get_forces(self):
        """
        Get forces on all vertices.

        Returns:
            torch.Tensor : shape (B, n_vertices, 3) where B is batch size
        """
        if not self.is_active():
            return None

        return ti_field_to_torch(self.elements_v_energy.force)

    @ti.kernel
    def _kernel_add_rods(
        self,
        rod_idx: ti.i32,
        is_loop: ti.u1,
        use_inextensible: ti.u1,
        stretching_stiffness: ti.f64,
        bending_stiffness: ti.f64,
        twisting_stiffness: ti.f64,
        plastic_yield: ti.f64,
        plastic_creep: ti.f64,
        v_start: ti.i32,
        e_start: ti.i32,
        iv_start: ti.i32,
        n_verts: ti.i32,
    ):
        self.rods_info[rod_idx].use_inextensible = use_inextensible
        self.rods_info[rod_idx].stretching_stiffness = stretching_stiffness
        self.rods_info[rod_idx].bending_stiffness = bending_stiffness
        self.rods_info[rod_idx].twisting_stiffness = twisting_stiffness
        self.rods_info[rod_idx].plastic_yield = plastic_yield
        self.rods_info[rod_idx].plastic_creep = plastic_creep

        self.rods_info[rod_idx].is_loop = is_loop

        for i, i_b in ti.ndrange(self._n_rods, self._B):
            self.rods_energy[i, i_b].stretching_energy = 0.0
            self.rods_energy[i, i_b].bending_energy = 0.0
            self.rods_energy[i, i_b].twisting_energy = 0.0

        # -------------------------------- build indices --------------------------------

        self.rods_info[rod_idx].first_vert_idx = v_start
        self.rods_info[rod_idx].first_edge_idx = e_start
        self.rods_info[rod_idx].first_internal_vert_idx = iv_start
        self.rods_info[rod_idx].n_verts = n_verts

        # rod id of verts
        for i_v in range(n_verts):
            vert_idx = i_v + v_start
            self.vertices_info[vert_idx].rod_idx = rod_idx

        # vert id of edges
        n_edges = n_verts if is_loop else n_verts - 1
        for i_e in range(n_edges):
            vert_idx = i_e + v_start
            edge_idx = i_e + e_start
            self.edges_info[edge_idx].vert_idx = vert_idx

        # edge id of internal verts
        n_internal_verts = n_verts - (0 if is_loop else 2)
        for i_iv in range(n_internal_verts):
            edge_idx = -1
            if is_loop:
                edge_idx = tm.mod(i_iv - 1, n_internal_verts) + e_start
            else:
                edge_idx = i_iv + e_start
            iv_idx = i_iv + iv_start
            self.internal_vertices_info[iv_idx].edge_idx = edge_idx

    @ti.kernel
    def _kernel_finalize_rest_states(
        self,
        f: ti.i32,
        rod_idx: ti.i32,
        v_start: ti.i32,
        e_start: ti.i32,
        iv_start: ti.i32,
        verts_rest: ti.types.ndarray(dtype=tm.vec3, ndim=1),
        edges_rest: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    ):
        n_verts_local = verts_rest.shape[0]
        for i_v in range(n_verts_local):
            i_global = i_v + v_start

            # finalize rest vertices

            self.vertices_info[i_global].vert_rest[i_v] = verts_rest[i_v]

        is_loop = self.rods_info[rod_idx].is_loop
        n_edges_local = n_verts_local if is_loop else n_verts_local - 1
        for i_e in range(n_edges_local):
            i_global = i_e + e_start
            # v_s, v_e = self.get_edge_vertices(i_global)

            # finalize rest edges

            # self.edges_info[i_global].edge_rest = self.vertices_info[v_e].vert_rest - self.vertices_info[v_s].vert_rest
            self.edges_info[i_global].edge_rest = edges_rest[i_e]
            self.edges_info[i_global].length_rest = tm.length(self.edges_info[i_global].edge_rest)
            self.edges_info[i_global].d3_rest = self.edges_info[i_global].edge_rest.normalized()

            # finalize rest material frame (d1, d2, d3)

            if i_e == 0: # first edge
                self.edges_info[i_global].d1_rest = get_perpendicular_vector(self.edges_info[i_global].d3_rest)
            else:
                self.edges_info[i_global].d1_rest = parallel_transport_normalized(
                    self.edges_info[i_global - 1].d3_rest,
                    self.edges_info[i_global].d3_rest,
                    self.edges_info[i_global - 1].d1_rest,
                )
            self.edges_info[i_global].d2_rest = tm.cross(self.edges_info[i_global].d3_rest, self.edges_info[i_global].d1_rest)

        # deal with loop topology

        if self.rods_info[rod_idx].is_loop:
            e_end = e_start + n_edges_local - 1

            d1_final_transport = parallel_transport_normalized(
                self.edges_info[e_end].d3_rest,
                self.edges_info[e_start].d3_rest,
                self.edges_info[e_end].d1_rest,
            )

            total_holonomy_angle = get_angle(
                self.edges_info[e_start].d3_rest,
                d1_final_transport,
                self.edges_info[e_start].d1_rest,
            )

            for i_e in range(n_edges_local):
                i_global = i_e + e_start

                correction_angle = - total_holonomy_angle * (i_e / n_edges_local)
                d1_uncorrected = self.edges_info[i_global].d1_rest
                d2_uncorrected = self.edges_info[i_global].d2_rest
                c, s = ti.cos(correction_angle), ti.sin(correction_angle)
                self.edges_info[i_global].d1_rest = c * d1_uncorrected + s * d2_uncorrected
                self.edges_info[i_global].d2_rest = -s * d1_uncorrected + c * d2_uncorrected

        n_internal_verts_local = n_verts_local - (0 if is_loop else 2)
        for i_iv, i_b in ti.ndrange(n_internal_verts_local, self._B):
            i_global = i_iv + iv_start
            e_s, e_e = self.get_hinge_edges(i_global)

            # finalize rest curvature binormal

            rest_kbs = curvature_binormal(self.edges_info[e_s].d3_rest, self.edges_info[e_e].d3_rest)
            self.internal_vertices_ng[f, i_iv, i_b].kappa_rest = ti.Vector([
                  0.5 * tm.dot(rest_kbs, self.edges_info[e_s].d2_rest + self.edges_info[e_e].d2_rest),
                - 0.5 * tm.dot(rest_kbs, self.edges_info[e_s].d1_rest + self.edges_info[e_e].d1_rest),
            ])
            self.internal_vertices_info[i_global].twist_rest = 0.0  # assume no initial twist

    @ti.kernel
    def _kernel_finalize_states(
        self,
        f: ti.i32,
        rod_idx: ti.i32,
        v_start: ti.i32,
        e_start: ti.i32,
        iv_start: ti.i32,
        segment_mass: ti.f64,        # NOTE: we can use array
        segment_radius: ti.f64,      # NOTE: we can use array
        static_friction: ti.f64,     # NOTE: we can use array
        kinetic_friction: ti.f64,    # NOTE: we can use array
        verts: ti.types.ndarray(dtype=tm.vec3, ndim=1),
        edges: ti.types.ndarray(dtype=tm.vec3, ndim=1),
    ):
        n_verts_local = verts.shape[0]
        for i_v, i_b in ti.ndrange(n_verts_local, self._B):
            i_global = i_v + v_start

            # info (static)
            self.vertices_info[i_global].mass = segment_mass
            self.vertices_info[i_global].radius = segment_radius
            self.vertices_info[i_global].mu_s = static_friction
            self.vertices_info[i_global].mu_k = kinetic_friction
            self.vertices_info[i_global].rod_idx = rod_idx

            # state (dynamic)
            self.vertices[f, i_global, i_b].vert = verts[i_v]
            self.vertices[f, i_global, i_b].vert_prev = verts[i_v]
            self.vertices[f, i_global, i_b].vel = ti.Vector.zero(gs.ti_float, 3)

            # state (dynamic w/o grad)
            self.vertices_ng[f, i_global, i_b].fixed = False
            self.vertices_ng[f, i_global, i_b].f_s = ti.Vector.zero(gs.ti_float, 3)
            self.vertices_ng[f, i_global, i_b].f_b = ti.Vector.zero(gs.ti_float, 3)
            self.vertices_ng[f, i_global, i_b].f_t = ti.Vector.zero(gs.ti_float, 3)

        is_loop = self.rods_info[rod_idx].is_loop
        n_edges_local = n_verts_local if is_loop else n_verts_local - 1
        for i_e, i_b in ti.ndrange(n_edges_local, self._B):
            i_global = i_e + e_start
            # v_s, v_e = self.get_edge_vertices(i_global)

            # state (dynamic)

            # self.edges[f, i_global, i_b].edge = self.vertices[f, v_e, i_b].vert - self.vertices[f, v_s, i_b].vert
            self.edges[f, i_global, i_b].edge = edges[i_e]
            self.edges[f, i_global, i_b].length = tm.length(self.edges[f, i_global, i_b].edge)
            self.edges[f, i_global, i_b].d3 = self.edges[f, i_global, i_b].edge.normalized()

            if i_e == 0: # first edge
                self.edges[f, i_global, i_b].d1 = get_perpendicular_vector(self.edges[f, i_global, i_b].d3)
            else:
                self.edges[f, i_global, i_b].d1 = parallel_transport_normalized(
                    self.edges[f, i_global - 1, i_b].d3,
                    self.edges[f, i_global, i_b].d3,
                    self.edges[f, i_global - 1, i_b].d1,
                )
            self.edges[f, i_global, i_b].d1_ref = self.edges[f, i_global, i_b].d1

            self.edges[f, i_global, i_b].d2 = tm.cross(self.edges[f, i_global, i_b].d3, self.edges[f, i_global, i_b].d1)
            self.edges[f, i_global, i_b].d2_ref = self.edges[f, i_global, i_b].d2

            self.edges[f, i_global, i_b].theta = 0.0  # assume no initial twist

        n_internal_verts_local = n_verts_local - (0 if is_loop else 2)
        for i_iv, i_b in ti.ndrange(n_internal_verts_local, self._B):
            i_global = i_iv + iv_start
            e_s, e_e = self.get_hinge_edges(i_global)

            # state (dynamic)

            self.internal_vertices[f, i_global, i_b].kb = curvature_binormal(
                self.edges[f, e_s, i_b].d3, self.edges[f, e_e, i_b].d3
            )
            self.internal_vertices[f, i_global, i_b].twist = 0.0    # assume no initial twist

    @ti.kernel
    def _kernel_set_vertices_pos(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        pos: ti.types.ndarray(),
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.vertices[f, i_global, i_b].vert[j] = pos[i_b, i_v, j]

    @ti.kernel
    def _kernel_set_vertices_pos_grad(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        pos_grad: ti.types.ndarray(),
    ):  
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.vertices.grad[f, i_global, i_b].vert[j] = pos_grad[i_b, i_v, j]

    @ti.kernel
    def _kernel_set_vertices_vel(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        vel: ti.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.vertices[f, i_global, i_b].vel[j] = vel[i_b, i_v, j]

    @ti.kernel
    def _kernel_set_vertices_vel_grad(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        vel_grad: ti.types.ndarray(),  # shape [B, n_vertices, 3]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            for j in ti.static(range(3)):
                self.vertices.grad[f, i_global, i_b].vel[j] = vel_grad[i_b, i_v, j]

    @ti.kernel
    def _kernel_set_fixed_states(
        self,
        f: ti.i32,
        v_start: ti.i32,
        n_vertices: ti.i32,
        fixed: ti.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in ti.ndrange(n_vertices, self._B):
            i_global = i_v + v_start
            self.vertices_ng[f, i_global, i_b].fixed = fixed[i_b, i_v]

    @ti.kernel
    def _kernel_get_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),  # shape [B, n_vertices, 3]
        vel: ti.types.ndarray(),  # shape [B, n_vertices, 3]
        fixed: ti.types.ndarray(),  # shape [B, n_elements]
    ):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self.vertices[f, i_v, i_b].vert[j]
                vel[i_b, i_v, j] = self.vertices[f, i_v, i_b].vel[j]
            fixed[i_b, i_v] = self.vertices_ng[f, i_v, i_b].fixed

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        pass

    @ti.kernel
    def _kernel_set_state(
        self,
        f: ti.i32,
        pos: ti.types.ndarray(),    # shape [B, n_vertices, 3]
        vel: ti.types.ndarray(),    # shape [B, n_vertices, 3]
        fixed: ti.types.ndarray(),  # shape [B, n_vertices]
    ):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for j in ti.static(range(3)):
                self.vertices[f, i_v, i_b].vert[j] = pos[i_b, i_v, j]
                self.vertices[f, i_v, i_b].vel[j] = vel[i_b, i_v, j]
            self.vertices_ng[f, i_v, i_b].fixed = fixed[i_b, i_v]

    # ------------------------------------------------------------------------------------
    # --------------------------------- index utilities -----------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def get_edge_vertices(self, i_e: ti.i32):
        v_start = self.edges_info[i_e].vert_idx
        rod_id = self.vertices_info[v_start].rod_idx

        v_end = -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_v_start = v_start - first_vert_idx
            next_local_idx = ti.cast(tm.mod(local_v_start + 1, n_verts), ti.i32)
            v_end = first_vert_idx + next_local_idx
        else:
            v_end = v_start + 1

        return v_start, v_end

    @ti.func
    def get_hinge_edges(self, i_iv: ti.i32):
        e_start = self.internal_vertices_info[i_iv].edge_idx
        v_start_of_e_start = self.edges_info[e_start].vert_idx
        rod_id = self.vertices_info[v_start_of_e_start].rod_idx

        e_end = -1
        if self.rods_info[rod_id].is_loop:
            first_edge_idx = self.rods_info[rod_id].first_edge_idx
            n_verts = self.rods_info[rod_id].n_verts
            n_edges = n_verts - 1   # normal case
            if self.rods_info[rod_id].is_loop:
                n_edges = n_verts

            local_e_start = e_start - first_edge_idx
            next_local_idx = ti.cast(tm.mod(local_e_start + 1, n_edges), ti.i32)
            e_end = first_edge_idx + next_local_idx
        else:
            e_end = e_start + 1

        return e_start, e_end

    @ti.func
    def get_hinge_vertices(self, i_e: ti.i32):
        v_start = self.edges_info[i_e].vert_idx
        rod_id = self.vertices_info[v_start].rod_idx

        v_middle, v_end = -1, -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_v_start = v_start - first_vert_idx
            local_v_middle = ti.cast(tm.mod(local_v_start + 1, n_verts), ti.i32)
            local_v_end = ti.cast(tm.mod(local_v_start + 2, n_verts), ti.i32)

            v_middle = first_vert_idx + local_v_middle
            v_end = first_vert_idx + local_v_end
        else:
            v_middle = v_start + 1
            v_end = v_start + 2

        return v_start, v_middle, v_end

    @ti.func
    def get_next_vertex_of_edge(self, i_v: ti.i32):
        rod_id = self.vertices_info[i_v].rod_idx
        
        ip1_v = -1
        if self.rods_info[rod_id].is_loop:
            first_vert_idx = self.rods_info[rod_id].first_vert_idx
            n_verts = self.rods_info[rod_id].n_verts

            local_i_v = i_v - first_vert_idx
            next_local_idx = ti.cast(tm.mod(local_i_v + 1, n_verts), ti.i32)
            ip1_v = first_vert_idx + next_local_idx
        else:
            ip1_v = i_v + 1

        return ip1_v

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def floor_height(self):
        return self._floor_height
    
    @property
    def floor_normal(self):
        return self._floor_normal

    @property
    def damping(self):
        return self._damping

    @property
    def n_dofs(self):
        return sum([entity.n_dofs for entity in self._entities])

    @property
    def n_vertices(self):
        return sum([entity.n_vertices for entity in self._entities])

    @property
    def n_edges(self):
        return sum([entity.n_edges for entity in self._entities])

    @property
    def n_internal_vertices(self):
        return sum([entity.n_internal_vertices for entity in self._entities])

    # ------------------------------------------------------------------------------------
    # -------------------------------- pbd constraints --------------------------------
    # ------------------------------------------------------------------------------------

    @ti.func
    def _func_get_inverse_mass(self, f: ti.i32, i_v: ti.i32, i_b: ti.i32):
        mass = self.vertices_info[i_v].mass
        inv_mass = 0.0
        if self.vertices_ng[f, i_v, i_b].fixed or mass <= 0.:
            inv_mass = 0.0
        else:
            inv_mass = 1.0 / mass
        return inv_mass

    @ti.kernel
    def clear_contact_states(self):
        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            for j in ti.static(range(3)):
                self.rp_constraints[i_v, i_b].normal[j] = 0.0
            self.rp_constraints[i_v, i_b].penetration = 0.0
            self.rp_constraints[i_v, i_b].plane_idx = -1

        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            for j in ti.static(range(3)):
                self.rr_constraints[i_p, i_b].normal[j] = 0.0
            self.rr_constraints[i_p, i_b].penetration = 0.0

    @ti.kernel
    def _kernel_apply_inextensibility_constraints(self, f: ti.i32):
        for i_e, i_b in ti.ndrange(self._n_edges, self._B):
            v_s, v_e = self.get_edge_vertices(i_e)
            rod_id = self.vertices_info[v_s].rod_idx

            # check inextensibility enabled
            if not self.rods_info[rod_id].use_inextensible:
                continue
            
            inv_mass_s = self._func_get_inverse_mass(f, v_s, i_b)
            inv_mass_e = self._func_get_inverse_mass(f, v_e, i_b)
            inv_mass_sum = inv_mass_s + inv_mass_e

            if inv_mass_sum > EPS:
                p_s, p_e = self.vertices[f, v_s, i_b].vert, self.vertices[f, v_e, i_b].vert

                edge_vec = p_e - p_s
                dist = tm.length(edge_vec)

                constraint_error = dist - self.edges_info[i_e].length_rest

                if dist > EPS:
                    normal = edge_vec / dist
                    lambda_ = constraint_error / inv_mass_sum
                    delta_p_s = lambda_ * inv_mass_s * normal
                    delta_p_e = -lambda_ * inv_mass_e * normal

                    # apply corrections
                    self.vertices[f, v_s, i_b].vert += delta_p_s
                    self.vertices[f, v_e, i_b].vert += delta_p_e

    @ti.kernel
    def _kernel_apply_plane_collision_constraints(self, f: ti.i32, iter_idx: ti.i32):
        if ti.static(self._n_planes == 0):
            return

        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            inv_mass = self._func_get_inverse_mass(f, i_v, i_b)
            if inv_mass > EPS:
                pos = self.vertices[f, i_v, i_b].vert
                radius = self.vertices_info[i_v].radius

                for j in range(self._n_planes):
                    pp = self.planes_info[j].point
                    pn = self.planes_info[j].normal

                    dist = (pos - pp).dot(pn)
                    penetration = radius - dist

                    if penetration > 0.0:
                        self.vertices[f, i_v, i_b].vert += penetration * pn 

                        # TODO: currently only store contact data for the DEEPEST penetration for friction
                        if iter_idx == 0 and penetration > self.rp_constraints[i_v, i_b].penetration:
                            self.rp_constraints[i_v, i_b].normal = pn
                            self.rp_constraints[i_v, i_b].penetration = penetration
                            self.rp_constraints[i_v, i_b].plane_idx = j

    @ti.kernel
    def _kernel_apply_rod_collision_constraints(self, f: ti.i32, iter_idx: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
            idx_a2 = self.get_next_vertex_of_edge(idx_a1)
            idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
            idx_b2 = self.get_next_vertex_of_edge(idx_b1)

            p_a1, p_a2 = self.vertices[f, idx_a1, i_b].vert, self.vertices[f, idx_a2, i_b].vert
            p_b1, p_b2 = self.vertices[f, idx_b1, i_b].vert, self.vertices[f, idx_b2, i_b].vert

            radius_a = (self.vertices_info[idx_a1].radius + self.vertices_info[idx_a2].radius) * 0.5
            radius_b = (self.vertices_info[idx_b1].radius + self.vertices_info[idx_b2].radius) * 0.5

            # compute closest points (t, u) and distance
            e1, e2 = p_a2 - p_a1, p_b2 - p_b1
            e12 = p_b1 - p_a1
            d1, d2 = e1.dot(e1), e2.dot(e2)
            r = e1.dot(e2)
            s1, s2 = e1.dot(e12), e2.dot(e12)
            den = d1 * d2 - r * r

            t = 0.0
            if den > EPS:
                t = (s1 * d2 - s2 * r) / den
            t = tm.clamp(t, 0.0, 1.0)

            u_unclamped = 0.0
            if d2 > EPS:
                u_unclamped = (t * r - s2) / d2
            u = tm.clamp(u_unclamped, 0.0, 1.0)

            # re-compute t if u was clamped
            if ti.abs(u - u_unclamped) > EPS:
                if d1 > EPS:
                    t = (u * r + s1) / d1
                t = tm.clamp(t, 0.0, 1.0)

            # check for penetration
            closest_p_a = p_a1 + t * e1
            closest_p_b = p_b1 + u * e2
            dist_vec = closest_p_a - closest_p_b
            dist = tm.length(dist_vec)


            penetration = radius_a + radius_b - dist
            if penetration > 0.:
                normal = dist_vec.normalized() if dist > EPS else ti.Vector([0.0, 0.0, 1.0])

                w = ti.Vector([1.0 - t, t, 1.0 - u, u])
                im = ti.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = tm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    lambda_ = penetration / w_sum_sq_inv_mass

                    self.vertices[f, idx_a1, i_b].vert += lambda_ * im[0] * w[0] * normal
                    self.vertices[f, idx_a2, i_b].vert += lambda_ * im[1] * w[1] * normal
                    self.vertices[f, idx_b1, i_b].vert -= lambda_ * im[2] * w[2] * normal
                    self.vertices[f, idx_b2, i_b].vert -= lambda_ * im[3] * w[3] * normal
                
                if iter_idx == 0:
                    self.rr_constraints[i_p, i_b].normal = normal
                    self.rr_constraints[i_p, i_b].penetration = penetration

    @ti.kernel
    def _kernel_apply_plane_friction(self, f: ti.i32):
        if ti.static(self._n_planes == 0):
            return

        for i_v, i_b in ti.ndrange(self._n_vertices, self._B):
            plane_idx = self.rp_constraints[i_v, i_b].plane_idx
            if plane_idx != -1:
                inv_mass = self._func_get_inverse_mass(f, i_v, i_b)
                if inv_mass > EPS:
                    pn = self.rp_constraints[i_v, i_b].normal
                    v_i = self.vertices[f, i_v, i_b].vel
                    v_tangent = v_i - v_i.dot(pn) * pn
                    v_tangent_norm = tm.length(v_tangent)

                    normal_vel_mag = self.rp_constraints[i_v, i_b].penetration / self._substep_dt
                    mu_s = (self.vertices_info[i_v].mu_s + self.planes_info[plane_idx].mu_s) * 0.5
                    mu_k = (self.vertices_info[i_v].mu_k + self.planes_info[plane_idx].mu_k) * 0.5

                    if v_tangent_norm < mu_s * normal_vel_mag:
                        # static friction
                        self.vertices[f, i_v, i_b].vel -= v_tangent
                    else:
                        self.vertices[f, i_v, i_b].vel -= v_tangent.normalized() * mu_k * normal_vel_mag

    @ti.kernel
    def _kernel_apply_rod_friction(self, f: ti.i32):
        for i_p, i_b in ti.ndrange(self._n_valid_edge_pairs, self._B):
            penetration = self.rr_constraints[i_p, i_b].penetration
            if penetration > 0.0:
                idx_a1 = self.rr_constraint_info[i_p].valid_pair[0]
                idx_a2 = self.get_next_vertex_of_edge(idx_a1)
                idx_b1 = self.rr_constraint_info[i_p].valid_pair[1]
                idx_b2 = self.get_next_vertex_of_edge(idx_b1)

                p_a1, p_a2 = self.vertices[f, idx_a1, i_b].vert, self.vertices[f, idx_a2, i_b].vert
                p_b1, p_b2 = self.vertices[f, idx_b1, i_b].vert, self.vertices[f, idx_b2, i_b].vert

                # compute closest points (t, u) and distance
                e1, e2 = p_a2 - p_a1, p_b2 - p_b1
                e12 = p_b1 - p_a1
                d1, d2 = e1.dot(e1), e2.dot(e2)
                r = e1.dot(e2)
                s1, s2 = e1.dot(e12), e2.dot(e12)
                den = d1 * d2 - r * r

                t = 0.0
                if den > EPS:
                    t = (s1 * d2 - s2 * r) / den
                t = tm.clamp(t, 0.0, 1.0)

                u_unclamped = 0.0
                if d2 > EPS:
                    u_unclamped = (t * r - s2) / d2
                u = tm.clamp(u_unclamped, 0.0, 1.0)

                # Re-compute t if u was clamped
                if ti.abs(u - u_unclamped) > EPS:
                    if d1 > EPS:
                        t = (u * r + s1) / d1
                    t = tm.clamp(t, 0.0, 1.0)
                
                v_a1, v_a2 = self.vertices[f, idx_a1, i_b].vel, self.vertices[f, idx_a2, i_b].vel
                v_b1, v_b2 = self.vertices[f, idx_b1, i_b].vel, self.vertices[f, idx_b2, i_b].vel

                v_a = (1 - t) * v_a1 + t * v_a2
                v_b = (1 - u) * v_b1 + u * v_b2
                v_rel = v_a - v_b

                normal = self.rr_constraints[i_p, i_b].normal
                v_normal_mag = v_rel.dot(normal)
                v_tangent = v_rel - v_normal_mag * normal
                v_tangent_norm = tm.length(v_tangent)

                w = ti.Vector([1.0 - t, t, 1.0 - u, u])
                im = ti.Vector([
                    self._func_get_inverse_mass(f, idx_a1, i_b),
                    self._func_get_inverse_mass(f, idx_a2, i_b),
                    self._func_get_inverse_mass(f, idx_b1, i_b),
                    self._func_get_inverse_mass(f, idx_b2, i_b),
                ])

                w_sum_sq_inv_mass = tm.dot(w * w, im)
                if w_sum_sq_inv_mass > EPS:
                    normal_vel_mag = penetration / self._substep_dt

                    mu_s = (self.vertices_info[idx_a1].mu_s + self.vertices_info[idx_a2].mu_s + self.vertices_info[idx_b1].mu_s + self.vertices_info[idx_b2].mu_s) * 0.25
                    mu_k = (self.vertices_info[idx_a1].mu_k + self.vertices_info[idx_a2].mu_k + self.vertices_info[idx_b1].mu_k + self.vertices_info[idx_b2].mu_k) * 0.25

                    delta_v_tangent = ti.Vector.zero(gs.ti_float, 3)
                    if v_tangent_norm < mu_s * normal_vel_mag:
                        delta_v_tangent = -v_tangent
                    else:
                        delta_v_tangent = -v_tangent.normalized() * mu_k * normal_vel_mag

                    lambda_ = delta_v_tangent / w_sum_sq_inv_mass
                    self.vertices[f, idx_a1, i_b].vel += lambda_ * im[0] * w[0]
                    self.vertices[f, idx_a2, i_b].vel += lambda_ * im[1] * w[1]
                    self.vertices[f, idx_b1, i_b].vel -= lambda_ * im[2] * w[2]
                    self.vertices[f, idx_b2, i_b].vel -= lambda_ * im[3] * w[3]
