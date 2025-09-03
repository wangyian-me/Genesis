import numpy as np
import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.states.entities import RODEntityState
from genesis.utils.misc import ALLOCATE_TENSOR_WARNING, to_gs_tensor, tensor_to_array

from .base_entity import Entity


@ti.data_oriented
class RodEntity(Entity):
    """
    A discrete linear object (DLO)-based entity for rod simulation.

    This class represents a deformable object using tetrahedral elements. It interfaces with
    the physics solver to handle state updates, checkpointing, gradients, and actuation
    for physics-based simulation in batched environments.

    Parameters
    ----------
    scene : Scene
        The simulation scene that this entity belongs to.
    solver : Solver
        The physics solver instance used for simulation.
    material : Material
        The material properties defining elasticity, density, etc.
    morph : Morph
        The morph specification that defines the entity's shape.
    surface : Surface
        The surface mesh associated with the entity (for rendering or collision).
    idx : int
        Unique identifier of the entity within the scene.
    rod_idx : int, optional
        Index of this rod in the solver (default is 0).
    v_start : int, optional
        Starting index of this entity's vertices in the global vertex array (default is 0).
    e_start : int, optional
        Starting index of this entity's edges in the global edge array (default is 0).
    iv_start : int, optional
        Starting index of this entity's internal vertices in the global internal vertex array (default is 0).
    """

    def __init__(
        self, scene, solver, material, morph, surface, idx, 
        rod_idx=0, v_start=0, e_start=0, iv_start=0
    ):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._rod_idx = rod_idx     # index of this rod in the solver
        self._v_start = v_start     # offset for vertex index
        self._e_start = e_start     # offset for edge index
        self._iv_start = iv_start   # offset for internal vertex index
        self._step_global_added = None

        self.sample()

        self.init_tgt_vars()

        self.active = False  # This attribute is only used in forward pass. It should NOT be used during backward pass.

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic entity ops -------------------------------
    # ------------------------------------------------------------------------------------

    def set_position(self, pos):
        """
        Set the target position(s) for the Rod entity.

        Parameters
        ----------
        pos : torch.Tensor or array-like
            The desired position(s). Can be:
            - (3,): a single COM offset vector.
            - (n_vertices, 3): per-vertex positions for all vertices.
            - (n_envs, 3): per-environment COM offsets.
            - (n_envs, n_vertices, 3): full batched per-vertex positions.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element positions. This is not recommended and could break gradient flow.")

        pos = to_gs_tensor(pos)

        is_valid = False
        if pos.ndim == 1:
            if pos.shape == (3,):
                pos = self.init_positions_COM_offset + pos
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
        elif pos.ndim == 2:
            if pos.shape == (self.n_vertices, 3):
                self._tgt["pos"] = pos.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif pos.shape == (self._sim._B, 3):
                pos = self.init_positions_COM_offset.unsqueeze(0) + pos.unsqueeze(1)
                self._tgt["pos"] = pos
                is_valid = True
        elif pos.ndim == 3:
            if pos.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["pos"] = pos
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def set_velocity(self, vel):
        """
        Set the target velocity(ies) for the Rod entity.

        Parameters
        ----------
        vel : torch.Tensor or array-like
            The desired velocity(ies). Can be:
            - (3,): a global velocity vector for all vertices.
            - (n_vertices, 3): per-vertex velocities.
            - (n_envs, 3): per-environment velocities broadcast to all vertices.
            - (n_envs, n_vertices, 3): full batched per-vertex velocities.

        Raises
        ------
        Exception
            If the tensor shape is not supported.
        """
        self._assert_active()
        gs.logger.warning("Manually setting element velocities. This is not recommended and could break gradient flow.")

        vel = to_gs_tensor(vel)

        is_valid = False
        if vel.ndim == 1:
            if vel.shape == (3,):
                self._tgt["vel"] = vel.tile((self._sim._B, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 2:
            if vel.shape == (self.n_vertices, 3):
                self._tgt["vel"] = vel.unsqueeze(0).tile((self._sim._B, 1, 1))
                is_valid = True
            elif vel.shape == (self._sim._B, 3):
                self._tgt["vel"] = vel.unsqueeze(1).tile((1, self.n_vertices, 1))
                is_valid = True
        elif vel.ndim == 3:
            if vel.shape == (self._sim._B, self.n_vertices, 3):
                self._tgt["vel"] = vel
                is_valid = True
        if not is_valid:
            gs.raise_exception("Tensor shape not supported.")

    def get_state(self):
        state = RODEntityState(self, self._sim.cur_step_global)
        self.get_frame(
            self._sim.cur_substep_local,
            state.pos,
            state.vel
        )

        # we store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    def deactivate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self.id}> deactivated.")
        self._tgt["act"] = gs.INACTIVE
        self.active = False

    def activate(self):
        gs.logger.info(f"{self.__class__.__name__} <{self.id}> activated.")
        self._tgt["act"] = gs.ACTIVE
        self.active = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def instantiate(self, verts):
        """
        Initialize Rod entity with given vertices.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertex positions with shape (n_vertices, 3).

        Raises
        ------
        Exception
            If no vertices are provided.
        """
        verts = verts.astype(gs.np_float, copy=False)
        n_verts = verts.shape[0]

        # rotate
        R = gu.quat_to_R(np.array(self.morph.quat, dtype=gs.np_float))
        verts_COM = verts.mean(axis=0)
        init_positions = (verts - verts_COM) @ R.T + verts_COM

        if not init_positions.shape[0] > 0:
            gs.raise_exception(f"Entity has zero vertices.")

        self.init_positions = gs.tensor(init_positions)
        self.init_positions_COM_offset = self.init_positions - gs.tensor(verts_COM)

        edges = list()
        is_loop = self.morph.is_loop
        n_edges = n_verts if is_loop else n_verts - 1
        for i in range(n_edges):
            # NOTE: check this
            edges.append(verts[(i + 1) % n_verts] - verts[i])
        edges = np.array(edges, dtype=gs.np_float)

        self.edges = edges

    def sample(self):
        """
        Sample mesh and elements based on the entity's morph type.

        Raises
        ------
        Exception
            If the morph type is unsupported.
        """

        vertices = np.load(self.morph.file)
        assert vertices.ndim == 2, f"Loaded vertices should be of shape (n_vertices, 3), got {vertices.shape}."
        assert vertices.shape[1] == 3, f"Loaded vertices should be of shape (n_vertices, 3), got {vertices.shape}."
        vertices = vertices + self.morph.pos

        self.instantiate(vertices)

    def _add_to_solver(self, in_backward=False):
        if not in_backward:
            self._step_global_added = self._sim.cur_step_global
            gs.logger.info(
                f"Entity {self.uid}({self._rod_idx}) added. class: {self.__class__.__name__}, morph: {self.morph.__class__.__name__}, #verts: {self.n_vertices}, loop: {self.morph.is_loop}, material: {self.material}."
            )

        # Convert to appropriate numpy array types
        verts_np = tensor_to_array(self.init_positions, dtype=gs.np_float)
        edges_np = tensor_to_array(self.edges, dtype=gs.np_float)

        # TODO: @junyi, maybe want to add more material parameters or others
        self._solver._kernel_add_rods(
            rod_idx=self._rod_idx,
            is_loop=self.morph.is_loop,
            use_inextensible=self.material.use_inextensible,
            stretching_stiffness=self.material.K,
            bending_stiffness=self.material.E,
            twisting_stiffness=self.material.G,
            plastic_yield=self.material.plastic_yield,
            plastic_creep=self.material.plastic_creep,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            n_verts=self.n_vertices,
        )

        self._solver._kernel_finalize_rest_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            verts_rest=verts_np,
            edges_rest=edges_np,
        )

        self._solver._kernel_finalize_states(
            f=self._sim.cur_substep_local,
            rod_idx=self._rod_idx,
            v_start=self._v_start,
            e_start=self._e_start,
            iv_start=self._iv_start,
            segment_mass=self.material.segment_mass,
            segment_radius=self.material.segment_radius,
            static_friction=self.material.static_friction,
            kinetic_friction=self.material.kinetic_friction,
            verts=verts_np,
            edges=edges_np,
        )
        self.active = True

    # ------------------------------------------------------------------------------------
    # ---------------------------- checkpoint and buffer ---------------------------------
    # ------------------------------------------------------------------------------------

    def init_tgt_keys(self):
        """
        Initialize the keys used in target state management.

        This defines which physical properties (e.g., position, velocity) will be tracked for checkpointing and buffering.
        """
        self._tgt_keys = ["vel", "pos", "act"]

    def init_tgt_vars(self):
        """
        Initialize the target state variables and their buffers.

        This sets up internal dictionaries to store per-step target values for properties like velocity, position, actuation, and activation.
        """

        # temp variable to store targets for next step
        self._tgt = dict()
        self._tgt_buffer = dict()
        self.init_tgt_keys()

        for key in self._tgt_keys:
            self._tgt[key] = None
            self._tgt_buffer[key] = list()

    def process_input(self, in_backward=False):
        """
        Push position, velocity, and activation target states into the simulator.

        Parameters
        ----------
        in_backward : bool, default=False
            Whether the simulation is in the backward (gradient) pass.
        """
        # TODO: implement this
        pass

    def _assert_active(self):
        if not self.active:
            gs.raise_exception(f"{self.__class__.__name__} is inactive. Call `entity.activate()` first.")

    # ------------------------------------------------------------------------------------
    # ---------------------------- interfacing with solver -------------------------------
    # ------------------------------------------------------------------------------------

    def set_pos(self, f, pos):
        """
        Set element positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing new positions.
        """

        self._solver._kernel_set_vertices_pos(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos=pos,
        )

    def set_pos_grad(self, f, pos_grad):
        """
        Set gradient of element positions in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        pos_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of positions.
        """

        self._solver._kernel_set_vertices_pos_grad(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            pos_grad=pos_grad,
        )

    def set_vel(self, f, vel):
        """
        Set element velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing velocities.
        """

        self._solver._kernel_set_vertices_vel(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel=vel,
        )

    def set_vel_grad(self, f, vel_grad):
        """
        Set gradient of element velocities in the solver.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        vel_grad : gs.Tensor
            Tensor of shape (n_envs, n_vertices, 3) containing gradients of velocities.
        """

        self._solver._kernel_set_vertices_vel_grad(
            f=f,
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            vel_grad=vel_grad,
        )

    @gs.assert_built
    def set_fixed_states(self, fixed_states=None, fixed_ids=None):
        """
        Set the fixed status of each vertex.

        Parameters
        ----------
        f : int
            Current substep/frame index.

        fixed_states : gs.Tensor
            Tensor of shape (n_envs, n_vertices).
        """

        if fixed_ids is None and fixed_states is None:
            is_fixed = np.zeros(self.n_vertices, dtype=gs.np_bool)
        elif fixed_ids is None and fixed_states is not None:
            is_fixed = np.asarray(fixed_states).copy().reshape(-1).astype(gs.np_bool)
            assert is_fixed.shape[0] == self.n_vertices, \
                f"Fixed states has {is_fixed.shape[0]} vertices, but rod {self._rod_idx} has {self.n_vertices}."
        elif fixed_ids is not None:
            is_fixed = [1 if i in fixed_ids else 0 for i in range(self.n_vertices)]
            is_fixed = np.array(is_fixed, dtype=gs.np_bool)
        else:
            raise ValueError("`fixed_ids` and `fixed_states` cannot be provided at the same time.")

        self._solver._kernel_set_fixed_states(
            v_start=self._v_start,
            n_vertices=self.n_vertices,
            fixed=is_fixed,
        )

    @ti.kernel
    def _kernel_get_verts_pos(self, f: ti.i32, pos: ti.types.ndarray(), verts_idx: ti.types.ndarray()):
        # get current position of vertices
        for i_v, i_b in ti.ndrange(verts_idx.shape[0], verts_idx.shape[1]):
            i_global = verts_idx[i_v, i_b] + self.v_start
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vert[j]

    @ti.kernel
    def get_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray()):
        """
        Fetch the position, velocity, and activation state of the Rod entity at a specific substep.

        Parameters
        ----------
        f : int
            The substep/frame index to fetch the state from.

        pos : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store positions.

        vel : np.ndarray
            Output array of shape (n_envs, n_vertices, 3) to store velocities.
        """

        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            for j in ti.static(range(3)):
                pos[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vert[j]
                vel[i_b, i_v, j] = self._solver.vertices[f, i_global, i_b].vel[j]

    @ti.kernel
    def clear_grad(self, f: ti.i32):
        """
        Zero out the gradients of position, velocity, and actuation for the current substep.

        Parameters
        ----------
        f : int
            The substep/frame index for which to clear gradients.

        Notes
        -----
        This method is primarily used during backward passes to manually reset gradients
        that may be corrupted by explicit state setting.
        """
        # TODO: not well-tested
        for i_v, i_b in ti.ndrange(self.n_vertices, self._sim._B):
            i_global = i_v + self.v_start
            self._solver.vertices.grad[f, i_global, i_b].vert = 0
            self._solver.vertices.grad[f, i_global, i_b].vel = 0

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_vertices(self):
        """Number of vertices in the Rod entity."""
        return len(self.init_positions)

    @property
    def n_edges(self):
        """Number of edges in the Rod entity."""
        return len(self.edges)

    @property
    def n_internal_vertices(self):
        """Number of internal vertices in the Rod entity."""
        return len(self.init_positions) if self.morph.is_loop else len(self.init_positions) - 2

    @property
    def n_dofs(self):
        """Number of degrees of freedom (DOFs) in the Rod entity."""
        # 3 for each vertex + 1 for each edge
        return 3 * self.n_vertices + self.n_edges

    @property
    def v_start(self):
        """Global vertex index offset for this entity."""
        return self._v_start

    @property
    def e_start(self):
        """Global edge index offset for this entity."""
        return self._e_start

    @property
    def iv_start(self):
        """Global internal vertex index offset for this entity."""
        return self._iv_start

    @property
    def material(self):
        """Material properties of the Rod entity."""
        return self._material
