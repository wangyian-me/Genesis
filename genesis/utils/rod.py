import numpy as np
import trimesh


def mesh_from_centerline(
    verts: np.ndarray,
    radii: np.ndarray,
    radial_segs=16,
    cap_segs=8,
    endcaps=True
) -> trimesh.Trimesh:
    """
    Build a tube mesh with rounded ends around a polyline (rod centerline).

    Parameters
    ----------
    verts : (N,3) ndarray
        Sequence of 3D points along the rod centerline.
    radii : (N,) ndarray
        Radii at each vertex.
    radial_segs : int
        Number of segments around the tube's circumference.
    cap_segs : int
        Number of segments for the hemispherical end caps (from base to pole).
    endcaps : bool
        If True, close the ends with hemispherical caps.

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    verts = np.asarray(verts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    N = len(verts)
    if N < 2:
        raise ValueError("Need at least 2 vertices for a rod")
    if verts.shape[0] != radii.shape[0]:
        raise ValueError("verts and radii must have the same length")

    # Helper to create a stable orthonormal basis from a tangent vector
    def orthonormal_basis(tangent):
        tangent = tangent / np.linalg.norm(tangent)
        # Pick a helper vector not parallel to tangent
        helper = np.array([0, 0, 1]) if abs(tangent[2]) < 0.9 else np.array([0, 1, 0])
        normal = np.cross(tangent, helper)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        return tangent, normal, binormal

    V_list = []
    F_list = []
    
    # Store the indices of vertices for each ring along the tube
    ring_indices_list = []
    first_ring_basis = None
    last_ring_basis = None

    for i in range(N):
        # Calculate tangent to determine the orientation of the ring
        if i == 0:
            tangent = verts[1] - verts[0]
        elif i == N - 1:
            tangent = verts[-1] - verts[-2]
        else:
            # Use the average of adjacent segments for a smoother transition at joints
            tangent = (verts[i+1] - verts[i-1])
        
        _, normal, binormal = orthonormal_basis(tangent)
        
        if i == 0:
            first_ring_basis = (normal, binormal)
        if i == N - 1:
            last_ring_basis = (normal, binormal)

        # Generate the ring of vertices
        current_ring_indices = []
        for j in range(radial_segs):
            theta = 2 * np.pi * j / radial_segs
            offset = np.cos(theta) * normal + np.sin(theta) * binormal
            V_list.append(verts[i] + radii[i] * offset)
            current_ring_indices.append(len(V_list) - 1)
        ring_indices_list.append(current_ring_indices)

    # Connect the rings to form the tube walls
    for i in range(N - 1):
        ring0 = ring_indices_list[i]
        ring1 = ring_indices_list[i+1]
        for j in range(radial_segs):
            a = ring0[j]
            b = ring0[(j + 1) % radial_segs]
            c = ring1[j]
            d = ring1[(j + 1) % radial_segs]
            F_list.append([a, b, c])
            F_list.append([d, c, b])

    if endcaps and cap_segs > 0:
        # START CAP
        start_center = verts[0]
        start_radius = radii[0]
        tangent_start = (verts[0] - verts[1]) # Outward-pointing tangent
        tangent_start_norm, _, _ = orthonormal_basis(tangent_start)
        normal_start, binormal_start = first_ring_basis # Use stored basis
        
        prev_ring_indices = ring_indices_list[0]
        
        # Add latitude rings for the cap, moving towards the pole
        for k in range(1, cap_segs + 1):
            alpha = k * (np.pi / 2) / cap_segs
            ring_radius = start_radius * np.cos(alpha)
            displacement = start_radius * np.sin(alpha)
            ring_center = start_center + displacement * tangent_start_norm
            
            current_ring_indices = []
            if k < cap_segs: # Not the pole yet
                for j in range(radial_segs):
                    theta = 2 * np.pi * j / radial_segs
                    offset = np.cos(theta) * normal_start + np.sin(theta) * binormal_start
                    V_list.append(ring_center + ring_radius * offset)
                    current_ring_indices.append(len(V_list) - 1)
            else: # Add the pole vertex
                V_list.append(ring_center)
                pole_index = len(V_list) - 1
                current_ring_indices = [pole_index] * radial_segs


            # Connect new ring to previous one (note reversed winding for start cap)
            for j in range(radial_segs):
                a = prev_ring_indices[j]
                b = prev_ring_indices[(j + 1) % radial_segs]
                c = current_ring_indices[j]
                d = current_ring_indices[(j + 1) % radial_segs]
                if k < cap_segs:
                    F_list.append([a, c, b])
                    F_list.append([d, b, c])
                else: # Connect to pole
                    F_list.append([b, c, a])
            
            prev_ring_indices = current_ring_indices

        # END CAP
        end_center = verts[-1]
        end_radius = radii[-1]
        tangent_end = (verts[-1] - verts[-2]) # Outward-pointing tangent
        tangent_end_norm, _, _ = orthonormal_basis(tangent_end)
        normal_end, binormal_end = last_ring_basis # Use stored basis

        prev_ring_indices = ring_indices_list[-1]
        
        # Add latitude rings
        for k in range(1, cap_segs + 1):
            alpha = k * (np.pi / 2) / cap_segs
            ring_radius = end_radius * np.cos(alpha)
            displacement = end_radius * np.sin(alpha)
            ring_center = end_center + displacement * tangent_end_norm
            
            current_ring_indices = []
            if k < cap_segs: # Not the pole yet
                for j in range(radial_segs):
                    theta = 2 * np.pi * j / radial_segs
                    offset = np.cos(theta) * normal_end + np.sin(theta) * binormal_end
                    V_list.append(ring_center + ring_radius * offset)
                    current_ring_indices.append(len(V_list) - 1)
            else: # Add the pole vertex
                V_list.append(ring_center)
                pole_index = len(V_list) - 1
                current_ring_indices = [pole_index] * radial_segs

            # Connect new ring to previous one (standard winding order)
            for j in range(radial_segs):
                a = prev_ring_indices[j]
                b = prev_ring_indices[(j + 1) % radial_segs]
                c = current_ring_indices[j]
                d = current_ring_indices[(j + 1) % radial_segs]

                if k < cap_segs:
                    F_list.append([a, b, c])
                    F_list.append([d, c, b])
                else: # Connect to pole
                    F_list.append([a, b, c])

            prev_ring_indices = current_ring_indices

    V = np.array(V_list)
    F = np.array(F_list, dtype=int)
    
    return trimesh.Trimesh(vertices=V, faces=F, process=True)
