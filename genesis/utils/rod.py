import numpy as np
import trimesh


def mesh_from_centerline(
    verts: np.ndarray,
    radii: np.ndarray,
    radial_segs=16,
    cap_segs=8,
    endcaps=True,
    is_loop=False
) -> trimesh.Trimesh:
    """
    Build a tube mesh with rounded ends around a polyline (rod centerline).
    This implementation uses Parallel Transport Frames to create a smooth,
    twist-free mesh.

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
        If True, close the ends with hemispherical caps. Ignored if is_loop is True.
    is_loop : bool
        If True, connect the ends to form a closed loop (toroid-like).

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    verts = np.asarray(verts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    N = len(verts)
    if N < 2:
        raise ValueError("Need at least 2 vertices for a rod")
    if is_loop and N < 3:
        raise ValueError("Need at least 3 vertices for a loop")
    if verts.shape[0] != radii.shape[0]:
        raise ValueError("verts and radii must have the same length")

    V_list = []
    F_list = []
    ring_indices_list = []
    basis_list = [] # Store the (normal, binormal) basis for each ring

    # --- Frame Propagation using Double Reflection ---
    # This method creates a smooth, continuous orientation along the tube.
    prev_tangent = None
    prev_normal = None

    for i in range(N):
        # Calculate tangent for the current vertex
        if is_loop:
            # For loops, use wrapped indexing for a smooth tangent at the seam
            prev_idx = (i - 1 + N) % N
            next_idx = (i + 1) % N
            tangent = verts[next_idx] - verts[prev_idx]
        else:
            # Existing logic for open rods
            if i == 0:
                tangent = verts[1] - verts[0]
            elif i == N - 1:
                tangent = verts[-1] - verts[-2]
            else:
                tangent = verts[i + 1] - verts[i - 1]
        
        tangent_norm = tangent / np.linalg.norm(tangent)

        # Calculate the local coordinate system (basis) for the ring
        if i == 0:
            # For the first vertex, create an arbitrary initial basis
            # Pick a helper vector not parallel to tangent
            helper = np.array([0, 0, 1]) if abs(tangent_norm[2]) < 0.9 else np.array([0, 1, 0])
            normal = np.cross(tangent_norm, helper)
            normal /= np.linalg.norm(normal)
        else:
            # For subsequent vertices, transport the previous frame to the current one
            # This minimizes twisting artifacts.
            v = prev_tangent + tangent_norm
            v_dot_v = np.dot(v, v)
            if v_dot_v > 1e-8: # Avoid division by zero if tangents are opposite
                reflection_vec = 2 * np.dot(prev_normal, v) / v_dot_v
                normal = prev_normal - v * reflection_vec
            else:
                # Tangents are nearly opposite (a 180-degree bend)
                # Rotate the previous normal by 180 degrees around the previous tangent
                normal = -prev_normal
            
        binormal = np.cross(tangent_norm, normal)
        basis_list.append((normal, binormal))
        
        prev_tangent = tangent_norm
        prev_normal = normal

        # Generate the ring of vertices using the calculated basis
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
        ring1 = ring_indices_list[i + 1]
        for j in range(radial_segs):
            a = ring0[j]
            b = ring0[(j + 1) % radial_segs]
            c = ring1[j]
            d = ring1[(j + 1) % radial_segs]
            F_list.append([a, b, c])
            F_list.append([d, c, b])

    # If it's a loop, connect the last ring back to the first
    if is_loop:
        ring0 = ring_indices_list[N - 1]
        ring1 = ring_indices_list[0]
        for j in range(radial_segs):
            a = ring0[j]
            b = ring0[(j + 1) % radial_segs]
            c = ring1[j]
            d = ring1[(j + 1) % radial_segs]
            F_list.append([a, b, c])
            F_list.append([d, c, b])

    if not is_loop and endcaps and cap_segs > 0:
        # START CAP
        start_center = verts[0]
        start_radius = radii[0]
        tangent_start_norm = (verts[0] - verts[1]) / np.linalg.norm(verts[0] - verts[1])
        normal_start, binormal_start = basis_list[0]
        
        prev_ring_indices = ring_indices_list[0]
        for k in range(1, cap_segs + 1):
            alpha = k * (np.pi / 2) / cap_segs
            ring_radius = start_radius * np.cos(alpha)
            displacement = start_radius * np.sin(alpha)
            ring_center = start_center + displacement * tangent_start_norm
            
            is_pole = k == cap_segs
            current_ring_indices = []
            if not is_pole:
                for j in range(radial_segs):
                    theta = 2 * np.pi * j / radial_segs
                    offset = np.cos(theta) * normal_start + np.sin(theta) * binormal_start
                    V_list.append(ring_center + ring_radius * offset)
                    current_ring_indices.append(len(V_list) - 1)
            else:
                V_list.append(ring_center)
                pole_index = len(V_list) - 1
                current_ring_indices = [pole_index] * radial_segs

            for j in range(radial_segs):
                a = prev_ring_indices[j]
                b = prev_ring_indices[(j + 1) % radial_segs]
                c = current_ring_indices[j]
                d = current_ring_indices[(j + 1) % radial_segs]
                if not is_pole: F_list.append([a, c, b]); F_list.append([d, b, c])
                else: F_list.append([b, c, a]) # Reversed winding for start cap pole
            prev_ring_indices = current_ring_indices

        # END CAP
        end_center = verts[-1]
        end_radius = radii[-1]
        tangent_end_norm = (verts[-1] - verts[-2]) / np.linalg.norm(verts[-1] - verts[-2])
        normal_end, binormal_end = basis_list[-1]

        prev_ring_indices = ring_indices_list[-1]
        for k in range(1, cap_segs + 1):
            alpha = k * (np.pi / 2) / cap_segs
            ring_radius = end_radius * np.cos(alpha)
            displacement = end_radius * np.sin(alpha)
            ring_center = end_center + displacement * tangent_end_norm
            
            is_pole = k == cap_segs
            current_ring_indices = []
            if not is_pole:
                for j in range(radial_segs):
                    theta = 2 * np.pi * j / radial_segs
                    offset = np.cos(theta) * normal_end + np.sin(theta) * binormal_end
                    V_list.append(ring_center + ring_radius * offset)
                    current_ring_indices.append(len(V_list) - 1)
            else:
                V_list.append(ring_center)
                pole_index = len(V_list) - 1
                current_ring_indices = [pole_index] * radial_segs

            for j in range(radial_segs):
                a = prev_ring_indices[j]
                b = prev_ring_indices[(j + 1) % radial_segs]
                c = current_ring_indices[j]
                d = current_ring_indices[(j + 1) % radial_segs]
                if not is_pole: F_list.append([a, b, c]); F_list.append([d, c, b])
                else: F_list.append([a, b, c])
            prev_ring_indices = current_ring_indices

    V = np.array(V_list)
    F = np.array(F_list, dtype=int)
    
    return trimesh.Trimesh(vertices=V, faces=F, process=True)
