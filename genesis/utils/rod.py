import numpy as np
import trimesh

def mesh_from_centerline(
    verts: np.ndarray,
    radius=0.01,
    radial_segs=16,
    endcaps=True
) -> trimesh.Trimesh:
    """
    Build a tube mesh around a polyline (rod centerline).

    Parameters
    ----------
    verts : (N,3) ndarray
        Sequence of 3D points along the rod centerline.
    radius : float
        Tube radius (constant).
    radial_segs : int
        Number of segments around the circle.
    endcaps : bool
        If True, close the ends with disks.

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    verts = np.asarray(verts, dtype=float)
    N = len(verts)
    if N < 2:
        raise ValueError("Need at least 2 vertices for a rod")

    all_vertices = []
    all_faces = []
    vertex_count = 0

    # initial frame: tangent is along segment, pick arbitrary normal
    def orthonormal_basis(tangent):
        tangent = tangent / np.linalg.norm(tangent)
        # pick a helper vector not parallel to tangent
        helper = np.array([0,0,1]) if abs(tangent[2]) < 0.9 else np.array([0,1,0])
        normal = np.cross(tangent, helper)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        return tangent, normal, binormal

    prev_ring = None
    for i in range(N):
        if i == 0:
            tangent = verts[1] - verts[0]
        elif i == N-1:
            tangent = verts[-1] - verts[-2]
        else:
            tangent = verts[i+1] - verts[i-1]
        tangent, normal, binormal = orthonormal_basis(tangent)

        # ring around verts[i]
        ring = []
        for j in range(radial_segs):
            theta = 2*np.pi*j/float(radial_segs)
            offset = np.cos(theta)*normal + np.sin(theta)*binormal
            ring.append(verts[i] + radius*offset)
        ring = np.array(ring)
        all_vertices.append(ring)

        if prev_ring is not None:
            start0 = vertex_count - radial_segs
            start1 = vertex_count
            for j in range(radial_segs):
                a = start0 + j
                b = start0 + (j+1)%radial_segs
                c = start1 + j
                d = start1 + (j+1)%radial_segs
                all_faces.append([a,b,c])
                all_faces.append([d,c,b])
        vertex_count += radial_segs
        prev_ring = ring

    # optional endcaps
    if endcaps:
        # start cap
        center0 = len(np.vstack(all_vertices))
        all_vertices.append(verts[0].reshape(1,3))
        start0 = 0
        for j in range(radial_segs):
            a = center0
            b = start0 + j
            c = start0 + (j+1)%radial_segs
            all_faces.append([a,b,c])
        # end cap
        center1 = center0 + 1
        all_vertices.append(verts[-1].reshape(1,3))
        start1 = vertex_count - radial_segs
        for j in range(radial_segs):
            a = center1
            b = start1 + (j+1)%radial_segs
            c = start1 + j
            all_faces.append([a,b,c])

    V = np.vstack(all_vertices)
    F = np.array(all_faces, dtype=int)

    return trimesh.Trimesh(vertices=V, faces=F, process=True)
