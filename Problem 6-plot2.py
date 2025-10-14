import numpy as np
import matplotlib.pyplot as plt

# ===================== USER INPUT =====================
coordinates = np.array([
    [0.,  0.],
    [4.,  0.],
    [8.,  0.],
    [4., -6.]
])

# 0-based pairs of node indices
connectivity = np.array([
    [0, 3],
    [1, 3],
    [2, 3]
])

# Material/section (can be scalars; per-element arrays also supported)
E = 1.0      # Young's modulus
A = 1.0      # Cross-sectional area
I = 1.0      # Second moment of area

# supports: 1 = constrained, 0 = free, per node [ux, uy, theta]
supports = np.array([
    [1, 1, 1],   # node 0 fixed
    [1, 1, 1],   # node 1 fixed
    [1, 1, 1],   # node 2 fixed
    [0, 0, 0]    # node 3 free
])

# Nodal loads in GLOBAL axes per node: [Fx, Fy, Mz]
applied_loads = np.array([
    [0.,   0.,   0.],
    [0.,   0.,   0.],
    [0.,   0.,   0.],
    [100., -100., 0.]   # at node 3
])
# ======================================================


# ===================== FE CORE ========================
def length_cos_sin(x1, y1, x2, y2):
    L = np.hypot(x2 - x1, y2 - y1)
    c = (x2 - x1) / L
    s = (y2 - y1) / L
    return L, c, s

def ke_local_frame(E, A, I, L):
    """
    2D frame (Euler–Bernoulli) local stiffness (6x6)
    Local DOF order: [u1, v1, th1, u2, v2, th2]
    """
    k = np.zeros((6, 6), dtype=float)
    EA = E * A
    EI = E * I

    # axial (u)
    k[0,0] =  EA/L;  k[0,3] = -EA/L
    k[3,0] = -EA/L;  k[3,3] =  EA/L

    # bending (v, theta)
    k[1,1] =  12*EI/L**3;  k[1,2] =  6*EI/L**2;  k[1,4] = -12*EI/L**3;  k[1,5] =  6*EI/L**2
    k[2,1] =   6*EI/L**2;  k[2,2] =  4*EI/L;     k[2,4] =  -6*EI/L**2;  k[2,5] =  2*EI/L
    k[4,1] = -12*EI/L**3;  k[4,2] = -6*EI/L**2;  k[4,4] =  12*EI/L**3;  k[4,5] = -6*EI/L**2
    k[5,1] =   6*EI/L**2;  k[5,2] =  2*EI/L;     k[5,4] =  -6*EI/L**2;  k[5,5] =  4*EI/L
    return k

def T_global_to_local(c, s):
    """
    DOF transformation so that q_local = T * q_global (global -> local).
    (Use Kg = T^T Kl T for assembly; u_local = T u_global for recovery.)
    """
    R_T = np.array([[ c,  s, 0],
                    [-s,  c, 0],
                    [ 0,  0, 1]], dtype=float)
    T = np.zeros((6, 6), dtype=float)
    T[:3, :3] = R_T
    T[3:, 3:] = R_T
    return T

def assemble(coordinates, connectivity, E, A, I, applied_loads):
    """
    Assembles global stiffness and load vector.
    Returns KG, F, node_dof_map, elem_cache.
    """
    coords = np.asarray(coordinates, float)
    N = len(coords)
    Ne = len(connectivity)
    ndofs = 3 * N

    # Allow scalar or per-element arrays
    E_ = np.full(Ne, E, float) if np.isscalar(E) else np.asarray(E, float)
    A_ = np.full(Ne, A, float) if np.isscalar(A) else np.asarray(A, float)
    I_ = np.full(Ne, I, float) if np.isscalar(I) else np.asarray(I, float)

    node_dof = np.arange(ndofs, dtype=int).reshape(N, 3)
    elem_dofs = np.zeros((Ne, 6), dtype=int)
    for e, (n1, n2) in enumerate(connectivity):
        elem_dofs[e] = [3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2]

    KG = np.zeros((ndofs, ndofs), dtype=float)
    F  = np.zeros(ndofs, dtype=float)

    elem_cache = []  # (Kl, T_gl2loc, dofs, L, c, s, (x1,y1,x2,y2))
    for e, (n1, n2) in enumerate(connectivity):
        x1, y1 = coords[n1]; x2, y2 = coords[n2]
        L, c, s = length_cos_sin(x1, y1, x2, y2)

        Kl = ke_local_frame(E_[e], A_[e], I_[e], L)
        T  = T_global_to_local(c, s)
        Kg = T.T @ Kl @ T

        dofs = elem_dofs[e]
        KG[np.ix_(dofs, dofs)] += Kg
        elem_cache.append((Kl, T, dofs, L, c, s, (x1, y1, x2, y2)))

    # nodal loads
    loads = np.asarray(applied_loads, float)
    for n in range(N):
        Fx, Fy, M = loads[n]
        F[3*n+0] += Fx
        F[3*n+1] += Fy
        F[3*n+2] += M

    return KG, F, node_dof, elem_cache

def apply_bc_and_solve(KG, F, supports, node_dof):
    """
    Imposes essential BCs by row/column zeroing + unit diagonal and solves.
    Returns (U, Reactions).
    """
    KT = KG.copy()
    bF = F.copy()
    for n in range(len(supports)):
        for k in range(3):
            if supports[n, k] == 1:
                d = node_dof[n, k]
                KT[d, :] = 0.0
                KT[:, d] = 0.0
                KT[d, d] = 1.0
                bF[d] = 0.0

    U = np.linalg.solve(KT, bF)
    R = KG @ U - F  # reactions (nonzero mainly at constrained dofs)
    return U, R
# ======================================================


# ===================== PLOTTING =======================
def plot_nodes_only(coordinates, connectivity, U, supports=None, scale=None,
                    title="Frame: nodes-only (undeformed vs deformed)"):
    """
    Draw straight members between nodes (undeformed), and the same lines
    using ONLY nodal translations (deformed). Rotations are not drawn.
    """
    coords = np.asarray(coordinates, float)
    N = len(coords)
    ux = U[0::3]; uy = U[1::3]

    # characteristic length & auto-scale
    if len(connectivity):
        Lc = max(np.hypot(*(coords[j] - coords[i])) for i, j in connectivity)
    else:
        Lc = 1.0
    Um = float(np.max(np.sqrt(ux**2 + uy**2))) if U.size else 0.0
    if scale is None:
        scale = 0.15 * Lc / (Um if Um > 0 else 1.0)

    # deformed node coordinates (translations only)
    def_xy = np.column_stack([coords[:, 0] + scale*ux,
                              coords[:, 1] + scale*uy])

    fig, ax = plt.subplots(figsize=(7, 6))

    # undeformed
    for i, j in connectivity:
        ax.plot([coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color='0.7', lw=1.5)
    ax.scatter(coords[:, 0], coords[:, 1], s=30, color='0.35', label='undeformed nodes')

    # deformed
    for i, j in connectivity:
        ax.plot([def_xy[i, 0], def_xy[j, 0]],
                [def_xy[i, 1], def_xy[j, 1]],
                'r-', lw=2.0)
    ax.scatter(def_xy[:, 0], def_xy[:, 1], s=28, facecolors='none', edgecolors='r', label='deformed nodes')

    # simple support markers (optional)
    if supports is not None:
        for n, (fx, fy, th) in enumerate(supports):
            x, y = coords[n]
            if fx == 1 and fy == 1 and th == 1:
                ax.plot(x, y, marker='s', ms=8, mfc='none', mec='g', mew=2, label='fixed' if n == 0 else "")
            elif fx == 1 and fy == 1 and th == 0:
                ax.plot(x, y, marker='^', ms=7, mfc='none', mec='b', mew=2, label='pin' if n == 0 else "")

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc='best')
    plt.show()
# ======================================================


# ===================== RUN ============================
 
    KG, F, node_dof, cache = assemble(coordinates, connectivity, E, A, I, applied_loads)
    U, Reac = apply_bc_and_solve(KG, F, supports, node_dof)

    np.set_printoptions(precision=6, suppress=True)
    print("Displacements per node [ux, uy, th]:")
    for n in range(len(coordinates)):
        ux, uy, th = U[3*n:3*n+3]
        print(f"  Node {n}: ux={ux:.6e}, uy={uy:.6e}, th={th:.6e}")

    print("\nReactions per node [Fx, Fy, M]:")
    for n in range(len(coordinates)):
        Fx, Fy, M = Reac[3*n:3*n+3]
        print(f"  Node {n}: Fx={Fx:.6e}, Fy={Fy:.6e}, M={M:.6e}")

    # Nodes-only plot (straight members, deformed at nodes)
    plot_nodes_only(coordinates, connectivity, U, supports=supports, scale=None,
                    title="Frame: nodes-only (undeformed vs deformed)")
