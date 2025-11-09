# Re-run with corrected CCW element ordering to avoid negative areas.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER INPUTS
# -----------------------------
nseg = 5
xs = np.linspace(0.0, 5.0, nseg + 1)
y_top, y_bot = 1.0, 0.0

coords_top = np.column_stack([xs, np.full_like(xs, y_top)])
coords_bot = np.column_stack([xs, np.full_like(xs, y_bot)])
coords = np.vstack([coords_top, coords_bot])  # (12,2)

# Connectivity with ensured CCW orientation
conn = []
for i in range(nseg):
    n0 = i            # top i
    n1 = i + 1        # top i+1
    n2 = 6 + i        # bot i
    n3 = 6 + i + 1    # bot i+1
    # CCW Tri A: (top_i, bot_{i+1}, top_{i+1})
    conn.append([n0, n3, n1])
    # CCW Tri B: (top_i, bot_i, bot_{i+1})
    conn.append([n0, n2, n3])
conn = np.array(conn, dtype=int)  # shape (10,3)

material = {
    "E": 30e9,         # Pa
    "nu": 0.2,         # -
    "thickness": 0.1,  # m
    "type": "plane_stress",
}

ndof_per_node = 2
ndof = coords.shape[0] * ndof_per_node
f = np.zeros(ndof)

# Example force: downward at rightmost top node (node 5)
right_top_node = 5
f[right_top_node * ndof_per_node + 1] = -1000.0

# Essential BCs: fix left edge nodes (0 and 6): u=v=0
bc = {}
for n in [0, 6]:
    bc[n * ndof_per_node + 0] = 0.0  # u
    bc[n * ndof_per_node + 1] = 0.0  # v


# -----------------------------
# FEM helpers
# -----------------------------
def constitutive_matrix(E, nu, mtype="plane_stress"):
    if mtype == "plane_stress":
        factor = E / (1.0 - nu**2)
        D = factor * np.array([
            [1.0,    nu,          0.0],
            [nu,     1.0,         0.0],
            [0.0,    0.0,  (1.0 - nu) / 2.0]
        ])
    elif mtype == "plane_strain":
        factor = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = factor * np.array([
            [1.0,            nu/(1.0 - nu),          0.0],
            [nu/(1.0 - nu),  1.0,                    0.0],
            [0.0,            0.0,        (1.0 - 2.0*nu)/(2.0*(1.0 - nu))]
        ])
    else:
        raise ValueError("mtype must be 'plane_stress' or 'plane_strain'")
    return D


def tri_geom_terms(xy):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    twoA = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    A = 0.5 * twoA
    b1 = y2 - y3; c1 = x3 - x2
    b2 = y3 - y1; c2 = x1 - x3
    b3 = y1 - y2; c3 = x2 - x1
    b = np.array([b1, b2, b3], dtype=float)
    c = np.array([c1, c2, c3], dtype=float)
    return A, b, c


def B_matrix_CST(A, b, c):
    denom = 2.0 * A
    return (1.0/denom) * np.array([
        [b[0], 0.0,   b[1], 0.0,   b[2], 0.0],
        [0.0,  c[0],  0.0,  c[1],  0.0,  c[2]],
        [c[0], b[0],  c[1], b[1],  c[2], b[2]],
    ], dtype=float)


def element_stiffness_CST(xy, E, nu, t, mtype="plane_stress"):
    A, b, c = tri_geom_terms(xy)
    if A <= 0.0:
        raise ValueError("Triangle area is non-positive. Check node order (use counter-clockwise).")
    B = B_matrix_CST(A, b, c)
    D = constitutive_matrix(E, nu, mtype)
    return t * A * (B.T @ D @ B)


def build_dof_map_for_element(elem_nodes, ndof_per_node):
    dofs = []
    for n in elem_nodes:
        base = n * ndof_per_node
        dofs.extend([base + 0, base + 1])
    return np.array(dofs, dtype=int)


def assemble_global_stiffness(coords, conn, material, ndof_per_node=2):
    nnode = coords.shape[0]
    ndof = nnode * ndof_per_node
    K = np.zeros((ndof, ndof), dtype=float)

    E = material["E"]
    nu = material["nu"]
    t  = material["thickness"]
    mtype = material.get("type", "plane_stress")

    for elem in conn:
        xy = coords[elem, :]
        Ke = element_stiffness_CST(xy, E, nu, t, mtype)
        edofs = build_dof_map_for_element(elem, ndof_per_node)
        for a in range(len(edofs)):
            A = edofs[a]
            for b in range(len(edofs)):
                B = edofs[b]
                K[A, B] += Ke[a, b]
    return K


def apply_dirichlet_bc(K, f, bc):
    K_mod = K.copy()
    f_mod = f.copy()
    for dof, val in bc.items():
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = 1.0
        f_mod[dof] = val
    return K_mod, f_mod


# Assemble and solve
K = assemble_global_stiffness(coords, conn, material, ndof_per_node=ndof_per_node)
Kb, fb = apply_dirichlet_bc(K, f, bc)
U = np.linalg.solve(Kb, fb)

# Plot
triangles = conn.tolist()
U_xy = U.reshape((-1, 2))
Lref = np.max(np.linalg.norm(coords - coords.mean(axis=0), axis=1))
umax = np.max(np.linalg.norm(U_xy, axis=1))
scale = 0.0 if umax == 0 else 0.2 * Lref / umax
coords_def = coords + scale * U_xy

fig, ax = plt.subplots(figsize=(7, 3.2))

for tri in triangles:
    poly = np.vstack([coords[tri], coords[tri[0]]])
    ax.plot(poly[:,0], poly[:,1])

for tri in triangles:
    poly = np.vstack([coords_def[tri], coords_def[tri[0]]])
    ax.plot(poly[:,0], poly[:,1], linestyle='--')

ax.set_aspect('equal', 'box')
ax.set_title('Undeformed (solid) and Deformed (dashed) Mesh — 10 CST elements')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

plt.show()

# Print compact results
np.set_printoptions(precision=4, suppress=True)
print("Nodes:\n", coords)
print("\nConnectivity (triangles):\n", conn)
print("\nDisplacements U [u0 v0 u1 v1 ...]^T :\n", U)
