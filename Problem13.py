import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Material and loading data
# -------------------------------------------------
E = 210e9        # Young's modulus [Pa]
nu = 0.3         # Poisson's ratio [-]
t = 0.1          # thickness [m]

alpha = 1.2e-5   # thermal expansion coefficient [1/K]
dT = 50.0        # uniform temperature rise [K]

p = 1.0e6        # traction on right edge [Pa] (in x-direction)

# Plane stress elasticity matrix D
D = E / (1.0 - nu**2) * np.array([
    [1.0,  nu,   0.0],
    [nu,  1.0,   0.0],
    [0.0, 0.0, (1.0 - nu) / 2.0]
])

# -------------------------------------------------
# 2. Mesh: 2x2 quads, 9 nodes
# -------------------------------------------------
# Node numbering:
# (0,2) -7----8----9- (2,2)
#        | e3 | e4 |
# (0,1) -4----5----6- (2,1)
#        | e1 | e2 |
# (0,0) -1----2----3- (2,0)

coords = np.array([
    [0.0, 0.0],  # node 1 -> index 0
    [1.0, 0.0],  # node 2 -> index 1
    [2.0, 0.0],  # node 3 -> index 2
    [0.0, 1.0],  # node 4 -> index 3
    [1.0, 1.0],  # node 5 -> index 4
    [2.0, 1.0],  # node 6 -> index 5
    [0.0, 2.0],  # node 7 -> index 6
    [1.0, 2.0],  # node 8 -> index 7
    [2.0, 2.0]   # node 9 -> index 8
], dtype=float)

# Elements (4-node quads), 0-based node indices
elements = [
    [0, 1, 4, 3],  # e1: (1,2,5,4)
    [1, 2, 5, 4],  # e2: (2,3,6,5)
    [3, 4, 7, 6],  # e3: (4,5,8,7)
    [4, 5, 8, 7]   # e4: (5,6,9,8)
]

# -------------------------------------------------
# 3. Shape functions and derivatives
# -------------------------------------------------
def quad4_shape_funcs(xi, eta):
    """4-node bilinear shape functions N_i(xi,eta)."""
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return np.array([N1, N2, N3, N4])

def quad4_dNdXi(xi, eta):
    """Derivatives of N_i w.r.t xi, eta."""
    dN_dxi = np.array([
        -0.25 * (1.0 - eta),
         0.25 * (1.0 - eta),
         0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
         0.25 * (1.0 + xi),
         0.25 * (1.0 - xi)
    ])
    return dN_dxi, dN_deta

# -------------------------------------------------
# 4. Element stiffness and thermal load
# -------------------------------------------------
def element_matrices(node_coords, D, t, alpha, dT):
    """
    Compute element stiffness Ke and thermal equivalent nodal forces f_th
    for a 4-node quad in plane stress with uniform dT.
    node_coords: (4,2) array of [x,y] for local nodes [1..4]
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gp, -gp), ( gp, -gp), ( gp,  gp), (-gp,  gp)]

    Ke = np.zeros((8, 8))
    fth = np.zeros(8)
    eps_th = np.array([alpha * dT, alpha * dT, 0.0])  # plane stress

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = quad4_dNdXi(xi, eta)

        # Jacobian
        J = np.zeros((2, 2))
        J[0, 0] = np.sum(dN_dxi * node_coords[:, 0])
        J[0, 1] = np.sum(dN_deta * node_coords[:, 0])
        J[1, 0] = np.sum(dN_dxi * node_coords[:, 1])
        J[1, 1] = np.sum(dN_deta * node_coords[:, 1])

        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # Gradients in x,y
        grad = np.zeros((4, 2))  # [dN/dx, dN/dy]
        for i in range(4):
            grad[i, :] = invJ @ np.array([dN_dxi[i], dN_deta[i]])

        # B matrix (3x8)
        B = np.zeros((3, 8))
        for i in range(4):
            dNdx, dNdy = grad[i, 0], grad[i, 1]
            B[0, 2*i    ] = dNdx
            B[1, 2*i + 1] = dNdy
            B[2, 2*i    ] = dNdy
            B[2, 2*i + 1] = dNdx

        Ke  +=  B.T @ D @ B * detJ * t
        fth += (B.T @ (D @ eps_th)) * detJ * t

    return Ke, fth

# -------------------------------------------------
# 5. Edge traction load for right boundary (local edge 2-3)
# -------------------------------------------------
def edge_traction_load(node_coords, p, t):
    """
    Equivalent nodal forces on the edge between local nodes 2 and 3
    (xi = +1, eta in [-1,1]) due to traction [p,0] in x-direction.
    """
    gp1d = 1.0 / np.sqrt(3.0)
    etas = [-gp1d, gp1d]
    weights = [1.0, 1.0]

    f_edge = np.zeros(8)
    for eta, w in zip(etas, weights):
        xi = 1.0
        N = quad4_shape_funcs(xi, eta)
        dN_dxi, dN_deta = quad4_dNdXi(xi, eta)

        # Tangent for edge (derivative wrt eta)
        dx_deta = np.sum(dN_deta * node_coords[:, 0])
        dy_deta = np.sum(dN_deta * node_coords[:, 1])
        J_edge = np.sqrt(dx_deta**2 + dy_deta**2)  # |ds/deta|

        # Only local nodes 2 and 3 on the right edge
        N2 = N[1]
        N3 = N[2]

        # Shape matrix (2x8) for edge DOFs [ux1,uy1,...,ux4,uy4]
        Nmat = np.zeros((2, 8))
        Nmat[0, 2*1] = N2   # ux at local node 2
        Nmat[0, 2*2] = N3   # ux at local node 3

        traction = np.array([p, 0.0])  # [tx, ty]
        f_edge += (Nmat.T @ traction) * t * J_edge * w

    return f_edge

# -------------------------------------------------
# 6. Global assembly
# -------------------------------------------------
nnodes = coords.shape[0]
ndof   = nnodes * 2

K      = np.zeros((ndof, ndof))
f_th   = np.zeros(ndof)
f_mech = np.zeros(ndof)

# Assemble element matrices
for conn in elements:
    xe = coords[conn, :]  # (4,2)
    Ke, fthe = element_matrices(xe, D, t, alpha, dT)

    # DOF indices for this element: [ux1,uy1,ux2,uy2,ux3,uy3,ux4,uy4]
    edofs = []
    for n in conn:
        edofs.extend([2*n, 2*n + 1])
    edofs = np.array(edofs, dtype=int)

    f_th[edofs] += fthe
    for a in range(8):
        A = edofs[a]
        for b in range(8):
            B = edofs[b]
            K[A, B] += Ke[a, b]

# Edge traction on right boundary: elements e2 and e4 (indices 1 and 3)
for e_idx in [1, 3]:
    conn = elements[e_idx]
    xe = coords[conn, :]
    f_edge = edge_traction_load(xe, p, t)

    edofs = []
    for n in conn:
        edofs.extend([2*n, 2*n + 1])
    edofs = np.array(edofs, dtype=int)

    f_mech[edofs] += f_edge

# -------------------------------------------------
# 7. Apply boundary conditions
# -------------------------------------------------
# Fix nodes 1,4,7 (indices 0,3,6) fully: ux=uy=0
fixed_nodes = [0, 3, 6]
fixed_dofs = []
for n in fixed_nodes:
    fixed_dofs.extend([2*n, 2*n + 1])
fixed_dofs = np.array(fixed_dofs, dtype=int)

all_dofs  = np.arange(ndof, dtype=int)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

# Reduced system
Kff = K[np.ix_(free_dofs, free_dofs)]
rhs = (f_mech - f_th)[free_dofs]

# Solve
u = np.zeros(ndof)
u_free = np.linalg.solve(Kff, rhs)
u[free_dofs] = u_free

# -------------------------------------------------
# 8. Post-processing: nodal displacements
# -------------------------------------------------
u_mm = u * 1000.0  # convert to mm

print("Nodal displacements (in mm):")
for i in range(nnodes):
    ux = u_mm[2*i]
    uy = u_mm[2*i + 1]
    print(f" Node {i+1:2d}: ux = {ux: .6f} mm,  uy = {uy: .6f} mm")

# Displacement magnitude (for coloring)
u_mag = np.sqrt(u_mm[0::2]**2 + u_mm[1::2]**2)

# -------------------------------------------------
# 9. Visualization (fixed)
# -------------------------------------------------
# Deformed coordinates (scale factor)
scale = 50.0  # try 10–100 to adjust visually

coords_def = coords.copy()
for i in range(nnodes):
    coords_def[i, 0] += scale * u[2*i]     # ux
    coords_def[i, 1] += scale * u[2*i + 1] # uy

fig, ax = plt.subplots(figsize=(7, 7))

# Plot undeformed and deformed mesh
for e_idx, conn in enumerate(elements):
    # undeformed
    xs  = np.append(coords[conn, 0], coords[conn[0], 0])
    ys  = np.append(coords[conn, 1], coords[conn[0], 1])
    ax.plot(xs, ys, 'k--', linewidth=0.8,
            label='undeformed' if e_idx == 0 else "")
    # deformed
    xsd = np.append(coords_def[conn, 0], coords_def[conn[0], 0])
    ysd = np.append(coords_def[conn, 1], coords_def[conn[0], 1])
    ax.plot(xsd, ysd, 'b-', linewidth=1.5,
            label='deformed (scaled)' if e_idx == 0 else "")

# Scatter nodes colored by displacement magnitude
sc = ax.scatter(coords_def[:, 0], coords_def[:, 1],
                c=u_mag, cmap='viridis', s=80, edgecolors='k')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Displacement magnitude [mm]')

# Plot nodal numbers
for i, (x, y) in enumerate(coords_def):
    ax.text(x + 0.02, y + 0.02, str(i+1), fontsize=9, color='red')

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Thermo-mechanical 2×2 quad mesh\nundeformed (dashed) and scaled deformed (solid)')
ax.grid(True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
