import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Material and loading data
# -------------------------------------------------
E = 210e9        # Young's modulus [Pa]
nu = 0.3         # Poisson's ratio [-]
t = 0.1          # thickness [m]
rho = 7850.0     # density [kg/m^3] (mechanical)

alpha = 1.2e-5   # thermal expansion coefficient [1/K]

# Thermal properties
rho_th = 7850.0  # density [kg/m^3] for thermal capacity
c_th   = 500.0   # specific heat [J/(kg K)]
kappa  = 50.0    # thermal conductivity [W/(m K)]

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
# 4. Element mechanical matrices
# -------------------------------------------------
def element_mech_matrices(node_coords, D, t, rho):
    """
    Compute element mechanical stiffness Ke and consistent mass Me
    for a 4-node quad in plane stress.
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gp, -gp), ( gp, -gp), ( gp,  gp), (-gp,  gp)]

    Ke = np.zeros((8, 8))
    Me = np.zeros((8, 8))

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = quad4_dNdXi(xi, eta)
        N = quad4_shape_funcs(xi, eta)

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

        # Stiffness
        Ke += B.T @ D @ B * detJ * t

        # Consistent mass: Me_ij = ∫ rho t N_i N_j dΩ * I2
        for i in range(4):
            for j in range(4):
                m_ij = rho * t * N[i] * N[j] * detJ
                Me[2*i,   2*j  ] += m_ij
                Me[2*i+1, 2*j+1] += m_ij

    return Ke, Me

# -------------------------------------------------
# 5. Element thermal matrices
# -------------------------------------------------
def element_thermal_matrices(node_coords, rho_th, c_th, kappa, t):
    """
    Compute element thermal capacity Ce and conductivity Ke_th for
    a 4-node quad in 2D heat conduction (per unit thickness).
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gp, -gp), ( gp, -gp), ( gp,  gp), (-gp,  gp)]

    Ce = np.zeros((4, 4))
    Ke_th = np.zeros((4, 4))

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = quad4_dNdXi(xi, eta)
        N = quad4_shape_funcs(xi, eta)

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

        # Capacity and conductivity
        for i in range(4):
            for j in range(4):
                Ce[i, j]    += rho_th * c_th * N[i] * N[j] * detJ * t
                Ke_th[i, j] += kappa * np.dot(grad[i, :], grad[j, :]) * detJ * t

    return Ce, Ke_th

# -------------------------------------------------
# 6. Element thermal load from current temperature
# -------------------------------------------------
def element_thermal_load_from_T(node_coords, D, alpha, t, T_e):
    """
    Compute element thermal equivalent nodal forces f_th (size 8)
    for a 4-node quad given nodal temperatures T_e (size 4).
    Uses interpolated T at Gauss points.
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = [(-gp, -gp), ( gp, -gp), ( gp,  gp), (-gp,  gp)]

    fth = np.zeros(8)

    for xi, eta in gauss_points:
        dN_dxi, dN_deta = quad4_dNdXi(xi, eta)
        N = quad4_shape_funcs(xi, eta)

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

        # Temperature at Gauss point
        T_gp = np.dot(N, T_e)
        eps_th = np.array([alpha * T_gp, alpha * T_gp, 0.0])

        # B matrix (3x8)
        B = np.zeros((3, 8))
        for i in range(4):
            dNdx, dNdy = grad[i, 0], grad[i, 1]
            B[0, 2*i    ] = dNdx
            B[1, 2*i + 1] = dNdy
            B[2, 2*i    ] = dNdy
            B[2, 2*i + 1] = dNdx

        fth += (B.T @ (D @ eps_th)) * detJ * t

    return fth

# -------------------------------------------------
# 7. Edge traction load for right boundary (local edge 2-3)
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
# 8. Global assembly (mechanical and thermal)
# -------------------------------------------------
nnodes = coords.shape[0]
ndof_mech = nnodes * 2  # ux, uy at each node

K_mech = np.zeros((ndof_mech, ndof_mech))
M_mech = np.zeros((ndof_mech, ndof_mech))
f_mech = np.zeros(ndof_mech)

# Thermal matrices (one DOF per node)
M_th = np.zeros((nnodes, nnodes))
K_th = np.zeros((nnodes, nnodes))
F_th_source = np.zeros(nnodes)  # no internal source

# Assemble mechanical and thermal
for conn in elements:
    xe = coords[conn, :]  # (4,2)

    # Mechanical
    Ke, Me = element_mech_matrices(xe, D, t, rho)
    edofs_mech = []
    for n in conn:
        edofs_mech.extend([2*n, 2*n + 1])
    edofs_mech = np.array(edofs_mech, dtype=int)

    for a in range(8):
        A = edofs_mech[a]
        for b in range(8):
            B = edofs_mech[b]
            K_mech[A, B] += Ke[a, b]
            M_mech[A, B] += Me[a, b]

    # Thermal
    Ce, Ke_th_e = element_thermal_matrices(xe, rho_th, c_th, kappa, t)
    edofs_th = np.array(conn, dtype=int)
    for a in range(4):
        A = edofs_th[a]
        for b in range(4):
            B = edofs_th[b]
            M_th[A, B] += Ce[a, b]
            K_th[A, B] += Ke_th_e[a, b]

# Mechanical edge traction on right boundary: elements e2 and e4 (indices 1 and 3)
for e_idx in [1, 3]:
    conn = elements[e_idx]
    xe = coords[conn, :]
    f_edge = edge_traction_load(xe, p, t)

    edofs_mech = []
    for n in conn:
        edofs_mech.extend([2*n, 2*n + 1])
    edofs_mech = np.array(edofs_mech, dtype=int)

    f_mech[edofs_mech] += f_edge

# -------------------------------------------------
# 9. Boundary conditions (thermal and mechanical)
# -------------------------------------------------
# Mechanical: fix nodes 1,4,7 fully: ux=uy=0
fixed_nodes_mech = [0, 3, 6]
fixed_dofs_mech = []
for n in fixed_nodes_mech:
    fixed_dofs_mech.extend([2*n, 2*n + 1])
fixed_dofs_mech = np.array(fixed_dofs_mech, dtype=int)

all_dofs_mech  = np.arange(ndof_mech, dtype=int)
free_dofs_mech = np.setdiff1d(all_dofs_mech, fixed_dofs_mech)

Kff_mech = K_mech[np.ix_(free_dofs_mech, free_dofs_mech)]
Mff_mech = M_mech[np.ix_(free_dofs_mech, free_dofs_mech)]
f_mech_f = f_mech[free_dofs_mech]

# Thermal: Dirichlet BCs on left and right edges
left_nodes_th  = [0, 3, 6]   # x=0
right_nodes_th = [2, 5, 8]   # x=2
fixed_nodes_th = np.array(left_nodes_th + right_nodes_th, dtype=int)
all_nodes_th   = np.arange(nnodes, dtype=int)
free_nodes_th  = np.setdiff1d(all_nodes_th, fixed_nodes_th)

# -------------------------------------------------
# 10. Temperature time history at right boundary
# -------------------------------------------------
dt = 50.0  # global time step [s] (matches temperature history step)

# Example time history: replace this array with your real data
T_right_values = np.array([0.0, 10.0, 25.0, 40.0, 50.0])  # [K]
n_hist = len(T_right_values)

T_end = dt * (n_hist - 1)
nsteps = n_hist - 1
time = np.linspace(0.0, T_end, nsteps+1)

def T_right_at_step(n):
    """
    Temperature at right boundary at time step n.
    time[n] = n * dt, T_right = T_right_values[n].
    """
    return T_right_values[n]

# -------------------------------------------------
# 11. Time integration parameters (mechanical)
# -------------------------------------------------
beta = 0.25
gamma = 0.5

ndof_fm = len(free_dofs_mech)

# Thermal initial condition
T = np.zeros(nnodes)
T_hist_node5 = np.zeros(nsteps+1)
T_hist_node5[0] = T[4]  # node 5 index = 4

# Mechanical initial conditions
u_free = np.zeros((nsteps+1, ndof_fm))
v_free = np.zeros((nsteps+1, ndof_fm))
a_free = np.zeros((nsteps+1, ndof_fm))

# Initial mechanical acceleration (T=0 => no thermal load)
F_th_initial = np.zeros(ndof_fm)
F0_mech = f_mech_f + F_th_initial
a_free[0, :] = np.linalg.solve(Mff_mech, F0_mech - Kff_mech @ u_free[0, :])

# Effective mechanical stiffness (no damping)
K_eff_mech = Kff_mech + Mff_mech * (1.0 / (beta * dt**2))

# Thermal system matrix for backward Euler
A_T = M_th / dt + K_th

# For plotting mech response: node 9, ux
node9 = 8
dof_x_node9 = 2 * node9
idx_free_x = np.where(free_dofs_mech == dof_x_node9)[0][0]
ux_hist_node9 = np.zeros(nsteps+1)
ux_hist_node9[0] = u_free[0, idx_free_x]

# -------------------------------------------------
# 12. Time-stepping loop (staggered thermo-mechanical)
# -------------------------------------------------
for n in range(nsteps):
    t_n1 = time[n+1]

    # ---------------- THERMAL STEP: backward Euler ----------------
    # Apply Dirichlet BCs on T at t_{n+1}
    T_fixed = np.zeros_like(T)
    # Left edge at 0
    for nid in left_nodes_th:
        T_fixed[nid] = 0.0
    # Right edge at prescribed history value
    T_R = T_right_at_step(n+1)
    for nid in right_nodes_th:
        T_fixed[nid] = T_R

    # Full RHS for BE: M_th * T^n / dt + F_source (0 here)
    rhs_T_full = M_th @ (T / dt)

    # Partition system: A_T * T^{n+1} = rhs_T_full
    A_ff = A_T[np.ix_(free_nodes_th, free_nodes_th)]
    A_fF = A_T[np.ix_(free_nodes_th, fixed_nodes_th)]

    rhs_T_free = rhs_T_full[free_nodes_th] - A_fF @ T_fixed[fixed_nodes_th]
    T_new_free = np.linalg.solve(A_ff, rhs_T_free)

    # Assemble T^{n+1}
    T_new = T_fixed.copy()
    T_new[free_nodes_th] = T_new_free

    T = T_new.copy()
    T_hist_node5[n+1] = T[4]  # node 5

    # ---------------- MECHANICAL STEP: Newmark-β ----------------
    # Build thermal mechanical load from current T
    f_th = np.zeros(ndof_mech)
    for conn in elements:
        xe = coords[conn, :]
        T_e = T[conn]  # nodal temps for element
        fth_e = element_thermal_load_from_T(xe, D, alpha, t, T_e)

        edofs_mech = []
        for nd in conn:
            edofs_mech.extend([2*nd, 2*nd + 1])
        edofs_mech = np.array(edofs_mech, dtype=int)

        f_th[edofs_mech] += fth_e

    f_th_f = f_th[free_dofs_mech]
    Fnp1_mech = f_mech_f + f_th_f

    # Newmark effective RHS
    rhs_eff_mech = (
        Fnp1_mech
        + Mff_mech @ (
            (1.0 / (beta * dt**2)) * u_free[n, :]
            + (1.0 / (beta * dt))    * v_free[n, :]
            + (1.0 / (2.0*beta) - 1.0) * a_free[n, :]
        )
    )

    # Solve for u_{n+1}
    u_free[n+1, :] = np.linalg.solve(K_eff_mech, rhs_eff_mech)

    # Update a_{n+1}
    a_free[n+1, :] = (
        1.0 / (beta * dt**2) *
        (u_free[n+1, :] - u_free[n, :] - dt * v_free[n, :]
         - dt**2 * 0.5 * (1.0 - 2.0*beta) * a_free[n, :])
    )

    # Update v_{n+1}
    v_free[n+1, :] = (
        v_free[n, :]
        + dt * ((1.0 - gamma) * a_free[n, :] + gamma * a_free[n+1, :])
    )

    ux_hist_node9[n+1] = u_free[n+1, idx_free_x]

# -------------------------------------------------
# 13. Post-processing
# -------------------------------------------------
# Temperature history at node 5
plt.figure()
plt.plot(time, T_hist_node5)
plt.xlabel("Time [s]")
plt.ylabel("Temperature at node 5 [K]")
plt.title("Transient temperature T(t) at node 5")
plt.grid(True)
plt.tight_layout()
plt.show()

# Mechanical response at node 9 (ux)
plt.figure()
plt.plot(time, ux_hist_node9 * 1000.0)
plt.xlabel("Time [s]")
plt.ylabel("u_x at node 9 [mm]")
plt.title("Thermo-mechanical dynamic response u_x(t) at node 9")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final deformed shape
u_final = np.zeros(ndof_mech)
u_final[free_dofs_mech] = u_free[-1, :]
u_mm = u_final * 1000.0

print("Nodal displacements at final time (in mm):")
for i in range(nnodes):
    ux = u_mm[2*i]
    uy = u_mm[2*i + 1]
    print(f" Node {i+1:2d}: ux = {ux: .6f} mm,  uy = {uy: .6f} mm")

# Deformed coordinates (scale factor)
scale = 50.0
coords_def = coords.copy()
for i in range(nnodes):
    coords_def[i, 0] += scale * u_final[2*i]
    coords_def[i, 1] += scale * u_final[2*i + 1]

fig, ax = plt.subplots(figsize=(7, 7))
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

u_mag = np.sqrt(u_final[0::2]**2 + u_final[1::2]**2) * 1000.0
sc = ax.scatter(coords_def[:, 0], coords_def[:, 1],
                c=u_mag, cmap='viridis', s=80, edgecolors='k')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Displacement magnitude [mm] (final time)')

for i, (x, y) in enumerate(coords_def):
    ax.text(x + 0.02, y + 0.02, str(i+1), fontsize=9, color='red')

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Coupled transient thermo–mechanical 2×2 quad mesh\nundeformed (dashed) and final deformed (solid)')
ax.grid(True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
