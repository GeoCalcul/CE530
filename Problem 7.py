# Solve the user's 1D heat conduction example with the FEM code from above (no Jacobian/Gauss in ke).
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters from the slide:
k = 6.0          # W/(m·°C)
A = 0.1          # m^2
L = 0.4          # m
q_flux = 5000.0  # W/m^2 (positive into the rod at the right end)

# Mesh: nodes at 0, L/4, L/2, 3L/4, L
x = np.linspace(0.0, L, 5)
coordinates = np.column_stack([x, np.zeros_like(x)])
connectivity = np.column_stack([np.arange(4), np.arange(1,5)])

# Essential BC
T_prescribed = {0: 100.0}  # °C at x=0

# Natural flux at right end node (total W into domain)
Q_flux_nodes = {4: q_flux * A}  # F_N = q * A

# No convection in this example (title says convection but problem text shows flux)
convection_nodes = {}

# FEM helpers
def shape_functions(x_local, Le):
    N = np.array([[1.0 - x_local/Le, x_local/Le]])
    B = np.array([[-1.0/Le, 1.0/Le]])
    return N, B

def element_conduction(k_e, A_e, Le):
    _, B = shape_functions(Le/2, Le)
    ke = (B.T @ (k_e * B)) * A_e * Le
    return ke

def element_source_load(q_e, A_e, Le):
    if q_e == 0.0:
        return np.zeros((2,1))
    return (q_e * A_e * Le / 2.0) * np.array([[1.0],[1.0]])

def assemble_global(K, F, x, conn, kval, Aval, qval):
    ne = conn.shape[0]
    k_e = np.full(ne, kval) if np.isscalar(kval) else np.asarray(kval, float)
    A_e = np.full(ne, Aval) if np.isscalar(Aval) else np.asarray(Aval, float)
    q_e = np.full(ne, qval) if np.isscalar(qval) else np.asarray(qval, float)
    for e, (i, j) in enumerate(conn):
        Le = x[j] - x[i]
        ke = element_conduction(k_e[e], A_e[e], Le)
        fe = element_source_load(q_e[e], A_e[e], Le)
        K[np.ix_([i,j],[i,j])] += ke
        # RHS assembly
        F[np.ix_([i,j],[0])] += fe
    return K, F

def apply_natural_flux(F, flux_nodes):
    for node, qn in flux_nodes.items():
        F[node,0] += qn
    return F

def apply_convection(K, F, conv_nodes):
    for node, (h, Tinf, Aend) in conv_nodes.items():
        K[node, node] += h * Aend
        F[node, 0]    += h * Aend * Tinf
    return K, F

def apply_essential(K, F, T_dict):
    n = K.shape[0]
    free = np.ones(n, dtype=bool)
    T_known = np.zeros((n,1))
    for node, val in T_dict.items():
        free[node] = False
        T_known[node,0] = val
    if np.any(~free):
        F[free] -= K[np.ix_(free, ~free)] @ T_known[~free]
    Kff = K[np.ix_(free, free)]
    Ff  = F[free]
    return Kff, Ff, free, T_known

def heat_flux_per_element(kval, x, conn, T):
    ne = conn.shape[0]
    qcond = np.zeros(ne)
    for e, (i, j) in enumerate(conn):
        Le = x[j] - x[i]
        _, B = shape_functions(Le/2, Le)
        Te = np.array([[T[i,0]],[T[j,0]]])
        dTdx = float(B @ Te)
        ke = kval if np.isscalar(kval) else kval[e]
        qcond[e] = - ke * dTdx
    return qcond

# Assemble and solve
nn = len(x)
K = np.zeros((nn, nn))
F = np.zeros((nn, 1))
K, F = assemble_global(K, F, x, connectivity, k, A, 0.0)  # no volumetric source
F = apply_natural_flux(F, Q_flux_nodes)
K, F = apply_convection(K, F, convection_nodes)

Kff, Ff, free, T_known = apply_essential(K.copy(), F.copy(), T_prescribed)
T = np.zeros((nn,1))
T[free] = np.linalg.solve(Kff, Ff)
T[~free] = T_known[~free]

q_e = heat_flux_per_element(k, x, connectivity, T)

# Print results
print("Node x (m):", x)
print("Temperatures (°C):", T.flatten())
print("Element conductive heat flux q_e (W/m):", q_e)

# Plot
plt.figure()
plt.plot(x, T.flatten(), '-o')
plt.xlabel("x (m)")
plt.ylabel("Temperature (°C)")
plt.title("1D Heat Conduction with Right-End Heat Flux")
plt.grid(True)
plt.show()
