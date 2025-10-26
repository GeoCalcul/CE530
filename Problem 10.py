import numpy as np

# ------------------------------
# Problem data (from the slides)
# ------------------------------
E   = 30e6        # psi
A   = 1.0         # in^2
L   = 100.0       # in
rho = 0.00073     # lb*s^2/in^4
dt  = 0.25e-3     # s
t_end = 0.50e-3   # s  (we'll match the example up to 0.5 ms)

# Two equal-length bar elements -> 3 nodes (1--2--3)
# Global stiffness for 2-element uniform bar (standard 1D linear FE):
K3 = (E*A/L) * np.array([[ 1, -1,  0],
                         [-1,  2, -1],
                         [ 0, -1,  1]])   # lb/in

# Consistent mass (lumped in pattern given in slides): (rho*A*L/2)*diag([1,2,1])
M3 = (rho*A*L/2.0) * np.diag([1.0, 2.0, 1.0])   # lb*s^2/in

# Apply essential BC: u1 = 0  -> reduce to DOFs [2,3]
keep = [1, 2]  # zero-based indices for nodes 2 and 3
K = K3[np.ix_(keep, keep)]
M = M3[np.ix_(keep, keep)]

# Force vector: load at node 3 only, constant 1000 lb (matches slides)
def force_at_time(t):
    F3 = 1000.0
    return np.array([0.0, F3])

# ------------------------------
# Central difference scheme
#   d_{i+1} = M^{-1}[ dt^2*F_i + (2M - dt^2*K) d_i - M d_{i-1} ]
# ------------------------------
t_grid = np.arange(0.0, t_end + dt, dt)
n = len(t_grid)

d = np.zeros((n, 2))   # displacements [u2, u3]
v = np.zeros((n, 2))   # velocities
a = np.zeros((n, 2))   # accelerations

# Initial conditions (at rest)
d[0] = np.array([0.0, 0.0])
v[0] = np.array([0.0, 0.0])

# Initial acceleration a0 = M^{-1}(F0 - K d0)
F0 = force_at_time(t_grid[0])
a[0] = np.linalg.solve(M, F0 - K @ d[0])

# "Ghost" step d_{-1} (from slides)
d_minus1 = d[0] - dt*v[0] + 0.5*dt**2*a[0]

Minv = np.linalg.inv(M)

# March in time
for i in range(n-1):
    Fi = force_at_time(t_grid[i])
    rhs = (dt**2)*Fi + (2*M - (dt**2)*K) @ d[i] - M @ d_minus1
    d[i+1] = Minv @ rhs

    # New acceleration and velocity (for reference/output)
    a[i+1] = np.linalg.solve(M, force_at_time(t_grid[i+1]) - K @ d[i+1])
    v[i+1] = (d[i+1] - d_minus1) / (2*dt)

    # shift
    d_minus1 = d[i].copy()

# ------------------------------
# Print table like the slide
# ------------------------------
print("t (ms) | u2 (in)      u3 (in)      a2 (in/s^2)   a3 (in/s^2)   v2 (in/s)    v3 (in/s)")
for i, t in enumerate(t_grid):
    print(f"{1e3*t:6.2f} | {d[i,0]:10.4e} {d[i,1]:10.4e} "
          f"{a[i,0]:12.1f} {a[i,1]:12.1f} {v[i,0]:11.2f} {v[i,1]:11.2f}")

# Quick checks against the worked example:
# Expect around:
#  t=0.25 ms: u2 ~ 0,        u3 ~ 0.858e-3 in
#  t=0.50 ms: u2 ~ 0.221e-3, u3 ~ 2.99e-3  in
