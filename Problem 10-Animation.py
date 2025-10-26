import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter

# ------------------------------
# Problem data (2 linear bar elements)
# ------------------------------
E   = 30e6        # psi
A   = 1.0         # in^2
L   = 100.0       # in
rho = 0.00073     # lb*s^2/in^4
dt  = 0.25e-3     # s
t_end = 2.0e-3    # s  (2 ms)

# Geometry: three nodes (1--2--3)
x_nodes = np.array([0.0, L/2, L])  # in

# Global stiffness (1D linear FE for 2 elements)
K3 = (E*A/L) * np.array([[ 1.0, -1.0,  0.0],
                         [-1.0,  2.0, -1.0],
                         [ 0.0, -1.0,  1.0]])

# Lumped mass (ρA L/2)*diag([1,2,1])
M3 = (rho*A*L/2.0) * np.diag([1.0, 2.0, 1.0])

# Apply BC: u1 = 0  (remove row/col 0)
keep = [1, 2]
K = K3[np.ix_(keep, keep)]
M = M3[np.ix_(keep, keep)]

# Constant load 1000 lb at node 3
def F(t): return np.array([0.0, 1000.0])

# Time discretization
t_grid = np.arange(0.0, t_end + dt, dt)
n = len(t_grid)

# Initialize response
d = np.zeros((n, 2))
v = np.zeros((n, 2))
a = np.zeros((n, 2))

# Initial conditions (at rest)
d[0] = v[0] = 0.0
a[0] = np.linalg.solve(M, F(0) - K @ d[0])

# Central difference integration
d_prev = d[0] - dt*v[0] + 0.5*dt**2*a[0]
Minv = np.linalg.inv(M)

for i in range(n-1):
    rhs = (dt**2)*F(t_grid[i]) + (2*M - dt**2*K) @ d[i] - M @ d_prev
    d[i+1] = Minv @ rhs
    a[i+1] = np.linalg.solve(M, F(t_grid[i+1]) - K @ d[i+1])
    v[i+1] = (d[i+1] - d_prev) / (2*dt)
    d_prev = d[i].copy()

# Add fixed node 1 displacement = 0
d_full = np.zeros((n, 3))
d_full[:, 1:] = d

# ------------------------------
# Animation (true axial deformation)
# ------------------------------
scale = 2000  # exaggerate deformation for visibility

fig, ax = plt.subplots(figsize=(7, 2))
ax.set_aspect('equal')
ax.set_xlim(-5, L + 10)
ax.set_ylim(-2, 2)

line_undeformed, = ax.plot(x_nodes, np.zeros_like(x_nodes),
                           'k--', lw=1.0, label='Undeformed')
line_deformed, = ax.plot([], [], 'r-', lw=3, label='Deformed')
nodes_scatter = ax.scatter([], [], color='red', s=40)
time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)

ax.set_xlabel("x (in)")
ax.get_yaxis().set_visible(False)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

def init():
    line_deformed.set_data([], [])
    nodes_scatter.set_offsets(np.c_[[], []])
    time_text.set_text('')
    return line_deformed, nodes_scatter, time_text

def update(frame):
    x_def = x_nodes + scale * d_full[frame]  # displace along x-axis
    y = np.zeros_like(x_nodes)
    line_deformed.set_data(x_def, y)
    nodes_scatter.set_offsets(np.c_[x_def, y])
    time_text.set_text(f"t = {1e3*t_grid[frame]:.3f} ms")
    return line_deformed, nodes_scatter, time_text

anim = FuncAnimation(fig, update, frames=n, init_func=init, blit=True)
gif_path = Path(r"C:\Users\armc\spring_mass_oscillation.gif")
anim.save(gif_path, writer=PillowWriter(fps=30))

gif_path
