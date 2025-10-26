# Horizontal 1D spring–mass animation (moves both directions)
# Motion computed by the central difference method (same system parameters).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# Force function (triangular ramp)
def force_ramp(t, F0=2000.0, t_end=0.2):
    if t < 0:
        return 0.0
    if t <= t_end:
        return F0 * (1 - t / t_end)
    return 0.0

# Central difference solver
def central_difference(m, k, F_func, dt, t_end, d0=0.0, v0=0.0):
    n_steps = int(np.round(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, n_steps)
    d = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)

    d[0] = d0
    v[0] = v0
    a[0] = (F_func(0.0) - k * d[0]) / m
    d_minus1 = d[0] - dt * v[0] + 0.5 * (dt**2) * a[0]

    for i in range(0, n_steps - 1):
        Fi = F_func(t[i])
        d_ip1 = ((dt**2) * Fi + (2*m - (dt**2)*k) * d[i] - m * (d_minus1 if i == 0 else d[i-1])) / m
        if i == 0:
            d[1] = d_ip1
        else:
            d[i+1] = d_ip1

    a = (np.array([F_func(ti) for ti in t]) - k*d) / m
    v[1:-1] = (d[2:] - d[:-2]) / (2*dt)
    v[0] = (d[1] - d_minus1) / (2*dt)
    v[-1] = (d[-1] - d[-3]) / (2*dt) if len(d) >= 3 else v[0]
    return t, d, v, a

# Parameters
m = 31.83
k = 100.0
dt = 0.01  # smaller step for smooth motion
t_end = 1.0

# Compute motion
t, d, v, a = central_difference(m, k, force_ramp, dt, t_end, d0=0.0, v0=0.0)

# Center motion about equilibrium (oscillatory view)
d_centered = d - np.mean(d[:len(d)//4])  # roughly equilibrium before oscillation fully develops

# Build figure for animation
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("1D Spring–Mass System (Central Difference Simulation)")

# Elements
wall_x = [0, 0]
wall_y = [-0.5, 0.5]
spring_line, = ax.plot([], [], lw=2)
mass_rect = plt.Rectangle((0, -0.3), 0.6, 0.6, ec='k', fc='lightgray')
ax.add_patch(mass_rect)
ax.plot(wall_x, wall_y, color='k', lw=4)  # fixed wall

# Geometry constants
x0 = 0.5  # equilibrium spring length (left end fixed at 0)
L0 = 3.0  # reference position of mass center
scale = 1.0  # horizontal scale factor (for visible motion)

# Function to make a coiled spring line
def spring_shape(x_fixed, x_mass, n_coils=8, amplitude=0.1):
    xs = np.linspace(x_fixed, x_mass, 200)
    ys = amplitude * np.sin(2 * np.pi * n_coils * (xs - x_fixed) / (x_mass - x_fixed))
    return xs, ys

# Animation update
def init():
    xs, ys = spring_shape(0, L0)
    spring_line.set_data(xs, ys)
    mass_rect.set_x(L0 - 0.3)
    return spring_line, mass_rect

def update(frame):
    x_mass = L0 + scale * d_centered[frame]
    xs, ys = spring_shape(0, x_mass)
    spring_line.set_data(xs, ys)
    mass_rect.set_x(x_mass - 0.3)
    return spring_line, mass_rect

anim = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=40)

# Save as GIF
gif_path = Path(r"C:\Users\armc\spring_mass_oscillation.gif")
anim.save(gif_path, writer=PillowWriter(fps=25))
 
 

gif_path
