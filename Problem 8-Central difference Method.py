# Central Difference Method for an SDOF spring–mass system
# Example matches the slide: m=31.83 lb*s^2/in, k=100 lb/in
# Force: triangular ramp 2000 lb at t=0 to 0 lb at t=0.2 s (then 0 afterwards)
# Δt = 0.05 s

import numpy as np
import matplotlib.pyplot as plt

def force_ramp(t, F0=2000.0, t_end=0.2):
    """Linear ramp from F0 at t=0 to 0 at t_end; zero afterwards."""
    if t < 0:
        return 0.0
    if t <= t_end:
        return F0 * (1 - t / t_end)
    return 0.0

def central_difference(m, k, F_func, dt, t_end, d0=0.0, v0=0.0):
    n_steps = int(np.round(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, n_steps)
    d = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    
    # Step 1: Initial conditions (from slide)
    d[0] = d0
    v[0] = v0
    
    # Step 2: Initial acceleration
    F0 = F_func(0.0)
    a[0] = (F0 - k * d[0]) / m
    
    # "Ghost" displacement at i=-1
    d_minus1 = d[0] - dt * v[0] + 0.5 * (dt**2) * a[0]
    
    # Explicit recurrence
    M = m
    K = k
    for i in range(0, n_steps - 1):
        Fi = F_func(t[i])
        # M d_{i+1} = dt^2 F_i + (2M - dt^2 K) d_i - M d_{i-1}
        d_ip1 = ( (dt**2) * Fi + (2*M - (dt**2)*K) * d[i] - M * (d_minus1 if i==0 else d[i-1]) ) / M
        # acceleration and velocity at i (centered)
        a[i] = (Fi - K * d[i]) / M
        if i >= 1:
            v[i] = (d[i+1] - d[i-1]) / (2*dt)  # will fill for i=1..n-2 later
        # shift
        if i == 0:
            d[1] = d_ip1
        else:
            d[i+1] = d_ip1
    
    # Acceleration at last step using equilibrium
    a[-1] = (F_func(t[-1]) - K * d[-1]) / M
    # Fill velocities at boundaries with one-sided central approximations
    v[1:-1] = (d[2:] - d[:-2]) / (2*dt)
    v[0] = (d[1] - d_minus1) / (2*dt)
    v[-1] = (d[-1] - d[-3]) / (2*dt) if n_steps >= 3 else v[0]
    
    return t, d, v, a

# Parameters from slide
m = 31.83     # lb*s^2/in
k = 100.0     # lb/in
dt = 0.05     # s
t_end = 0.20  # s

t, d, v, a = central_difference(m, k, force_ramp, dt, t_end, d0=0.0, v0=0.0)

# Print step-by-step values to match slides
print("Central Difference (m=31.83, k=100, Δt=0.05 s)")
print(f"{'i':>2} {'t(s)':>6} {'F(t) (lb)':>12} {'d (in)':>12} {'v (in/s)':>12} {'a (in/s^2)':>14}")
d_minus1_example = 0 - dt*0 + 0.5*(dt**2)*((2000 - 100*0)/m)
print(f" -1 {(-dt):6.2f} {force_ramp(-dt):12.2f} {d_minus1_example:12.5f} {'-':>12} {'-':>14}")
for i, ti in enumerate(t):
    print(f"{i:2d} {ti:6.2f} {force_ramp(ti):12.2f} {d[i]:12.5f} {v[i]:12.5f} {a[i]:14.5f}")

# Plots (one per figure, no style/colors specified)
plt.figure()
plt.plot(t, d, marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Displacement d (in)")
plt.title("Central Difference: Displacement")

plt.figure()
plt.plot(t, v, marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Velocity v (in/s)")
plt.title("Central Difference: Velocity")

plt.figure()
plt.plot(t, a, marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration a (in/s^2)")
plt.title("Central Difference: Acceleration")

plt.show()
