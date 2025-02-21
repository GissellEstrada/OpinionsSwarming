import numpy as np
import matplotlib.pyplot as plt
import os

# Morse potential function
def morse_potential(r, c_rep, c_att, l_rep, l_att):
    return -(c_rep / l_rep) * np.exp(-r / l_rep) + (c_att / l_att) * np.exp(-r / l_att)

def potential_sum(x_step, nabla_u):
    xi = x_step[:, np.newaxis, :]
    xj = x_step[np.newaxis, :, :]
    z = xi - xj
    d_ij = np.linalg.norm(z, axis=2)
    d_ij[d_ij == 0] = 1
    forces = nabla_u(d_ij)[:, :, np.newaxis] * z / d_ij[:, :, np.newaxis]
    return np.sum(forces, axis=1)

def velocity_alignment(v_step, v_ref, w_step, w_ref, r_w):
    bool_aligned = np.abs(w_step - w_ref) < r_w
    return bool_aligned[:, np.newaxis] * (v_ref - v_step)

def opinion_alignment_sum(x_step, w_step, n, r_x, r_w):
    wi = np.tile(w_step, (n, 1))
    wj = wi.T
    xi = x_step[:, np.newaxis, :]
    xj = x_step[np.newaxis, :, :]
    d_ij = np.linalg.norm(xi - xj, axis=2)
    bool_phi = (np.abs(wi - wj) < r_w) & (d_ij < r_x)
    return np.sum(bool_phi * (wj - wi), axis=1)

def ode_system(x_step, v_step, w_step, n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue):
    v_blue, v_red = 1, -1
    w_blue, w_red = 1, -1
    term_1 = (alpha - beta * np.sum(v_step**2, axis=1))[:, np.newaxis] * v_step
    term_2 = -1 / n * potential_sum(x_step, nabla_u)
    term_3_red = tau_red * velocity_alignment(v_step, v_red, w_step, w_red, r_w)
    term_3_blue = tau_blue * velocity_alignment(v_step, v_blue, w_step, w_blue, r_w)
    phi = opinion_alignment_sum(x_step, w_step, n, r_x, r_w)
    dx = v_step
    dv = term_1 + term_2 + term_3_red + term_3_blue
    dw = phi / n + tau_red * (w_red - w_step) + tau_blue * (w_blue - w_step)
    return dx, dv, dw

# Problem data
t_final = 100
dt = 1.0e-2
steps = int(np.floor(t_final / dt))

n = 100
x = np.zeros((steps, n, 2))
v = np.zeros((steps, n, 2))
w = np.zeros((steps, n))

r_x = 0.5
r_w = 1
alpha = 1
beta = 5
tau_blue = 0.1
tau_red = 0.1

c_att, l_att = 100, 1.2
c_rep, l_rep = 350, 0.8
nabla_u = lambda r: morse_potential(r, c_rep, c_att, l_rep, l_att)

# Initial conditions

# np.random.seed(1234)

x[0] = -1 + 2 * np.random.rand(n, 2)
v[0] = -1 + 2 * np.random.rand(n, 2)
w[0] = (-1 + 2 * np.random.rand(n))

# Adams-Bashforth 2-step method
for i in range(2, steps):
    dx_1, dv_1, dw_1 = ode_system(x[i - 1], v[i - 1], w[i - 1], n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue)
    x[i] = x[i - 1] + (dt / 2) * (3 * dx_1 - x[i - 2])
    v[i] = v[i - 1] + (dt / 2) * (3 * dv_1 - v[i - 2])
    w[i] = w[i - 1] + (dt / 2) * (3 * dw_1 - w[i - 2])
    if np.any(np.isnan(x[i])) or np.any(np.isinf(x[i])):
        raise ValueError(f'NaN or Inf encountered at time step {i}')

# Plot final velocity configuration
output_folder = 'Figures_onePopNoMill'
os.makedirs(output_folder, exist_ok=True)
plt.figure()
plt.scatter(x[-1, :, 0], x[-1, :, 1], color='b', label='Positions')
plt.quiver(x[-1, :, 0], x[-1, :, 1], v[-1, :, 0], v[-1, :, 1], color='r', label='Velocities')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Velocities at the final time')
plt.axis('equal')
plt.legend()
plt.show()

# Plot opinion over time
plt.figure()
plt.plot(range(steps), w.T, 'k', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Opinion')
plt.title('Opinion Over Time')
plt.grid(True)
plt.show()
