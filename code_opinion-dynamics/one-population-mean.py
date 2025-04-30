import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os



# -----------------------------
# FUNCTIONS
# -----------------------------

def nabla_morse_potential(r, c_rep, c_att, l_rep, l_att):
    return -(c_rep/l_rep) * np.exp(-r/l_rep) + (c_att/l_att) * np.exp(-r/l_att)


def ode_system(x_step, v_step, w_step, n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue):
    v_blue, v_red = 1, -1       # (1,1) and (-1,-1)
    w_blue, w_red = 1, -1

    term_1 = (alpha - beta*np.sum(v_step**2, axis=1))[:, np.newaxis] * v_step

    term_2 = - potential_sum(x_step, nabla_u) / n

    term_3_red = tau_red * velocity_alignment(v_step, v_red, w_step, w_red, r_w)
    term_3_blue = tau_blue * velocity_alignment(v_step, v_blue, w_step, w_blue, r_w)
    term_3 = term_3_red + term_3_blue

    phi_sum = opinion_alignment_sum(x_step, w_step, n, r_x, r_w)

    dx = v_step
    dv = term_1 + term_2 + term_3
    dw = phi_sum/n + tau_red*(w_red-w_step) + tau_blue*(w_blue-w_step)

    return dx, dv, dw


def potential_sum(x_step, nabla_u):
    # Compute m*m*2 matrix where x_diff(i,j) = x_j - x_i
    # (each element of the matrix is a point in R^2)
    xi = x_step[:, np.newaxis, :]
    xj = x_step[np.newaxis, :, :]
    x_diff = xi - xj

    # Compute norm of the elements x_diff(i,j), d_ij is an m*m matrix
    d_ij = np.linalg.norm(x_diff, axis=2)
    d_ij[d_ij == 0] = 1

    # Derive forces(i,j) = nabla_u(|x_j-x_i|), m*m*2 matrix, "chain rule"
    forces = nabla_u(d_ij)[:, :, np.newaxis] * x_diff / d_ij[:, :, np.newaxis]

    # Return array m*2 such that sum_forces(i) = sum_{j} nabla_u(|x_j-x_i|)
    sum_forces = np.sum(forces, axis=1)
    return sum_forces


def velocity_alignment(v_step, v_ref, w_step, w_ref, r_w):
    bool_aligned = np.abs(w_step - w_ref) < r_w
    
    # change bool = [0 1 0 0...] to bool = [[0], [1], [0], [0] ...]
    return bool_aligned[:, np.newaxis] * (v_ref - v_step)


def opinion_alignment_sum(x_step, w_step, n, r_x, r_w):
    wi = w_step[:, np.newaxis]
    wj = w_step[np.newaxis, :]
    xi = x_step[:, np.newaxis, :]
    xj = x_step[np.newaxis, :, :]

    x_diff = xj - xi
    w_diff = wj - wi
    d_ij = np.linalg.norm(x_diff, axis=2)

    bool_phi = (np.abs(w_diff) < r_w) & (d_ij < r_x)
    phi_sum = np.sum(bool_phi*w_diff, axis=1)
    return phi_sum



# -----------------------------
# PARAMETERS
# -----------------------------

t_final = 100
dt = 1.0e-2
steps = int(np.floor(t_final / dt))

n = 100

# Case II
alpha = 1
beta = 0.5
c_att = 50
l_att = 1
c_rep = 60
l_rep = 0.5
r_x = 1
r_w = 0.5

# # Case VIII
# alpha = 1
# beta = 5
# c_att = 100
# l_att = 1.2
# c_rep = 350
# l_rep = 0.8
# r_x = 0.5
# r_w = 1

tau_blue = 0.1
tau_red = 0.01

nabla_u = lambda r: nabla_morse_potential(r, c_rep, c_att, l_rep, l_att)
ode = lambda x_step, v_step, w_step: ode_system (
    x_step, v_step, w_step, n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue)



# -----------------------------
# SIMULATION FUNCTION
# -----------------------------

def simulate():
    x = np.zeros((steps, n, 2))
    v = np.zeros((steps, n, 2))
    w = np.zeros((steps, n))
    
    x[0] = -1 + 2*np.random.rand(n, 2)
    v[0] = -1 + 2*np.random.rand(n, 2)
    w[0] = -1 + 2*np.random.rand(n)

    dx_0, dv_0, dw_0 = ode(x[0], v[0], w[0])
    x[1] = x[0] + dt*dx_0
    v[1] = v[0] + dt*dv_0
    w[1] = w[0] + dt*dw_0
    
    for i in range(1, steps-1):
        dx_1, dv_1, dw_1 = ode(x[i], v[i], w[i])

        x[i+1] = x[i] + (dt/2) * (3*dx_1 - dx_0)
        v[i+1] = v[i] + (dt/2) * (3*dv_1 - dv_0)
        w[i+1] = w[i] + (dt/2) * (3*dw_1 - dw_0)

        dx_0, dv_0, dw_0 = dx_1, dv_1, dw_1

        if np.any(np.isnan(x[i])) or np.any(np.isinf(x[i])):
            raise ValueError(f'NaN or Inf encountered at time step {i}')
    
    return x, v, w



# -----------------------------
# SIMULATION and AVERAGING
# -----------------------------

runs = 5

all_w = np.zeros((runs, steps, n))
all_v_means = np.zeros((runs, steps, 2))
ensemble_avg_opinion = np.zeros(steps)

np.random.seed(1234)
for run in range(runs):
    x, v, w = simulate()

    all_v_means[run] = np.mean(v, axis=1)
    all_w[run] = w
    ensemble_avg_opinion += np.mean(w, axis=1)

v_means = all_v_means.transpose(1, 0, 2)
ensemble_avg_opinion /= runs



# -----------------------------
# DATA & PLOTS
# -----------------------------

output_folder = 'figures/one-population'
os.makedirs(output_folder, exist_ok=True)


# Plot mean opinions

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(range(1, steps+1), ensemble_avg_opinion, 'k')
ax.set_title("Mean opinion", fontweight="bold", fontsize=16)
ax.set_xlabel("timestep", fontsize=14)
ax.set_ylabel("opinion", fontsize=14)
ax.set_xlim(1, steps)
ax.set_ylim(-1, 1)

# ax.set_xscale('log')
plt.tight_layout()
output_file = os.path.join(output_folder, f"avg.svg")
plt.savefig(output_file)


# Plot all opinions

plt.figure()
fig, ax = plt.subplots(figsize=(5,4))
for run in range(runs):
    ax.plot(range(1, steps+1), all_w[run], 'k', alpha=0.1)
ax.set_title("All opinions", fontweight="bold", fontsize=16)
ax.set_xlabel("timestep", fontsize=14)
ax.set_ylabel("opinion", fontsize=14)
ax.set_xlim(1, steps)
ax.set_ylim(-1, 1)

# ax.set_xscale('log')
plt.tight_layout()
output_file = os.path.join(output_folder, f"all.svg")
plt.savefig(output_file)


# PLOT: mean velocities over time

# time_indices = np.arange(0, steps, 1000)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for t in time_indices:
#     ax.scatter(v[t, :, 0], v[t, :, 1], t, color='b', marker='o')
#     ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('steps')
# ax.set_title('Mean velocities over time')
# plt.grid(True)

# output_file = os.path.join(output_folder, 'velocities_over_time.svg')
# plt.savefig(output_file)

timesteps_to_plot = np.arange(0, steps, 100)
v_means_subsampled = v_means[timesteps_to_plot]

x = v_means_subsampled[..., 0].flatten()
y = v_means_subsampled[..., 1].flatten()
z = np.repeat(timesteps_to_plot, v_means_subsampled.shape[1])

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', alpha=0.6)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('timestep')
plt.title('Mean velocities')
output_file = os.path.join(output_folder, f"velocities.svg")
plt.savefig(output_file)

# plt.show()