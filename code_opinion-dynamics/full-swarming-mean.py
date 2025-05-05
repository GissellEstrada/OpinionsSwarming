import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os



# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def nabla_morse_potential(r, c_rep, c_att, l_rep, l_att):
    return -(c_rep / l_rep) * np.exp(-r / l_rep) + (c_att / l_att) * np.exp(-r / l_att)


def ode_system(x_step, v_step, w_step, n_l, n_f, n_u, 
               alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f, ks):  
    xs = [
        x_step[:n_l],
        x_step[n_l : n_l+n_f],
        x_step[n_l+n_f:]
    ]
    vs = [
        v_step[:n_l],
        v_step[n_l : n_l+n_f],
        v_step[n_l+n_f:]
    ]
    ws = [
        w_step[:n_l],
        w_step[n_l : n_l+n_f],
        w_step[n_l+n_f:]
    ]
    ns = np.array([n_l, n_f, n_u])
    
    v_blue, v_red = 1, -1
    w_blue, w_red = 1, -1

    dvs = [np.zeros_like(v) for v in vs]
    for i in range(3):
        term_1 = (alpha - beta * np.sum(vs[i] ** 2, axis=1, keepdims=True)) * vs[i]

        term_2 = sum(potential_sum(xs[i], xs[j], nabla_u) / ns[j] for j in range(3))
        
        term_3 = gammas_blue[i] * velocity_alignment(vs[i], v_blue, ws[i], w_blue, r_w) \
                    + gammas_red[i] * velocity_alignment(vs[i], v_red, ws[i], w_red, r_w)
        dvs[i] = term_1 + term_2 + term_3

    phis = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            phis[i,j] = ks[i,j] * opinion_alignment_sum(xs[i], xs[j], ws[i], ws[j], r_x, r_w) / ns[j]

    dw_l = phis[0,0] + phis[0,1] + phis[0,2] - tau_blue_l * (ws[0]-w_blue)
    dw_f = phis[1,0] + phis[1,1] + phis[1,2] - tau_red_f * (ws[1]-w_red)
    dw_u = phis[2,0] + phis[2,1] + phis[2,2]
    
    dx = np.vstack(vs)
    dv = np.vstack(dvs)
    dw = np.hstack([dw_l, dw_f, dw_u])

    return dx, dv, dw


def potential_sum(x1, x2, nabla_u):
    # Compute m*m*2 matrix where x_diff(i,j) = x1_i - x2_j
    # (each element of the matrix is a point in R^2)
    x_diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]

    d_ij = np.sqrt(np.sum(x_diff ** 2, axis=2))
    d_ij[d_ij == 0] = 1

    potential_term = nabla_u(d_ij)
    forces = -potential_term[:, :, np.newaxis] * x_diff / d_ij[:, :, np.newaxis]

    return np.sum(forces, axis=1)


def velocity_alignment(v, v_ref, w, w_ref, r_w):
    bool_alignment = np.abs(w - w_ref) < r_w

    return bool_alignment[:, np.newaxis] * (v_ref - v)


def opinion_alignment_sum(x1, x2, w1, w2, r_x, r_w):
    wi = w1[:, np.newaxis]
    wj = w2
    xi = x1[:, np.newaxis, :]
    xj = x2

    x_diff = xi - xj
    d_ij = np.sqrt(np.sum(x_diff ** 2, axis=2))

    w_diff = wj - wi
    bool_phi = (np.abs(w_diff) < r_w) & (d_ij < r_x)

    return np.sum(bool_phi * (w_diff), axis=1)



# --------------------------------------------------
# PROBLEM DATA
# --------------------------------------------------

n_l = 20
n_f = 50
n_u = 20
n = n_l + n_f + n_u

t_final = 100
dt = 1.0e-2
steps = int(np.floor(t_final / dt))

r_x = 1
r_w = 0.5
alpha = 1
beta = 0.5

gammas_red = [1, 1, 0]
gammas_blue = [1, 1, 0]
# tau_blue_l = 0
# tau_red_f = 0
tau_blue_l = 0.1
tau_red_f = 0.01

ks = np.array([
    [1, 1, 0],  # ll lf lu
    [1, 1, 1],  # fl ff fu
    [0, 0, 0]   # ul uf uu
])

c_att, l_att = 50, 1
c_rep, l_rep = 60, 0.5

nabla_u = lambda r: nabla_morse_potential(r, c_rep, c_att, l_rep, l_att)
ode = lambda x_step, v_step, w_step: ode_system(x_step, v_step, w_step, n_l, n_f, n_u, 
               alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f, ks)



# -----------------------------
# SIMULATION FUNCTION
# -----------------------------

def simulate():
    x = np.zeros((steps, n, 2))
    v = np.zeros((steps, n, 2))
    w = np.zeros((steps, n))
    
    # if n_u == 1: n_u = 0

    x[0] = np.random.uniform(-1, 1, (n, 2))
    v[0] = np.random.uniform(-1, 1, (n, 2))
    w[0] = np.hstack([
        np.random.uniform(-1, 1, (n_l+n_f,)), 
        np.zeros((n_u,))    # uninformed opinions start at zero
    ])

    # if n_u == 0: n_u = 1

    dx_0, dv_0, dw_0 = ode(x[0], v[0], w[0])
    x[1] = x[0] + dt*dx_0
    v[1] = v[0] + dt*dv_0
    w[1] = w[0] + dt*dw_0
    
    for i in range(1, steps-1):
        dx_1, dv_1, dw_1 = ode(x[i], v[i], w[i])
        
        x[i+1] = x[i] + (dt / 2) * (3 * dx_1 - dx_0)
        v[i+1] = v[i] + (dt / 2) * (3 * dv_1 - dv_0)
        w[i+1] = w[i] + (dt / 2) * (3 * dw_1 - dw_0)

        dx_0, dv_0, dw_0 = dx_1, dv_1, dw_1

        if np.any(np.isnan(x[i])) or np.any(np.isinf(x[i])):
            raise ValueError(f'NaN or Inf encountered at time step {i}')
    
    return x, v, w



# -----------------------------
# SIMULATION and AVERAGING
# -----------------------------

runs = 50

all_w = np.zeros((runs, steps, n))
all_v_means = np.zeros((runs, steps, 2))
ensemble_avg_opinion = np.zeros(steps)
polarisation = np.zeros((runs, steps))
momentum = np.zeros((runs, steps))

np.random.seed(1234)

for run in range(runs):
    x, v, w = simulate()

    all_v_means[run] = np.mean(v, axis=1)
    all_w[run] = w
    ensemble_avg_opinion += np.mean(w, axis=1)

    sum_velocities = np.sum(v, axis=1)
    polarisation_numerator = np.linalg.norm(sum_velocities, axis=1)
    norms_velocities = np.linalg.norm(v, axis=2)
    polarisation_denominator = np.sum(norms_velocities, axis=1)
    polarisation[run] = polarisation_numerator / polarisation_denominator

    x_cm = np.mean(x, axis=1, keepdims=True)  
    r = x - x_cm
    cross = r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]
    mom_numerator = np.abs(np.sum(cross, axis=1)) 
    norms_r = np.linalg.norm(r, axis=2)
    mom_denominator = np.sum(norms_r * norms_velocities, axis=1) 

    momentum[run] = mom_numerator / mom_denominator 

ensemble_avg_opinion /= runs
ensemble_velocity   = np.mean(all_v_means, axis=0)
momentum_mean = np.mean(momentum, axis=0)
polarisation_mean = np.mean(polarisation, axis=0)



# -----------------------------
# DATA & PLOTS
# -----------------------------

output_folder = 'figures/full-swarming'
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
output_file = os.path.join(output_folder, f"case-4-mean_opinion.svg")
plt.savefig(output_file)


# Plot all opinions

plt.figure()
fig, ax = plt.subplots(figsize=(5,4))
for run in range(runs):
    # ax.plot(range(1, steps+1), all_w[run], 'k', alpha=0.1)
    plt.plot(range(1, steps+1), all_w[run, :, :n_l], 'b', alpha=0.1)
    plt.plot(range(1, steps+1), all_w[run, :, n_l:n_l+n_f], 'r', alpha=0.1)
    plt.plot(range(1, steps+1), all_w[run, :, n_l+n_f:], 'k', alpha=0.1)
ax.set_title("All opinions", fontweight="bold", fontsize=16)
ax.set_xlabel("timestep", fontsize=14)
ax.set_ylabel("opinion", fontsize=14)
ax.set_xlim(1, steps)
ax.set_ylim(-1.05, 1.05)

# ax.set_xscale('log')
plt.tight_layout()
output_file = os.path.join(output_folder, f"case-4-mean_all-opinions.svg")
plt.savefig(output_file)


# PLOT: velocity components over time

fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(6, 5),
    sharex=True,
    constrained_layout=True
)

ax1.plot(range(1, steps+1), ensemble_velocity[:, 0], 'b-', linewidth=2)
ax1.set_xlabel('timestep')
ax1.set_ylabel('v_x')
ax1.set_title('Mean velocity component v_x over time', fontweight='bold')
ax1.grid(True)

ax2.plot(range(1, steps+1), ensemble_velocity[:, 1], 'r-', linewidth=2)
ax2.set_xlabel('timestep')
ax2.set_ylabel('v_y')
ax2.set_title('Mean velocity component v_y over time', fontweight='bold')
ax2.grid(True)

output_file = os.path.join(output_folder, 'case-4-mean_velocity-components.svg')
fig.savefig(output_file)


# # PLOT: mean velocities over time

# timesteps_to_plot = np.arange(0, steps, 100)
# v_means_subsampled = v_means[timesteps_to_plot]

# x = v_means_subsampled[..., 0].flatten()
# y = v_means_subsampled[..., 1].flatten()
# z = np.repeat(timesteps_to_plot, v_means_subsampled.shape[1])

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', alpha=0.6)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('timestep')
# plt.title('Mean velocities')
# output_file = os.path.join(output_folder, f"case-4-mean_velocities.svg")
# plt.savefig(output_file)


# Plot mean quantities

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(range(1, steps+1), polarisation_mean, 'k', label="Polarisation")
ax.plot(range(1, steps+1), momentum_mean, 'r', label="Momentum")
ax.set_title("Mean quantities over time", fontweight="bold", fontsize=16)
ax.set_xlabel('timestep', fontsize=14)
ax.set_ylabel('amount', fontsize=14)
ax.set_xlim(1, steps)
ax.set_ylim(-0.05, 1.05)

plt.legend()

output_file = os.path.join(output_folder, 'case-4-mean_quantities.svg')
plt.savefig(output_file)


# Plot all polarisations

fig, ax = plt.subplots(figsize=(5, 4))
for run in range(runs):
    ax.plot(range(1, steps+1), polarisation[run], 'k', alpha=0.25)
ax.plot(range(1, steps+1), ensemble_avg_opinion, 'k')
ax.set_title("All polarisations", fontweight="bold", fontsize=16)
ax.set_xlabel("timestep", fontsize=14)
ax.set_ylabel("amount", fontsize=14)
ax.set_xlim(1, steps)
ax.set_ylim(-0.05, 1.05)

# ax.set_xscale('log')
plt.tight_layout()
output_file = os.path.join(output_folder, f"case-4-mean_all-polarisations.svg")
plt.savefig(output_file)


# Plot all momentums

fig, ax = plt.subplots(figsize=(5, 4))
for run in range(runs):
    ax.plot(range(1, steps+1), momentum[run], 'r', alpha=0.25)
ax.set_title("All momentums", fontweight="bold", fontsize=16)
ax.set_xlabel("timestep", fontsize=14)
ax.set_ylabel("amount", fontsize=14)
ax.set_xlim(1, steps)
ax.set_ylim(-0.05, 1.05)

# ax.set_xscale('log')
plt.tight_layout()
output_file = os.path.join(output_folder, f"case-4-mean_all-momentums.svg")
plt.savefig(output_file)