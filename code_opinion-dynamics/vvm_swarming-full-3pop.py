import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def morse_potential(r, c_rep, c_att, l_rep, l_att):
    return -(c_rep / l_rep) * np.exp(-r / l_rep) + (c_att / l_att) * np.exp(-r / l_att)


def ode_system(x_step, v_step, w_step, n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f):
    ns = np.array([n_l, n_f, n_u])
    n = n_l + n_f + n_u
    
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
    
    v_blue, v_red = 1, -1
    w_blue, w_red = 1, -1

    dvs = [np.zeros_like(v) for v in vs]
    for i in range(3):
        term_1 = (alpha - beta * np.sum(vs[i] ** 2, axis=1, keepdims=True)) * vs[i]

        term_2 = sum(potential_sum(xs[i], xs[j], nabla_u) / ns[j] for j in range(3))
        
        term_3 = gammas_blue[i] * velocity_alignment(vs[i], ws[i], r_w, v_blue, w_blue) \
                    + gammas_red[i] * velocity_alignment(vs[i], ws[i], r_w, v_red, w_red)
                    
        dvs[i] = term_1 + term_2 + term_3
    
    dx = np.vstack(vs)
    dv = np.vstack(dvs)

    # phis = [[1 / ns[j] * opinion_alignment_sum(xs[i], xs[j], ws[i], ws[j], ns[i], ns[j], r_x, r_w)
            #  for j in range(3)] for i in range(3)]

    phis = [[None for _ in range(3)] for _ in range(3)]
    
    for i in range(3):
        for j in range(3):
            phis[i][j] = opinion_alignment_sum(xs[i], xs[j], ws[i], ws[j], ns[i], ns[j], r_x, r_w) / ns[j]
    
    dw_l = phis[0][0] + phis[0][1] + 0*phis[0][2] - tau_blue_l * (ws[0]-w_blue)
    dw_f = phis[1][1] + phis[1][0] + phis[1][2] - tau_red_f * (ws[1]-w_red)
    dw_u = 0*phis[2][2] + 0*phis[2][1] + 0*phis[2][0]
    
    dw = np.hstack([dw_l, dw_f, dw_u])

    return dx, dv, dw


def potential_sum(x_step, y_step, nabla_u):
    xi = x_step[:, np.newaxis, :]
    xj = y_step[np.newaxis, :, :]

    x_diff = xi - xj
    d_ij = np.sqrt(np.sum(x_diff ** 2, axis=2))
    d_ij[d_ij == 0] = 1

    potential_term = nabla_u(d_ij)
    forces = -potential_term[:, :, np.newaxis] * x_diff / d_ij[:, :, np.newaxis]

    return np.sum(forces, axis=1)


def velocity_alignment(v, wi, r_w, vt, wt):
    bool_alignment = np.abs(wi - wt) < r_w
    # print(bool_alignment)
    # print(bool_alignment[:, np.newaxis])
    # print((vt - v).shape)
    return bool_alignment[:, np.newaxis] * (vt - v)


def opinion_alignment_sum(x1_step, x2_step, w1_step, w2_step, n_x, n_y, r_x, r_w):
    wi = w1_step[:, np.newaxis]
    wj = w2_step
    xi = x1_step[:, np.newaxis, :]
    xj = x2_step

    x_diff = xi - xj
    d_ij = np.sqrt(np.sum(x_diff ** 2, axis=2))

    w_diff = wj - wi
    bool_phi = (np.abs(w_diff) < r_w) & (d_ij < r_x)

    return np.sum(bool_phi * (w_diff), axis=1)


# All the parameters used in the model are stated and described right bellow

# Other possible parameters. Changing tau_red and tau_blue also gives very rich dynamics
# alpha = 1; beta = 5;
# c_att = 100; l_att = 1.2; c_rep = 350; l_rep = 0.8; 
# l_att = 2.2


# PROBLEM DATA

n_l = 20
n_f = 50
n_u = 50                                   # we need to change n_u=2, 20, 80n = n_l + n_f + n_u
n = n_l + n_f + n_u

t_final = 100
dt = 1.0e-2
steps = int(np.floor(t_final / dt))

x = np.zeros((steps, n, 2))
v = np.zeros((steps, n, 2))
w = np.zeros((steps, n))

r_x = 1
r_w = 0.5
alpha = 1
beta = 0.5
gammas_red = [1, 1, 0]
gammas_blue = [1, 1, 0]
tau_blue_l = 0.1
tau_red_f = 0.01

c_att, l_att = 50, 1
c_rep, l_rep = 60, 0.5
nabla_u = lambda r: morse_potential(r, c_rep, c_att, l_rep, l_att)

# Initial conditions

np.random.seed(1234)

# [VVM] why is this the domain
x[0] = np.random.uniform(-1, 1, (n, 2))
v[0] = np.random.uniform(-1, 1, (n, 2))
w[0] = np.hstack([
    np.random.uniform(-1, 1, (n-n_u,)), 
    np.zeros((n_u,))    # uninformed opinions start at zero
])


# INTEGRATION STEP

dx_1, dv_1, dw_1 = ode_system(x[0], v[0], w[0], n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)
x[1] = x[0] + dt * dx_1
v[1] = v[0] + dt * dv_1
w[1] = w[0] + dt * dw_1

dx_2, dv_2, dw_2 = ode_system(x[1], v[1], w[1], n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)

for i in range(2, steps):
    x[i] = x[i-1] + (dt / 2) * (3 * dx_2 - dx_1)
    v[i] = v[i-1] + (dt / 2) * (3 * dv_2 - dv_1)
    w[i] = w[i-1] + (dt / 2) * (3 * dw_2 - dw_1)
    dx_1, dv_1, dw_1 = dx_2, dv_2, dw_2
    dx_2, dv_2, dw_2 = ode_system(x[i], v[i], w[i], n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)


# # Plot Final Velocities

# Save figure
output_folder = 'Figures_Mill_allcases'  # Folder to save the figures
# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# plt.figure()
# plt.scatter(x[-1, :n_l, 0], x[-1, :n_l, 1], c='b')
# plt.scatter(x[-1, n_l:n_l+n_f, 0], x[-1, n_l:n_l+n_f, 1], c='r')
# plt.scatter(x[-1, n_l+n_f:, 0], x[-1, n_l+n_f:, 1], c='k')
# plt.quiver(x[-1, :, 0], x[-1, :, 1], v[-1, :, 0], v[-1, :, 1], color='r')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.axis('equal')
# plt.show()


# Plot of the final velocity configuration
plt.figure()
plt.plot(x[-1, :n_l, 0], x[-1, :n_l, 1], 'o', markeredgecolor='b', markerfacecolor='b')
plt.plot(x[-1, n_l:n_l+n_f, 0], x[-1, n_l:n_l+n_f, 1], 'o', markeredgecolor='r', markerfacecolor='r')
plt.plot(x[-1, n_l+n_f:n, 0], x[-1, n_l+n_f:n, 1], 'o', markeredgecolor='k', markerfacecolor='k')
plt.quiver(x[-1, :, 0], x[-1, :, 1], v[-1, :, 0], v[-1, :, 1], color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Velocities at the final time')
plt.axis('equal')
plt.savefig(os.path.join(output_folder, f'3PopOp_WithUninformed_FLUInteraction_case2_NL{n_l}_alpha_{alpha}_beta_{beta}_Ca_{c_att}_la_{l_att}_Cr_{c_rep}_lr_{l_rep}_rx_{r_x}_rw_{r_w}_taur_{tau_blue_l}_taub_{tau_red_f}.png'))
plt.show()

# Plot for several times
time = np.arange(1, x.shape[0], 1000)  # Time indices corresponding to the 1:1000:end sampling
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for t in time:
    ax.scatter(x[t, :, 0], x[t, :, 1], np.full(n, t), c='b', marker='o', edgecolors='b')
for t in time:
    ax.quiver(x[t, :, 0], x[t, :, 1], np.full(n, t), v[t, :, 0], v[t, :, 1], np.zeros(n), color='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.set_title('Velocities Over Time')
plt.grid(True)
plt.savefig(os.path.join(output_folder, '3PopulationVelocOverTime_FLU3.png'))
plt.show()

# Plot of the trajectories for some times
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for t in time:
    ax.scatter(x[t, :n_l, 0], x[t, :n_l, 1], np.full(n_l, t), c='b', marker='o', edgecolors='b')
    ax.scatter(x[t, n_l:n_l+n_f, 0], x[t, n_l:n_l+n_f, 1], np.full(n_f, t), c='r', marker='o', edgecolors='r')
    ax.scatter(x[t, n_l+n_f:n, 0], x[t, n_l+n_f:n, 1], np.full(n_u, t), c='k', marker='o', edgecolors='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.set_title('Position Over Time')
plt.grid(True)
plt.savefig(os.path.join(output_folder, '3PopulationPositOverTime_FLU3.png'))
plt.show()

# Another figure
plt.figure()
plt.plot(range(steps), w[:, :n_l], 'b', linewidth=1.5)
plt.plot(range(steps), w[:, n_l:n_l+n_f], 'r', linewidth=1.5)
plt.plot(range(steps), w[:, n_l+n_f:], 'k', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Opinion')
plt.title('Opinion Over Time')
plt.grid(True)
plt.savefig(os.path.join(output_folder, f'OpinionEvol_WithUninformed_FLUInteraction_case2_NL{n_l}_alpha_{alpha}_beta_{beta}_Ca_{c_att}_la_{l_att}_Cr_{c_rep}_lr_{l_rep}_rx_{r_x}_rw_{r_w}_taur_{tau_blue_l}_taub_{tau_red_f}.png'))
plt.show()

# Compute mean velocity components over time
mean_vx_time = np.mean(v[:, :, 0], axis=1)
mean_vy_time = np.mean(v[:, :, 1], axis=1)

time = np.arange(v.shape[0])
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, mean_vx_time, 'b-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Mean V_x')
plt.title('Mean Velocity Component V_x Over Time')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(time, mean_vy_time, 'r-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Mean V_y')
plt.title('Mean Velocity Component V_y Over Time')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'VelocityComponents_FLU2.png'))
plt.show()

# Mean opinion over time
mean_opinion = np.mean(w, axis=1)
plt.figure()
plt.plot(time, mean_opinion, 'b', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Mean Opinion')
plt.title('Mean Opinion of the Whole Population')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'MeanOpinion_FLU2.png'))
plt.show()

# Mean velocity magnitude
mean_velocity_magnitude = np.sqrt(mean_vx_time**2 + mean_vy_time**2)
plt.figure()
plt.plot(time, mean_velocity_magnitude, 'k-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Mean Velocity Magnitude')
plt.title('Mean Velocity Magnitude Over Time')
plt.grid(True)
plt.savefig(os.path.join(output_folder, 'MeanVelocityMagnitude_FLU2.png'))
plt.show()

