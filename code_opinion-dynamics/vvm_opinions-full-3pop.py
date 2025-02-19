import os
import numpy as np
import matplotlib.pyplot as plt


def morse_potential(r, c_rep, c_att, l_rep, l_att):
    return -(c_rep / l_rep) * np.exp(-r / l_rep) + (c_att / l_att) * np.exp(-r / l_att)


def velocity_alignment(v, w1, r_w, vt, wt):
    if np.abs(w1 - wt) < r_w:
        return (vt - v)
    else:
        return 0


def potential_sum(x_step, y_step, nabla_u):
    xi = x_step[:, np.newaxis, :]
    xj = y_step[np.newaxis, :, :]
    z = xi - xj
    d_ij = np.sqrt(np.sum(z ** 2, axis=2))
    d_ij[d_ij == 0] = 1  # Avoid division by zero
    potential_term = nabla_u(d_ij)
    forces = -potential_term[:, :, np.newaxis] * z / d_ij[:, :, np.newaxis]
    return np.sum(forces, axis=1)


def opinion_alignment_sum(x_step, y_step, w_step, q, n_x, n_y, r_x, r_w):
    wi = w_step[:, np.newaxis]
    wj = q[np.newaxis, :]
    xi = x_step[:, np.newaxis, :]
    xj = y_step[np.newaxis, :, :]
    z = xi - xj
    distXij = np.sqrt(np.sum(z ** 2, axis=2))
    bool_phi = (np.abs(wi - wj) < r_w) & (distXij < r_x)
    return np.sum(bool_phi * (wj - wi), axis=1)


def ode_system(x_step, v_step, w_step, n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f):
    ns = np.array([n_l, n_f, n_u])
    n = n_l + n_f + n_u
    
    xs = [x_step[:n_l], x_step[n_l:n_l + n_f], x_step[n_l + n_f:]]
    vs = [v_step[:n_l], v_step[n_l:n_l + n_f], v_step[n_l + n_f:]]
    ws = [w_step[:n_l], w_step[n_l:n_l + n_f], w_step[n_l + n_f:]]
    
    v_blue, v_red = 1, -1
    w_blue, w_red = 1, -1
    dvs = [np.zeros_like(v) for v in vs]
    
    for i in range(3):
        term_1 = (alpha - beta * np.sum(vs[i] ** 2, axis=1))[:, np.newaxis] * vs[i]
        term_2 = sum(potential_sum(xs[i], xs[j], nabla_u) / ns[j] for j in range(3))
        term_3 = gammas_blue[i] * velocity_alignment(vs[i], ws[i], r_w, v_blue, w_blue) + \
                 gammas_red[i] * velocity_alignment(vs[i], ws[i], r_w, v_red, w_red)
        dvs[i] = term_1 + term_2 + term_3
    
    dv = np.vstack(dvs)
    dx = np.vstack(vs)
    dw = np.zeros_like(w_step)  # Placeholder, needs opinion alignment
    return dx, dv, dw



# All the parameters used in the model are stated and described right bellow

# Other possible parameters. Changing tau_red and tau_blue also gives very rich dynamics
# alpha = 1; beta = 5;
# c_att = 100; l_att = 1.2; c_rep = 350; l_rep = 0.8; 
# l_att = 2.2


# PROBLEM DATA

n_l = 20
n_f = 50
n_u = 50                                   # we need to change n_u=2, 20, 80n = n_l + n_f + n_u

t_final = 100
dt = 1.0e-2
steps = int(np.floor(t_final / dt))

x = np.zeros((n, 2, steps))
v = np.zeros((n, 2, steps))
w = np.zeros((n, 1, steps))

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

x[:, :, 0] = np.vstack([
    np.random.uniform(-1, 1, (n_l, 2)), 
    np.random.uniform(-1, 1, (n_f, 2)), 
    np.random.uniform(-1, 1, (n_u, 2))
    ])
v[:, :, 0] = np.vstack([
    np.random.uniform(-1, 1, (n_l, 2)),
    np.random.uniform(-1, 1, (n_f, 2)),
    np.random.uniform(-1, 1, (n_u, 2))
    ])
w[:, 0] = np.vstack([
    np.random.uniform(-1, 1, (n_l, 1)), 
    np.random.uniform(-1, 1, (n_f, 1)), 
    np.zeros((n_u, 1))                      # uninformed opinions start at zero
    ])


# INTEGRATION STEP

dx_1, dv_1, dw_1 = ode_system(x[:, :, 0], v[:, :, 0], w[:, 0], n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)
x[:, :, 1] = x[:, :, 0] + dt * dx_1
v[:, :, 1] = v[:, :, 0] + dt * dv_1
w[:, 1] = w[:, 0] + dt * dw_1

dx_2, dv_2, dw_2 = ode_system(x[:, :, 1], v[:, :, 1], w[:, 1], n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)

for i in range(2, steps):
    x[:, :, i] = x[:, :, i-1] + (dt / 2) * (3 * dx_2 - dx_1)
    v[:, :, i] = v[:, :, i-1] + (dt / 2) * (3 * dv_2 - dv_1)
    w[:, i] = w[:, i-1] + (dt / 2) * (3 * dw_2 - dw_1)
    dx_1, dv_1, dw_1 = dx_2, dv_2, dw_2
    dx_2, dv_2, dw_2 = ode_system(x[:, :, i], v[:, :, i], w[:, i], n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)

# Plot Final Velocities

plt.figure()
plt.scatter(x[:n_l, 0, -1], x[:n_l, 1, -1], c='b')
plt.scatter(x[n_l:n_l+n_f, 0, -1], x[n_l:n_l+n_f, 1, -1], c='r')
plt.scatter(x[n_l+n_f:, 0, -1], x[n_l+n_f:, 1, -1], c='k')
plt.quiver(x[:, 0, -1], x[:, 1, -1], v[:, 0, -1], v[:, 1, -1], color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()
