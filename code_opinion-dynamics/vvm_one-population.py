import numpy as np
import matplotlib.pyplot as plt
import os



def nabla_morse_potential(r, c_rep, c_att, l_rep, l_att):
    return -(c_rep / l_rep) * np.exp(-r / l_rep) + (c_att / l_att) * np.exp(-r / l_att)


def ode_system(x_step, v_step, w_step, n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue):
    v_blue, v_red = 1, -1       # (1,1) and (-1,-1)
    w_blue, w_red = 1, -1

    term_1 = (alpha - beta * np.sum(v_step**2, axis=1))[:, np.newaxis] * v_step

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
    # Compute m*m*2 matrix where x_diff(i,j) = x_i - x_j
    # (each element of the matrix is a point in R^2)
    x_diff = x_step[:, np.newaxis, :] - x_step[np.newaxis, :, :]

    # Compute norm of the elements x_diff(i,j), d_ij is an m*m matrix
    d_ij = np.linalg.norm(x_diff, axis=2)
    d_ij[d_ij == 0] = 1

    # Derive forces(i,j) = nabla_u(|x_i-x_j|), m*m*2 matrix, "chain rule"
    forces = nabla_u(d_ij)[:, :, np.newaxis] * x_diff / d_ij[:, :, np.newaxis]

    # Return array m*2 such that sum_forces(i) = sum_{j} nabla_u(|x_i-x_j|)
    sum_forces = np.sum(forces, axis=1)
    return sum_forces


def velocity_alignment(v_step, v_ref, w_step, w_ref, r_w):
    bool_aligned = np.abs(w_step - w_ref) < r_w
    
    # change bool = [0 1 0 0...] to bool = [[0], [1], [0], [0] ...]
    return bool_aligned[:, np.newaxis] * (v_ref - v_step)


def opinion_alignment_sum(x_step, w_step, n, r_x, r_w):
    wi = np.tile(w_step, (n, 1))
    wj = wi.T
    xi = x_step[:, np.newaxis, :]
    xj = x_step[np.newaxis, :, :]

    x_diff = xj - xi
    w_diff = wj - wi
    d_ij = np.linalg.norm(x_diff, axis=2)

    bool_phi = (np.abs(w_diff) < r_w) & (d_ij < r_x)
    phi_sum = np.sum(bool_phi*w_diff, axis=1)
    return phi_sum


parameter_combinations = {
    '1': {
        'r_x': 0.5,
        'r_w': 1,
        'alpha': 1,
        'beta': 5,
        'tau_blue': 0.1,
        'tau_red': 0.1,
        'c_att': 100,
        'l_att': 1.2,
        'c_rep': 350,
        'l_rep': 0.8
    },
    '2': {
        'r_x': 0.5,
        'r_w': 1,
        'alpha': 1,
        'beta': 5,
        'tau_blue': 0.1,
        'tau_red': 0.1,
        'c_att': 100,
        'l_att': 1.2,
        'c_rep': 350,
        'l_rep': 0.8
    },
}



# --------------------------------------------------
# --------------------------------------------------
# PROBLEM DATA

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
c_att = 100
l_att = 1.2
c_rep = 350
l_rep = 0.8

# choice = input("Choose an option: 1, 2, 3, 4 or 5): ")

# if choice in parameter_combinations:
#     params = parameter_combinations[choice]

#     r_x = params['r_x']
#     r_w = params['r_w']
#     alpha = params['alpha']
#     beta = params['beta']
#     tau_blue = params['tau_blue']
#     tau_red = params['tau_red']
#     c_att = params['c_att']
#     l_att = params['l_att']
#     c_rep = params['c_rep']
#     l_rep = params['l_rep']
    
#     print(f"You selected option {choice}")
# else:
#     print("Invalid option. Exiting.")

nabla_u = lambda r: nabla_morse_potential(r, c_rep, c_att, l_rep, l_att)


# INTEGRATION STEP

np.random.seed(1234)

x[0] = -1 + 2*np.random.rand(n, 2)
v[0] = -1 + 2*np.random.rand(n, 2)
w[0] = -1 + 2*np.random.rand(n)

# Preparation: compute the first solution using Euler. Use it to compute the second system.

dx_0, dv_0, dw_0 = ode_system(x[0], v[0], w[0], n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue)
x[1] = x[0] + dt*dx_0
v[1] = v[0] + dt*dv_0
w[1] = w[0] + dt*dw_0

# Adams-Bashforth 2-step method

for i in range(1, steps-1):
    dx_1, dv_1, dw_1 = ode_system(x[i], v[i], w[i], n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue)

    x[i+1] = x[i] + (dt/2) * (3*dx_1 - dx_0)
    v[i+1] = v[i] + (dt/2) * (3*dv_1 - dv_0)
    w[i+1] = w[i] + (dt/2) * (3*dw_1 - dw_0)

    dx_0, dv_0, dw_0 = dx_1, dv_1, dw_1

    if np.any(np.isnan(x[i])) or np.any(np.isinf(x[i])):
        raise ValueError(f'NaN or Inf encountered at time step {i}')


# --------------------------------------------------
# --------------------------------------------------
output_folder = 'figures/one-population'
os.makedirs(output_folder, exist_ok=True)


# PLOT: final velocity configuration

plt.figure()
plt.scatter(x[-1, :, 0], x[-1, :, 1], color='b', label='Positions')
plt.quiver(x[-1, :, 0], x[-1, :, 1], v[-1, :, 0], v[-1, :, 1], color='r', label='Velocities')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('Velocities at the final time')
plt.axis('equal')
plt.legend()

output_file = os.path.join(output_folder, 'final_velocity_configuration.svg')
plt.savefig(output_file)

plt.show()


# PLOT: opinion over time

plt.figure()
plt.plot(range(steps), w, 'k')
plt.xlabel('steps')
plt.ylabel('Opinion')
plt.title('Opinion Over Time')
plt.grid(True)

output_file = os.path.join(output_folder, 'opinion_over_time.svg')
plt.savefig(output_file)

plt.show()


# PLOT: velocities over time

time_indices = np.arange(0, steps, 1000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for t in time_indices:
    ax.scatter(x[t, :, 0], x[t, :, 1], t, color='b', marker='o')
    ax.quiver(x[t, :, 0], x[t, :, 1], t, v[t, :, 0], v[t, :, 1], 0, color='r', length=6, linewidth=1)
    ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('steps')
ax.set_title('Velocities Over Time')
plt.grid(True)

output_file = os.path.join(output_folder, 'velocities_over_time.svg')
plt.savefig(output_file)

plt.show()


# PLOT: positions over time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for t in time_indices:
    ax.scatter(x[t, :, 0], x[t, :, 1], t, color='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('steps')
ax.set_title('Position Over Time')
plt.grid(True)

output_file = os.path.join(output_folder, 'positions_over_time.svg')
plt.savefig(output_file)

plt.show()