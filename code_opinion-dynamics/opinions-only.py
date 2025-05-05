import os
import numpy as np
import matplotlib.pyplot as plt



# -----------------------------
# FUNCTIONS
# -----------------------------

def psi_sum(wk, wm, r_w):
    w_diff = wm - wk[:, np.newaxis]

    bool_psi = np.abs(w_diff) < r_w
    return np.sum(w_diff * bool_psi, axis=1)


def plot_opinions(ax, data, final_avg_lf, steps, dom, title, color, do_log=False):   
    ax.set_title(title, fontweight='bold', fontsize=16)

    if do_log:
        ax.set_xscale('log')
    ax.set_xlim([1,steps])
    ax.set_xlabel("timestep")
    ax.set_ylim([-dom, dom])
    ax.set_ylabel("opinion")

    ax.plot(range(1, steps+1), data, color=color, linewidth=2)
    ax.plot(range(1, steps+1), [final_avg_lf] * steps, '--', color=colors['gray'], linewidth=2)



# -----------------------------
# SET PARAMETERS
# -----------------------------

t_final = 500                            # final time
dt = 0.1                                # timestep
steps = int(np.floor(t_final / dt))     # number of steps

n_l = 5                                 # number of leaders. they prefer A
n_f = 6                                 # number of followers. they prefer B
n_u = 10                                # number of uninformed. no preference

dom = 1
w_blue = 1                              # reference opinions
w_red = -1

# Example 1
k_ll, k_lf, k_lu = 1, 1, 0
k_fl, k_ff, k_fu = 1, 1, 0
k_ul, k_uf, k_uu = 0, 0, 1

# # Example 2
# k_ll, k_lf, k_lu = 1, 1, 1
# k_fl, k_ff, k_fu = 1, 1, 1
# k_ul, k_uf, k_uu = 0, 0, 1

# # Example 3
# k_ll, k_lf, k_lu = 1, 1, 1
# k_fl, k_ff, k_fu = 1, 1, 1
# k_ul, k_uf, k_uu = 1, 1, 1

# # Example 4
# k_ll, k_lf, k_lu = 1, 1, 0
# k_fl, k_ff, k_fu = 1, 1, 1
# k_ul, k_uf, k_uu = 0, 0, 0

r_w = 1

tau_blue = 0.1                            # conviction
tau_red = 0.01

sigma = 0                               # noise parameter



# -----------------------------
# SIMULATION
# -----------------------------

w_l = np.zeros((steps, n_l))            # opinion vectors
w_f = np.zeros((steps, n_f))
w_u = np.zeros((steps, n_u))

np.random.seed(1234)

w_l[0] = np.random.uniform(-dom, dom, n_l)
w_f[0] = np.random.uniform(-dom, dom, n_f)
w_u[0] = np.random.uniform(-dom, dom, n_u)
# w_l[0] = np.random.uniform(0, 1, n_l)
# w_f[0] = np.random.uniform(-1, 0, n_f)
# w_u[0] = np.random.uniform(-0.5, 0.5, n_u)

for k in range(steps - 1):
    # opinions change after these interactions
    psi_ll = psi_sum(w_l[k], w_l[k], r_w)
    psi_ff = psi_sum(w_f[k], w_f[k], r_w)
    psi_uu = psi_sum(w_u[k], w_u[k], r_w)
    psi_lf = psi_sum(w_l[k], w_f[k], r_w)
    psi_lu = psi_sum(w_l[k], w_u[k], r_w)
    psi_fl = psi_sum(w_f[k], w_l[k], r_w)
    psi_fu = psi_sum(w_f[k], w_u[k], r_w)
    psi_ul = psi_sum(w_u[k], w_l[k], r_w)
    psi_uf = psi_sum(w_u[k], w_f[k], r_w)
    
    # integration step
    dw_l = k_ll*psi_ll/n_l + k_lf*psi_lf/n_f + k_lu*psi_lu/n_u + tau_blue * (dom*w_blue - w_l[k])
    dw_f = k_fl*psi_fl/n_l + k_ff*psi_ff/n_f + k_fu*psi_fu/n_u + tau_red *  (dom*w_red  - w_f[k])
    dw_u = k_ul*psi_ul/n_l + k_uf*psi_uf/n_f + k_uu*psi_uu/n_u

    w_l[k+1] = w_l[k] + dt * dw_l
    w_f[k+1] = w_f[k] + dt * dw_f
    w_u[k+1] = w_u[k] + dt * dw_u



# -----------------------------
# PLOTS
# -----------------------------

output_folder = 'figures/opinions-only'
os.makedirs(output_folder, exist_ok=True)

colors = {
    'gray': (0.7, 0.7, 0.7),
    'red': (0, 0.4470, 0.7410),
    'blue': (0.8500, 0.3250, 0.0980)
}


# Plot: all opinions

fig, axes = plt.subplots(1, 3, figsize=(8, 5))

final_avg_lf = (np.sum(w_l[-1]) + np.sum(w_f[-1])) / (n_l + n_f)
print(final_avg_lf)

plot_opinions(axes[0], w_l, final_avg_lf, steps, dom, "Leaders", colors['red'], do_log=True)
plot_opinions(axes[1], w_f, final_avg_lf, steps, dom, "Followers", colors['blue'], do_log=True)
plot_opinions(axes[2], w_u, final_avg_lf, steps, dom, "Uninformed", 'k', do_log=True)
plt.tight_layout()

output_file = os.path.join(output_folder, 'example-1.svg')
plt.savefig(output_file)

# plt.show()


# Plot: mean opinion

# mean_w_l = np.mean(w_l, axis=1)
# mean_w_f = np.mean(w_f, axis=1)
# mean_w_u = np.mean(w_u, axis=1)

# fig, ax = plt.subplots(figsize=(5, 5))

# ax.plot(range(1, steps+1), mean_w_l, label="Leaders", color=colors['red'], linewidth=2)
# ax.plot(range(1, steps+1), mean_w_f, label="Followers", color=colors['blue'], linewidth=2)
# ax.plot(range(1, steps+1), mean_w_u, label="Uninformed", color='k', linewidth=2)

# ax.set_title("Mean opinion", fontweight='bold', fontsize=16)
# ax.set_xlim([1, steps])
# ax.set_ylim([-dom, dom])
# ax.set_xlabel("timestep")
# ax.set_ylabel("opinion")

# ax.legend()
# plt.tight_layout()

# output_file = os.path.join(output_folder, 'mean_opinions.svg')
# plt.savefig(output_file)

# plt.show()