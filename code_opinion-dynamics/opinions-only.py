import os
import numpy as np
import matplotlib.pyplot as plt



def psi_sum(wk, wm, n_k, n_m, r_w):
    psi = np.zeros(n_k)
    for i in range(n_k):
        for j in range(n_m):
            if abs(wm[j]-wk[i]) < r_w:
                psi[i] += (wm[j]-wk[i])
    return psi


def plot_opinions(ax, data, initial_avg_k, final_avg_total, steps, dom, title, color, do_log=False):   
    ax.set_title(title, fontweight='bold', fontsize=16)

    if do_log:
        ax.set_xscale('log')
    ax.set_xlim([1,steps])
    ax.set_xlabel("timesteps")
    ax.set_ylim([-dom, dom])
    ax.set_ylabel("w_i")

    ax.plot(range(steps), data, color=color, linewidth=2)
    ax.plot(range(steps), [initial_avg_k] * steps, '--', color=color,linewidth=2)
    ax.plot(range(steps), [final_avg_total] * steps, color=colors['gray'], linewidth=2)



# SET PARAMETERS

t_final = 20                            # final time
dt = 1.0e-1                             # timestep
steps = int(np.floor(t_final / dt))     # number of steps
dom = 1                                 # computational domain (w in [-dom,dom])

n_l = 5                                 # number of leaders. they prefer A
n_f = 6                                 # number of followers. they prefer B
n_u = 10                                # number of uninformed. no preference

w_blue = 1                              # reference opinions
w_red = -1

# # Example 1
# p_ll, p_lf, p_lu = 1, 1, 0
# p_fl, p_ff, p_fu = 1, 1, 0
# p_ul, p_uf, p_uu = 0, 0, 1

# # Example 2
# p_ll, p_lf, p_lu = 1, 1, 1
# p_fl, p_ff, p_fu = 1, 1, 1
# p_ul, p_uf, p_uu = 0, 0, 1

# # Example 3
# p_ll, p_lf, p_lu = 1, 1, 1
# p_fl, p_ff, p_fu = 1, 1, 1
# p_ul, p_uf, p_uu = 1, 1, 1

# Example 4
p_ll, p_lf, p_lu = 1, 1, 0
p_fl, p_ff, p_fu = 1, 1, 0
p_ul, p_uf, p_uu = 0, 0, 0

r_w = 1

tau_blue = 1                            # conviction
tau_red = 0.1

sigma = 0                               # noise parameter

w_l = np.zeros((steps, n_l))            # opinion vectors
w_f = np.zeros((steps, n_f))
w_u = np.zeros((steps, n_u))


# INITIALIZE POSITIONS

np.random.seed(1234)

w_l[0] = -dom + (2*dom) * np.random.rand(n_l)
w_f[0] = -dom + (2*dom) * np.random.rand(n_f)
w_u[0] = -dom + (2*dom) * np.random.rand(n_u)

for k in range(steps - 1):
    # opinions change after these interactions
    psi_ll = psi_sum(w_l[k], w_l[k], n_l, n_l, r_w)
    psi_ff = psi_sum(w_f[k], w_f[k], n_f, n_f, r_w)
    psi_uu = psi_sum(w_u[k], w_u[k], n_u, n_u, r_w)
    psi_lf = psi_sum(w_l[k], w_f[k], n_l, n_f, r_w)
    psi_lu = psi_sum(w_l[k], w_u[k], n_l, n_u, r_w)
    psi_fl = psi_sum(w_f[k], w_l[k], n_f, n_l, r_w)
    psi_fu = psi_sum(w_f[k], w_u[k], n_f, n_u, r_w)
    psi_ul = psi_sum(w_u[k], w_l[k], n_u, n_l, r_w)
    psi_uf = psi_sum(w_u[k], w_f[k], n_u, n_f, r_w)
    
    # integration step
    dw_l = p_ll*psi_ll/n_l + p_lf*psi_lf/n_f + p_lu*psi_lu/n_u + tau_blue * (dom*w_blue - w_l[k])
    dw_f = p_fl*psi_fl/n_l + p_ff*psi_ff/n_f + p_fu*psi_fu/n_u + tau_red * (dom*w_red - w_f[k])
    dw_u = p_ul*psi_ul/n_l + p_uf*psi_uf/n_f + p_uu*psi_uu/n_u

    w_l[k+1] = w_l[k] + dt * dw_l
    w_f[k+1] = w_f[k] + dt * dw_f
    w_u[k+1] = w_u[k] + dt * dw_u


# PLOT THE RESULTS

final_avg_total = (np.sum(w_l[-1]) + np.sum(w_f[-1]) + np.sum(w_u[-1])) / (n_l + n_f + n_u)

colors = {
    'gray': (0.7, 0.7, 0.7),
    'red': (0, 0.4470, 0.7410),
    'blue': (0.8500, 0.3250, 0.0980)
}


output_folder = 'figures/opinions-only'
os.makedirs(output_folder, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(8, 5))

plot_opinions(axes[0], w_l, np.mean(w_l[0]), final_avg_total, steps, dom, "Leaders", colors['red'])
plot_opinions(axes[1], w_f, np.mean(w_f[0]), final_avg_total, steps, dom, "Followers", colors['blue'])
plot_opinions(axes[2], w_u, np.mean(w_u[0]), final_avg_total, steps, dom, "Uninformed", 'k')
plt.tight_layout()

output_file = os.path.join(output_folder, 'opinions_only.svg')
plt.savefig(output_file)

plt.show()
