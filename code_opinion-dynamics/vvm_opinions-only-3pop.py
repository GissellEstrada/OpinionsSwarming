def psi(wk, wm, n_k, n_m, r_w):
    Psi = np.zeros(n_k)
    for i in range(n_k):
        for j in range(n_m):
            if abs(wm[j]-wk[i]) < r_w:
                Psi[i] += (wm[j]-wk[i])     # j-i? [GER]
    return Psi


# steps, dom, and final_avg_total are not passed as arguments 
# nor defined inside the function. may be an issue

def plot_opinions(ax, data, initial_avg_k, title, color, do_log=False):   
    ax.set_title(title, fontweight='bold', fontsize=16)

    if do_log:
        ax.set_xscale('log')
    ax.set_xlim([1,steps])
    ax.set_xlabel("timesteps")
    ax.set_ylim([-dom, dom])
    ax.set_ylabel("w_i")

    ax.plot(range(steps), data.T, color=color, linewidth=2)
    ax.plot(range(steps), [initial_avg_k] * steps, '--', color=color,linewidth=2)
    ax.plot(range(steps), [final_avg_total] * steps, color=colors['gray'], linewidth=2)



import numpy as np
import matplotlib.pyplot as plt

# SET PARAMETERS

t_final = 20                             # final time
dt = 1.0e-1                         # timestep
steps = int(np.floor(t_final / dt))      # number of steps
dom = 1                             # computational domain (w in [-dom,dom])

n_l = 6                             # number of leaders. they prefer A
n_f = 6                             # number of followers. they prefer B
n_u = 10                            # number of uninformed. no preference

w_blue = 1                         # reference opinions
w_red = -1

p_ll, p_ff, p_uu = 1, 1, 0      # interaction strength
p_lf = p_fl = 1
p_lu = p_ul = 0
p_fu = p_uf = 0

tau_blue = 0.1                     # conviction
tau_red = 0.1

sigma = 0                       # noise parameter

wl = np.zeros((n_l, steps))     # opinion vectors
wf = np.zeros((n_f, steps))
wu = np.zeros((n_u, steps))


# INITIALIZE POSITIONS

np.random.seed(1234)

wl[:, 0] = (2*dom) * np.random.rand(n_l) - dom
wf[:, 0] = (2*dom) * np.random.rand(n_f) - dom
wu[:, 0] = (2*dom) * np.random.rand(n_u) - dom

r_w = 1

for k in range(steps - 1):
    # opinions change after these interactions
    Psi_ll = psi(wl[:,k], wl[:,k], n_l, n_l, r_w)
    Psi_ff = psi(wf[:,k], wf[:,k], n_f, n_f, r_w)
    Psi_uu = psi(wu[:,k], wu[:,k], n_u, n_u, r_w)
    Psi_lf = psi(wl[:,k], wf[:,k], n_l, n_f, r_w)
    Psi_lu = psi(wl[:,k], wu[:,k], n_l, n_u, r_w)
    Psi_fl = psi(wf[:,k], wl[:,k], n_f, n_l, r_w)
    Psi_fu = psi(wf[:,k], wu[:,k], n_f, n_u, r_w)
    Psi_ul = psi(wu[:,k], wl[:,k], n_u, n_l, r_w)
    Psi_uf = psi(wu[:,k], wf[:,k], n_u, n_f, r_w)
    
    # integration step
    wl[:,k+1] = (wl[:,k]
        + dt * (p_ll*Psi_ll/n_l + p_lf*Psi_lf/n_f + p_lu*Psi_lu/n_u)
        + tau_blue * (dom*w_blue - wl[:,k]))
    wf[:,k+1] = (wf[:,k]
        + dt * (p_fl*Psi_ff/n_f + p_ff*Psi_fl/n_l + p_fu*Psi_fu/n_u)
        + tau_red * (dom*w_red - wf[:,k]))
    wu[:,k+1] = (wu[:,k]
        + dt * (p_ul*Psi_uu/n_u + p_uf*Psi_ul/n_l + p_uu*Psi_uf/n_f))
    

# PLOT THE RESULTS

final_avg_total = (np.sum(wl[:,-1]) + np.sum(wf[:,-1]) + np.sum(wu[:,-1])) / (n_l + n_f + n_u)

colors = {
    'gray': (0.7, 0.7, 0.7),
    'red': (0, 0.4470, 0.7410),
    'blue': (0.8500, 0.3250, 0.0980)
}

fig, axes = plt.subplots(1, 3, figsize=(8, 5))

plot_opinions(axes[0], wl, np.mean(wl[:,0]), "Leaders", colors['red'])
plot_opinions(axes[1], wf, np.mean(wf[:,0]), "Followers", colors['blue'])
plot_opinions(axes[2], wu, np.mean(wu[:,0]), "Uninformed", 'k')

plt.tight_layout()
plt.show()
