using Random
using PyPlot
using Statistics



# FUNCTIONS

function psi_sum(wk, wm, n_k, n_m, r_w)
    psi = zeros(n_k)
    for i in 1:n_k
        for j in 1:n_m
            if abs(wm[j] - wk[i]) < r_w
                psi[i] += wm[j] - wk[i]
            end
        end
    end
    return psi
end

colors = Dict(
    "gray" => (0.7, 0.7, 0.7),
    "red"  => (0, 0.4470, 0.7410),
    "blue" => (0.8500, 0.3250, 0.0980)
)

function plot_opinions(ax, data, initial_avg_k, final_avg_total, steps, dom, title_str, color; do_log=false)
    ax[:set_title](title_str, fontweight="bold", fontsize=16)
    if do_log
        ax[:set_xscale]("log")
    end
    ax[:set_xlim](1, steps)
    ax[:set_xlabel]("timesteps")
    ax[:set_ylim](-dom, dom)
    ax[:set_ylabel]("w_i")
    ax[:plot](1:steps, data, color=color, linewidth=2)
    ax[:plot](1:steps, fill(initial_avg_k, steps), "--", color=color, linewidth=2)
    ax[:plot](1:steps, fill(final_avg_total, steps), color=colors["gray"], linewidth=2)
end



# PARAMETERS

t_final = 20.0                      # final time
dt = 1.0e-1                         # timestep
steps = Int(floor(t_final/dt))      # number of steps
dom = 1.0                           # computational domain (w in [-dom,dom])

n_l = 5                             # number of leaders (they prefer A)
n_f = 6                             # number of followers (they prefer B)
n_u = 10                            # number of uninformed (no preference)

w_blue = 1.0                        # reference opinion for blue
w_red  = -1.0                       # reference opinion for red

# Example 1:
# (k_ll, k_lf, k_lu) = (1.0, 1.0, 0.0)
# (k_fl, k_ff, k_fu) = (1.0, 1.0, 0.0)
# (k_ul, k_uf, k_uu) = (0.0, 0.0, 1.0)

# Example 2:
# (k_ll, k_lf, k_lu) = (1.0, 1.0, 1.0)
# (k_fl, k_ff, k_fu) = (1.0, 1.0, 1.0)
# (k_ul, k_uf, k_uu) = (0.0, 0.0, 1.0)

# # Example 3:
# (k_ll, k_lf, k_lu) = (1.0, 1.0, 1.0)
# (k_fl, k_ff, k_fu) = (1.0, 1.0, 1.0)
# (k_ul, k_uf, k_uu) = (1.0, 1.0, 1.0)

# Example 4
(k_ll, k_lf, k_lu) = (1.0, 1.0, 0.0)
(k_fl, k_ff, k_fu) = (1.0, 1.0, 1.0)
(k_ul, k_uf, k_uu) = (0.0, 0.0, 0.0)

r_w = 1

tau_blue = 1                 
tau_red  = 0.1



# INITIALIZE POSITIONS

w_l = zeros(steps, n_l)
w_f = zeros(steps, n_f)
w_u = zeros(steps, n_u)

Random.seed!(1234)
w_l[1, :] = -dom .+ (2*dom) .* rand(n_l)
w_f[1, :] = -dom .+ (2*dom) .* rand(n_f)
w_u[1, :] = -dom .+ (2*dom) .* rand(n_u)

for k in 1:(steps - 1)
    psi_ll = psi_sum(w_l[k,:], w_l[k,:], n_l, n_l, r_w)
    psi_ff = psi_sum(w_f[k,:], w_f[k,:], n_f, n_f, r_w)
    psi_uu = psi_sum(w_u[k,:], w_u[k,:], n_u, n_u, r_w)
    psi_lf = psi_sum(w_l[k,:], w_f[k,:], n_l, n_f, r_w)
    psi_lu = psi_sum(w_l[k,:], w_u[k,:], n_l, n_u, r_w)
    psi_fl = psi_sum(w_f[k,:], w_l[k,:], n_f, n_l, r_w)
    psi_fu = psi_sum(w_f[k,:], w_u[k,:], n_f, n_u, r_w)
    psi_ul = psi_sum(w_u[k,:], w_l[k,:], n_u, n_l, r_w)
    psi_uf = psi_sum(w_u[k,:], w_f[k,:], n_u, n_f, r_w)
    
    # INTEGRATION STEP
    dw_l = k_ll*psi_ll/n_l .+ k_lf*psi_lf/n_f .+ k_lu*psi_lu/n_u .+ tau_blue * (dom*w_blue .- w_l[k,:])
    dw_f = k_fl*psi_fl/n_l .+ k_ff*psi_ff/n_f .+ k_fu*psi_fu/n_u .+ tau_red  * (dom*w_red  .- w_f[k,:])
    dw_u = k_ul*psi_ul/n_l .+ k_uf*psi_uf/n_f .+ k_uu*psi_uu/n_u

    w_l[k+1,:] = w_l[k,:] .+ dt.*dw_l
    w_f[k+1,:] = w_f[k,:] .+ dt.*dw_f
    w_u[k+1,:] = w_u[k,:] .+ dt.*dw_u
end

# Compute the final average total opinion
final_avg_total = (sum(w_l[end,:]) + sum(w_f[end,:]) + sum(w_u[end,:])) / (n_l + n_f + n_u)




# PLOT THE RESULTS

# plot the opinions of all individuals

fig, axes = subplots(1, 3, figsize=(8, 5))

plot_opinions(axes[1], w_l, mean(w_l[1,:]), final_avg_total, steps, dom, "Leaders", colors["red"])
plot_opinions(axes[2], w_f, mean(w_f[1,:]), final_avg_total, steps, dom, "Followers", colors["blue"])
plot_opinions(axes[3], w_u, mean(w_u[1,:]), final_avg_total, steps, dom, "Uninformed", "k")

tight_layout()
# display(fig)

mkpath(joinpath("figures", "opinions-only"))
output_file = joinpath("figures", "opinions-only", "opinions_only-julia.svg")
savefig(output_file)


# plot the mean opinon of each group

mean_w_l = [mean(w_l[k, :]) for k in 1:steps]
mean_w_f = [mean(w_f[k, :]) for k in 1:steps]
mean_w_u = [mean(w_u[k, :]) for k in 1:steps]

fig2, ax2 = subplots(figsize=(5, 5))

ax2[:set_title]("Mean opinion", fontweight="bold", fontsize=16)
ax2[:set_xlabel]("timesteps")
ax2[:set_ylabel]("mean opinion")
ax2[:set_xlim](1, steps)
ax2[:set_ylim](-dom, dom)
# ax2[:set_xscale]("log")

ax2[:plot](1:steps, mean_w_l, label="Leaders", color=colors["red"], linewidth=2)
ax2[:plot](1:steps, mean_w_f, label="Followers", color=colors["blue"], linewidth=2)
ax2[:plot](1:steps, mean_w_u, label="Uninformed", color="black", linewidth=2)

# ax2[:legend]()
tight_layout()

output_file = joinpath("figures", "opinions-only", "mean_opinion.svg")
savefig(output_file)