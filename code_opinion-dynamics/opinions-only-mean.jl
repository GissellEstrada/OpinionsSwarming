using Random
using Statistics
using PyPlot

# Define the psi_sum function
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

# -----------------------------
# SET PARAMETERS
# -----------------------------
t_final = 20.0                      # final time
dt = 0.1                            # timestep
steps = Int(floor(t_final / dt))    # number of time steps
dom = 1.0                           # domain: opinions in [-dom, dom]

n_l = 5  
n_f = 6  
n_u = 10 

w_blue = 1.0  
w_red  = -1.0 

# Example 1:
# (k_ll, k_lf, k_lu) = (1.0, 1.0, 0.0)
# (k_fl, k_ff, k_fu) = (1.0, 1.0, 0.0)
# (k_ul, k_uf, k_uu) = (0.0, 0.0, 1.0)

# Example 2:
# (k_ll, k_lf, k_lu) = (1.0, 1.0, 1.0)
# (k_fl, k_ff, k_fu) = (1.0, 1.0, 1.0)
# (k_ul, k_uf, k_uu) = (0.0, 0.0, 1.0)

# Example 3:
# (k_ll, k_lf, k_lu) = (1.0, 1.0, 1.0)
# (k_fl, k_ff, k_fu) = (1.0, 1.0, 1.0)
# (k_ul, k_uf, k_uu) = (1.0, 1.0, 1.0)

# Parameter examples (Example 4)
(k_ll, k_lf, k_lu) = (1.0, 1.0, 0.0)
(k_fl, k_ff, k_fu) = (1.0, 1.0, 1.0)
(k_ul, k_uf, k_uu) = (0.0, 0.0, 0.0)

r_w = 1.0

tau_blue = 1.0
tau_red  = 0.1

sigma = 0.0

# -----------------------------
# SIMULATION FUNCTION
# -----------------------------
function simulate()

    w_l = zeros(steps, n_l)
    w_f = zeros(steps, n_f)
    w_u = zeros(steps, n_u)
    
    w_l[1, :] = -dom .+ (2 * dom) .* rand(n_l)
    w_f[1, :] = -dom .+ (2 * dom) .* rand(n_f)
    w_u[1, :] = -dom .+ (2 * dom) .* rand(n_u)
    
    for k in 1:(steps - 1)
        psi_ll = psi_sum(w_l[k, :], w_l[k, :], n_l, n_l, r_w)
        psi_ff = psi_sum(w_f[k, :], w_f[k, :], n_f, n_f, r_w)
        psi_uu = psi_sum(w_u[k, :], w_u[k, :], n_u, n_u, r_w)
        psi_lf = psi_sum(w_l[k, :], w_f[k, :], n_l, n_f, r_w)
        psi_lu = psi_sum(w_l[k, :], w_u[k, :], n_l, n_u, r_w)
        psi_fl = psi_sum(w_f[k, :], w_l[k, :], n_f, n_l, r_w)
        psi_fu = psi_sum(w_f[k, :], w_u[k, :], n_f, n_u, r_w)
        psi_ul = psi_sum(w_u[k, :], w_l[k, :], n_u, n_l, r_w)
        psi_uf = psi_sum(w_u[k, :], w_f[k, :], n_u, n_f, r_w)
        
        # Integration step
        dw_l = k_ll * psi_ll / n_l .+ k_lf * psi_lf / n_f .+ k_lu * psi_lu / n_u .+ tau_blue * (dom * w_blue .- w_l[k, :])
        dw_f = k_fl * psi_fl / n_l .+ k_ff * psi_ff / n_f .+ k_fu * psi_fu / n_u .+ tau_red  * (dom * w_red  .- w_f[k, :])
        dw_u = k_ul * psi_ul / n_l .+ k_uf * psi_uf / n_f .+ k_uu * psi_uu / n_u

        w_l[k + 1, :] = w_l[k, :] .+ dt .* dw_l
        w_f[k + 1, :] = w_f[k, :] .+ dt .* dw_f
        w_u[k + 1, :] = w_u[k, :] .+ dt .* dw_u
    end
    return w_l, w_f, w_u
end

# -----------------------------
# ENSEMBLE AVERAGING OVER 100 RUNS
# -----------------------------
runs = 5000

ensemble_mean_leaders = zeros(steps)
ensemble_mean_followers = zeros(steps)
ensemble_mean_uninformed = zeros(steps)

Random.seed!(1234)

for run in 1:runs
    w_l, w_f, w_u = simulate()
    
    mean_l = [mean(w_l[k, :]) for k in 1:steps]
    mean_f = [mean(w_f[k, :]) for k in 1:steps]
    mean_u = [mean(w_u[k, :]) for k in 1:steps]
    
    ensemble_mean_leaders .+= mean_l
    ensemble_mean_followers .+= mean_f
    ensemble_mean_uninformed .+= mean_u
end

ensemble_mean_leaders ./= runs
ensemble_mean_followers ./= runs
ensemble_mean_uninformed ./= runs

# -----------------------------
# PLOTTING THE ENSEMBLE MEAN OPINIONS
# -----------------------------

colors = Dict(
    "red"  => (0, 0.4470, 0.7410),
    "blue" => (0.8500, 0.3250, 0.0980)
)

fig, ax = subplots(figsize=(5, 5))
ax[:set_title]("Ensemble mean", fontweight="bold", fontsize=16)
ax[:set_xlabel]("Time step")
ax[:set_ylabel]("Mean Opinion")
ax[:set_xlim](1, steps)
ax[:set_ylim](-dom, dom)

ax[:plot](1:steps, ensemble_mean_leaders, label="Leaders", color=colors["red"], linewidth=2)
ax[:plot](1:steps, ensemble_mean_followers, label="Followers", color=colors["blue"], linewidth=2)
ax[:plot](1:steps, ensemble_mean_uninformed, label="Uninformed", color="black", linewidth=2)

ax[:legend]()
tight_layout()

# Create the output directory if needed
mkpath(joinpath("figures", "opinions-only"))
output_file = joinpath("figures", "opinions-only", "ensemble_mean_opinions-$(runs).svg")
savefig(output_file)
display(fig)
