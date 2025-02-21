using PyPlot, LinearAlgebra

# Problem data
n_l = 20
n_f = 50
n_u = 50
ns = [n_l, n_f, n_u]
n = sum(ns)

t_final = 100.0
dt = 1.0e-2
steps = Int(floor(t_final / dt))

x = zeros(n, 2, steps)
v = zeros(n, 2, steps)
w = zeros(n, steps)

alpha = 1.0
beta = 0.5
c_att = 50.0
l_att = 1.0
c_rep = 60.0
l_rep = 0.5

function morse_potential(r, c_rep, c_att, l_rep, l_att)
    return -(c_rep / l_rep) * exp.(-r ./ l_rep) + (c_att / l_att) * exp.(-r ./ l_att)
end

nabla_u = r -> morse_potential(r, c_rep, c_att, l_rep, l_att)

r_x = 1.0
r_w = 0.5
gammas_red = [1, 1, 0]
gammas_blue = [1, 1, 0]
tau_blue_l = 0.1
tau_red_f = 0.01

# Initial conditions
Random.seed!(1234)
x[:, :, 1] = v[:, :, 1] = [-1 .+ 2 .* rand(n, 2)]
w[:, 1] = [-1 .+ 2 .* rand(n_l); -1 .+ 2 .* rand(n_f); zeros(n_u)]

function ode_system(x_step, v_step, w_step, ns, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)
    n = sum(ns)
    xs = [x_step[1:ns[1], :], x_step[ns[1]+1:ns[1]+ns[2], :], x_step[ns[1]+ns[2]+1:n, :]]
    vs = [v_step[1:ns[1], :], v_step[ns[1]+1:ns[1]+ns[2], :], v_step[ns[1]+ns[2]+1:n, :]]
    ws = [w_step[1:ns[1]], w_step[ns[1]+1:ns[1]+ns[2]], w_step[ns[1]+ns[2]+1:n]]

    v_blue = [1, 1]
    v_red = [-1, -1]
    w_blue, w_red = 1, -1

    dvs = [zeros(size(v)) for v in vs]
    
    for i in 1:3
        term_1 = (alpha .- beta .* sum(vs[i] .^ 2, dims=2)) .* vs[i]
        term_2 = sum(1 / ns[j] * potential_sum(xs[i], xs[j], nabla_u) for j in 1:3)
        term_3 = gammas_blue[i] .* velocity_alignment(vs[i], ws[i], r_w, v_blue, w_blue) .+ 
                 gammas_red[i] .* velocity_alignment(vs[i], ws[i], r_w, v_red, w_red)
        dvs[i] = term_1 .+ term_2 .+ term_3
    end
    
    dx = vcat(vs...)
    dv = vcat(dvs...)
    dw = zeros(n)
    return dx, dv, dw
end

function potential_sum(x_step, y_step, nabla_u)
    z = permutedims(x_step, [2, 1]) .- y_step'
    d_ij = sqrt.(sum(z .^ 2, dims=3))
    d_ij[d_ij .== 0] .= 1
    return -nabla_u(d_ij) .* z ./ d_ij
end

function velocity_alignment(v, wi, r_w, vt, wt)
    return (abs.(wi .- wt) .< r_w) .* (vt .- v)
end

for i in 3:steps
    dx_1, dv_1, dw_1 = ode_system(x[:, :, i-1], v[:, :, i-1], w[:, i-1], ns, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)
    dx_2, dv_2, dw_2 = ode_system(x[:, :, i], v[:, :, i], w[:, i], ns, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)
    x[:, :, i] = x[:, :, i-1] + (dt / 2) * (3 * dx_2 - dx_1)
    v[:, :, i] = v[:, :, i-1] + (dt / 2) * (3 * dv_2 - dv_1)
    w[:, i] = w[:, i-1] + (dt / 2) * (3 * dw_2 - dw_1)
end

# Plot final velocity configuration
figure()
plot(x[1:n_l, 1, end], x[1:n_l, 2, end], "bo")
plot(x[n_l+1:n_l+n_f, 1, end], x[n_l+1:n_l+n_f, 2, end], "ro")
plot(x[n_l+n_f+1:n, 1, end], x[n_l+n_f+1:n, 2, end], "ko")
quiver(x[:, 1, end], x[:, 2, end], v[:, 1, end], v[:, 2, end], color="r")
xlabel("X")
ylabel("Y")
grid(true)
title("Velocities at the final time")
axis("equal")
show()
