clear all;
close all;

%% -------------------READ ME-----------------------

% The only place where I multiply by the factor 1/n is in the sumphi term,
% which is only found in the equation regarding w

% All the parameters used in the model are stated and described right below

%----------------------------------------------%

% Other possible parameters. Changing tau_red and tau_blue gives very rich dynamics

% alpha = 1; beta = 0.5;                                % friction parameters
% c_att = 100; l_att = 1; c_rep = 60; l_rep = 0.5;      % clusters moving in opposite direction (rotating)
% c_att = 100; l_att = 1; c_rep = 50; l_rep = 0.5;      % circle formation
% c_att = 100; l_att = 1; c_rep = 40; l_rep = 6;        % compact clusters moving in the same direction (rotating)
% c_att = 100; l_att = 1; c_rep = 50; l_rep = 1.2;      % compact clusters moving in the same direction (rotating)
% c_att = 50; l_att = 1; c_rep = 60; l_rep = 0.5;       % rotating mill
% c_att = 1; l_att = 1; c_rep = 2; l_rep = 0.5;
% c_att = 100; l_att = 1; c_rep = 60; l_rep = 0.7;      % (small) clusters moving in opposite direction (rotating)


%% Problem data

t_final = 100;                      % final time
dt = 1.0e-2;                        % timestep
steps = floor(t_final/dt);          % number of time steps

n = 100;                            % total number of individuals
x = zeros(n, 2, steps);
v = zeros(n, 2, steps);
w = zeros(n, steps);

alpha = 1; beta = 5;
r_x = 0.5;                          % spatial radius for alignment in velocity
r_w = 1;                            % preference radius for alignment il velocity
c_att = 100; l_att = 1.2;
c_rep = 350; l_rep = 0.8;
tau_red = 0.1;                      % strength of preference towards -1, red target
tau_blue = 0.1;                     % strength of preference towards 1, blue target

nabla_u = @(r) morse_potential(r, c_rep, c_att, l_rep, l_att);

% Initial conditions

rng(1234);

x_0 = -1 + 2*rand(n,2);     % n*2, initial coordinates
v_0 = -1 + 2*rand(n,2);
w_0 = -1 + 2*rand(n,1);     % n*1, initial values


%% Integration step

% Preparation: compute the first solution using Euler. Use it to compute the second system.

x(:,:,1) = x_0;
v(:,:,1) = v_0;
w(:,1) = w_0;

[dx_1, dv_1, dw_1] = ode_system(x(:,:,1), v(:,:,1), w(:,1), n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue);

x(:,:,2) = x(:,:,1) + dt*dx_1;
v(:,:,2) = v(:,:,1) + dt*dv_1;
w(:,2) = w(:,1) + dt*dw_1;

[dx_2, dv_2, dw_2] = ode_system(x(:,:,2), v(:,:,2), w(:,2), n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue);

% Adams-Bashforth 2-step method

for i = 3:steps
    x(:,:,i) = x(:,:,i-1) + (dt/2) * (3*dx_2 - dx_1);
    v(:,:,i) = v(:,:,i-1) + (dt/2) * (3*dv_2 - dv_1);
    w(:,i) = w(:,i-1) + (dt/2) * (3*dw_2 - dw_1);
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')       % [VVM] also v?
        error('NaN or Inf encountered at time step %d', i);
    end

    dx_1 = dx_2;
    dv_1 = dv_2;
    dw_1 = dw_2;
    
    [dx_2, dv_2, dw_2] = ode_system(x(:,:,i), v(:,:,i), w(:,i), n, alpha, beta,nabla_u, r_x, r_w, tau_red, tau_blue);
end


%% Plot final velocity configuration

% Create the folder if it doesn't exist

output_folder = 'Figures_onePopNoMill';      % Folder to save the figures
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Plot
figure()
    plot(x(1:n,1,end), x(1:n,2,end), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b','LineWidth', 1);  % Plot positions as circles
    hold on;

    quiver(x(1:n,1,end), x(1:n,2,end), v(1:n,1,end), v(1:n,2,end), 'r','LineWidth', 1); % Plot velocities as arrows

    xlabel('X');
    ylabel('Y');

    grid on;
    title('Velocities at the final time');

    hold off;
    axis equal;
    set(gca,'FontSize',16)
    % filename1 = fullfile(output_folder, ['FinalVelocityNoMillNoPreference_NL', num2str(n),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_tau_blue_',num2str(tau_blue),'_tau_red_',num2str(tau_red), '.fig']);
    % saveas(gcf, filename1);


%% Plot for several times

% Parameters
time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% Prepare figure
figure;

    % Plot positions as circles in 3D space
    for tIdx = 1:length(time)
        t = time(tIdx); % Current time index
        plot3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
        hold on;
    end

    % Plot velocities as arrows in 3D space
    for tIdx = 1:length(time)
        t = time(tIdx); % Current time index
        quiver3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), ...
                v(1:n,1,t), v(1:n,2,t), zeros(n,1), 'r');
    end

    % Label axes
    xlabel('X');
    ylabel('Y');
    zlabel('Time');

    % Adjust plot appearance
    grid on;
    %axis equal;
    title('Velocities Over Time');
    view(3); % Use a 3D perspective
    hold off;
    set(gca,'FontSize',16)

    % filename2 = fullfile(output_folder, ['VelocityOverTime_NL', num2str(n),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_tau_blue_',num2str(tau_blue),'_tau_red_',num2str(tau_red), '.fig']);
    % saveas(gcf, filename2);


%% Plot of the trajectories for some times

% Parameters
time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% Prepare figure
figure;

    % Plot positions as circles in 3D space
    for tIdx = 1:length(time)
        t = time(tIdx); % Current time index
        plot3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
        hold on;
    end

    % Plot velocities as arrows in 3D space
    for tIdx = 1:length(time)
        t = time(tIdx); % Current time index
        quiver3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), ...
                x(1:n,1,t), x(1:n,2,t), zeros(n,1), 'r');
    end

    % Label axes
    xlabel('X');
    ylabel('Y');
    zlabel('Time');

    % Adjust plot appearance
    grid on;
    %axis equal;
    title('Position Over Time');
    view(3); % Use a 3D perspective
    hold off;
    set(gca,'FontSize',16)

    % filename3 = fullfile(output_folder, ['PositionOverTime_NL', num2str(n),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_tau_blue_',num2str(tau_blue),'_tau_red_',num2str(tau_red), '.fig']);
    % saveas(gcf, filename3);


%% Plot opinion over time

figure();
    plot(1:steps,w, 'k','LineWidth', 1);
    xlabel('Time');
    ylabel('Opinion');
    title('Opinion Over Time');
    set(gca,'FontSize',16)
    % 
    % filename4 = fullfile(output_folder, ['OpinionOverTimeNoMillNoPreference', '.fig']);
    % saveas(gcf, filename4);



%% -------------------PROBLEM-RELATED FUNCTIONS-------------------

function morse_potential = morse_potential(r, c_rep, c_att, l_rep, l_att)
    morse_potential = -(c_rep/l_rep)*exp(-r./l_rep) + (c_att/l_att)*exp(-r./l_att);
end


function [dx,dv,dw] = ode_system(x_step, v_step, w_step, n, alpha, beta, nabla_u, r_x, r_w, tau_red, tau_blue)
    % x_step, v_step, w_step are x,v,w at a given step.
    % In this case, we are considering tau = gamma. [VVM]

    v_blue = 1;     % vector (1,1)
    v_red = -1;     % vector (-1,-1)

    w_blue = 1;
    w_red = -1;

    % x_blue = 1;   
    % x_red = -1;

    term_1 = (alpha - beta*sum(v_step.^2,2)) .* v_step;
    term_2 = -1/n * potential_sum(x_step,nabla_u);

    term_3_red = tau_red * velocity_alignment(v_step,v_red,w_step,w_red,r_w);
    term_3_blue = tau_blue * velocity_alignment(v_step,v_blue,w_step,w_blue,r_w);

    phi = opinion_alignment_sum(x_step, w_step, n, r_x, r_w);

    % termVelAlign = 1/n * cucker_smale_aligment(x_step,w_step,v_step,n,r_x,r_w);
    % dv = term_1 + term_2 + term_3_red + term_3_blue + 0*termVelAlign;

    dx = v_step;
    dv = term_1 + term_2 + term_3_red + term_3_blue;
    dw = phi / n + tau_red * (w_red-w_step) + tau_blue * (w_blue-w_step); 
end


function forces = potential_sum(x_step, nabla_u)
    xi = x_step;
    xj = reshape(x_step', 1, size(x_step, 2), []); 
    z = xi - xj;
    d_ij = sqrt(sum(z.^2, 2));
    d_ij(d_ij == 0) = 1;            % vector will be zero and result too either way
    forces = nabla_u(d_ij) .* z ./ d_ij;
    forces = squeeze(sum(forces, 3));
end


function alignment = velocity_alignment(v_step, v_ref, w_step, w_ref, r_w)
    bool_aligned = abs(w_step - w_ref) < r_w;
    alignment = bool_aligned .* (v_ref-v_step);
end


function phi = opinion_alignment_sum(x_step, w_step, n, r_x, r_w)
    wi = repmat(w_step, 1, n);
    wj = repmat(w_step', n, 1);    
    xi = reshape(x_step, [n, 1, size(x_step, 2)]);
    xj = reshape(x_step, [1, n, size(x_step, 2)]);

    z  = xi - xj;
    d_ij = sqrt(sum((z).^2, 3));   
    bool_phi = (abs(wi - wj) < r_w) & (d_ij < r_x);
    
    phi = sum(bool_phi .* (wj - wi), 2);
end


% function U = cucker_smale_aligment(x_step,w_step,v_step,n,r_x,r_w) % this is velocity alignment as in Cucker-Smale
%     wi = repmat(w_step, 1, n);
%     wj = repmat(w_step', n, 1);    
%     xj = reshape(x_step, [n, 1, size(x_step, 2)]);
%     xi = reshape(x_step, [1, n, size(x_step, 2)]);
%     vj = reshape(v_step, [n, 1, size(v_step, 2)]);
%     vi = reshape(v_step, [1, n, size(v_step, 2)]);
%     incXij  = xi - xj;
%     distWij = sqrt(sum((incXij).^2, 3));   
%     isClosed = (abs(wi - wj) < r_w) & (distWij < r_x);  
%     velDiff = vj - vi; % Pairwise velocity difference, 150x150x2
%     U = squeeze(sum(isClosed .* velDiff, 2));
%     %U      = sum(isClosed .* (vj - vi), 2);
% end


% function alignment = alternative_opinion_alignment(x_step,w1,r_w,xt,wt) 
%     bool_aligned = abs(w1 - wt) < r_w;
%     alignment = bool_aligned.*(xt-x_step);
% end





