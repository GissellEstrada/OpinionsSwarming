clear all;
% close all;


%% -------------------READ ME-----------------------

% The only place where I multiply by the factor 1/n is in the sumphi term,
% which is only found in the equation regarding w

%----------------------------------------------%

% All the parameters used in the model are stated and described right bellow

% Other possible parameters. Changing tau_red and tau_blue also gives very rich dynamics
% alpha = 1; beta = 5;
% c_att = 100; l_att = 1.2; c_rep = 350; l_rep = 0.8; 
% l_att = 2.2


%% Problem data

n_l = 20;
n_f = 50;
n_u = 50;                                   % we need to change n_u=2, 20, 80
ns = [n_l; n_f; n_u];
n = sum(ns);

t_final = 100;                              % final time
dt = 1.0e-2;                                % timestep
steps = floor(t_final/dt);                  % number of time steps

x = zeros(n, 2, steps);
v = zeros(n, 2, steps);
w = zeros(n, 1, steps);

alpha = 1; beta = 0.5;                      % friction parameters
c_att = 50; l_att = 1;
c_rep = 60; l_rep = 0.5;

nabla_u = @(r) morse_potential(r, c_rep, c_att, l_rep, l_att);

r_x = 1;    %5;                             % spatial radius for alignment in velocity 1
r_w = 0.5;  %1;                             % preference radius for alignment il velocity 0.5

% blue: preference towards 1 (leaders), red: preference towards -1 (followers)

gammas_red = [1; 1; 0];                     % alignment in velocity
gammas_blue = [1; 1; 0];
tau_blue_l = 0.1;                           % alignment in opinion
tau_red_f = 0.01;

% Initial conditions

rng(1234);

x(:,:,1) = [
    -1 + 2*rand(n_l,2);
    -1 + 2*rand(n_f,2);
    -1 + 2*rand(n_u,2)
    ];
v(:,:,1) = [
    -1 + 2*rand(n_l,2);
    -1 + 2*rand(n_f,2);
    -1 + 2*rand(n_u,2)
    ];
w(:,1) = [
    -1 + 2*rand(n_l,1);
    -1 + 2*rand(n_f,1);
    zeros(n_u,1)              % 0 of size w_u_0 [VVM we dont use w_u_0]
    ];    


%% Integration step

% Preparation: compute the first solution using Euler. Use it to compute the second system.

[dx_1, dv_1, dw_1] = ode_system(x(:,:,1), v(:,:,1), w(:,1), ns, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f);

x(:,:,2) = x(:,:,1) + dt * dx_1;
v(:,:,2) = v(:,:,1) + dt * dv_1;
w(:,2) = w(:,1) + dt * dw_1;

[dx_2, dv_2, dw_2] = ode_system(x(:,:,2), v(:,:,2), w(:,2), ns, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f);

% Adams-Bashforth 2-step method

for i = 3:steps
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * dx_2 - dx_1);  
    v(:,:,i) = v(:,:,i-1) + (dt / 2) * (3 * dv_2 - dv_1);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * dw_2 - dw_1);
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')   % [VVM] also v?
        error('NaN or Inf encountered at time step %d', i);
    end

    dx_1 = dx_2;
    dv_1 = dv_2;
    dw_1 = dw_2;

    [dx_2, dv_2, dw_2] = ode_system(x(:,:,i), v(:,:,i), w(:,i), ns, alpha, beta,nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f);
end


%% Save figure

output_folder = 'Figures_Mill_allcases';      % Folder to save the figures
% Create the folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end


%% Plot of the final velocity configuration

figure()
    plot(x(1:n_l,1,end), x(1:n_l,2,end), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');  % Plot positions as circles
    hold on;
    plot(x(n_l+1:n_l+n_f,1,end), x(n_l+1:n_l+n_f,2,end), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
    hold on;
    plot(x(n_l+n_f+1:n,1,end), x(n_l+n_f+1:n,2,end), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');

    quiver(x(1:n,1,end), x(1:n,2,end), v(1:n,1,end), v(1:n,2,end), 'r'); % Plot velocities as arrows

    xlabel('X');
    ylabel('Y');

    grid on;
    title('Velocities at the final time');

    hold off;
    axis equal; fontsize(gca, 15,'points');

    filename1 = fullfile(output_folder, ['3PopOp_WithUninformed_FLUInteraction_case2_NL', num2str(n_l),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_taur_',num2str(tau_blue_l),'_taub_',num2str(tau_red_f), '.fig']);
    saveas(gcf, filename1);


% %% Plot for several times

% % Parameters
% time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% % Prepare figure
% figure;

%     % Plot positions as circles in 3D space
%     for i_t = 1:length(time)
%         t = time(i_t); % Current time index
%         plot3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
%         hold on;
%     end

%     % Plot velocities as arrows in 3D space
%     for i_t = 1:length(time)
%         t = time(i_t); % Current time index
%         quiver3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), ...
%                 v(1:n,1,t), v(1:n,2,t), zeros(n,1), 'r');
%     end

%     % Label axes
%     xlabel('X');
%     ylabel('Y');
%     zlabel('Time');

%     % Adjust plot appearance
%     grid on;
%     %axis equal;
%     title('Velocities Over Time');
%     view(3); % Use a 3D perspective
%     hold off; fontsize(gca, 15,'points');

%     % filename2 = fullfile(output_folder, '3PopulationVelocOverTime_FLU3.fig');
%     % saveas(gcf, filename2);


% %% Plot of the trajectories for some times

% % Parameters
% time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% % Prepare figure
% figure;

%     % Plot positions as circles in 3D space
%     for i_t = 1:length(time)
%         t = time(i_t); % Current time index
%         plot3(x(1:n_l,1,t), x(1:n_l,2,t), t * ones(n_l,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none'); hold on
%         plot3(x(n_l+1:n_l+n_f,1,t), x(n_l+1:n_l+n_f,2,t), t * ones(n_f,1), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');hold on
%         plot3(x(n_l+n_f+1:n,1,t), x(n_l+n_f+1:n,2,t), t * ones(n_u,1), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none');hold on
    
%         hold on;
%     end

%     % % Plot velocities as arrows in 3D space
%     % for i_t = 1:length(time)
%     %     t = time(i_t); % Current time index
%     %     quiver3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), ...
%     %             v(1:n,1,t), v(1:n,2,t), zeros(n,1), 'r');
%     % end

%     % Label axes
%     xlabel('X');
%     ylabel('Y');
%     zlabel('Time');

%     % Adjust plot appearance
%     grid on;
%     %axis equal;
%     title('Position Over Time');
%     view(3); % Use a 3D perspective
%     hold off; fontsize(gca, 15,'points');

%     % filename3 = fullfile(output_folder, '3PopulationPositOverTime_FLU3.fig');
%     % saveas(gcf, filename3);


%  %% Another figure

% figure();
%     plot(1:steps,w(1:n_l,:), 'b', 'LineWidth', 1.5); hold on
%     plot(1:steps,w(n_l+1:n_l+n_f,:),'r', 'LineWidth', 1.5); hold on
%     plot(1:steps,w(n_l+n_f+1:end,:),'k', 'LineWidth', 1.5)
%     xlabel('Time');
%     ylabel('Opinion');
%     title('Opinion Over Time'); fontsize(gca, 15,'points');

%     filename2 = fullfile(output_folder, ['OpinionEvol_WithUninformed_FLUInteraction_case2_NL', num2str(n_l),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_taur_',num2str(tau_blue_l),'_taub_',num2str(tau_red_f), '.fig']);
%     saveas(gcf, filename2);


% %% Another figure

% % Get the total number of time steps
% num_time_steps = size(x, 3);

% % Initialize arrays to store mean velocities
% mean_vx_time = zeros(1, num_time_steps);
% mean_vy_time = zeros(1, num_time_steps);

% % Loop through each time step to calculate the mean velocity components
% for t = 1:num_time_steps
%     vx = v(1:n, 1, t); % x-component of velocity at time t
%     vy = v(1:n, 2, t); % y-component of velocity at time t
    
%     mean_vx_time(t) = mean(vx); % Mean x-component
%     mean_vy_time(t) = mean(vy); % Mean y-component
% end

% % Time vector
% time = 1:num_time_steps; % Modify if you have actual time values

% % Plot mean velocity components over time
% figure;

%     subplot(2, 1, 1);
%     plot(time, mean_vx_time, 'b-', 'LineWidth', 2);
%     xlabel('Time');
%     ylabel('Mean V_x');
%     title('Mean Velocity Component V_x Over Time');
%     grid on; fontsize(gca, 15,'points');

%     subplot(2, 1, 2);
%     plot(time, mean_vy_time, 'r-', 'LineWidth', 2);
%     xlabel('Time');
%     ylabel('Mean V_y');
%     title('Mean Velocity Component V_y Over Time');
%     grid on; fontsize(gca, 15,'points');

%     % filename5 = fullfile(output_folder, 'VelocityComponents_FLU2.fig');
%     % saveas(gcf, filename5);


% %% Another figure

% figure();
%     for t = 1:num_time_steps
%     wmean = w(1:n, t); % x-component of velocity at time t
%     mean_opinion(t) = mean(wmean); % Mean x-component
%     end
%     plot(time,mean_opinion, 'b', 'LineWidth', 1.5); hold on
%     xlabel('Time');
%     ylabel('Mean Opnion');
%     title('Mean opinion of the whole population');
%     grid on; fontsize(gca, 15,'points');

%     % filename6 = fullfile(output_folder, 'MeanOpinion_FLU2.fig');
%     % saveas(gcf, filename6);


% %% Another figure

% % Combine velocity components into magnitude (optional)
% figure;
%     mean_velocity_other = (mean_vx_time + mean_vy_time)/2;
%     plot(time, mean_velocity_other, 'k-', 'LineWidth', 2);
%     xlabel('Time');
%     ylabel('Mean Velocity');
%     title('Mean Velocity Over Time');
%     grid on; fontsize(gca, 15,'points');

%     % filename7 = fullfile(output_folder, 'MeanVelocityOther_FLU2.fig');
%     % saveas(gcf, filename7);


% %% Another figure

% figure;
%     mean_velocity_magnitude = sqrt(mean_vx_time.^2 + mean_vy_time.^2);
%     plot(time, mean_velocity_magnitude, 'k-', 'LineWidth', 2);
%     xlabel('Time');
%     ylabel('Mean Velocity Magnitude');
%     title('Mean Velocity Magnitude Over Time');
%     grid on; fontsize(gca, 15,'points');

%     % filename8 = fullfile(output_folder, 'MeanVelocityMagnitude_FLU2.fig');
%     % saveas(gcf, filename8);



%% -------------------PROBLEM-RELATED FUNCTIONS-------------------

function [dx, dv, dw] = ode_system(x_step, v_step, w_step, ns, alpha, beta, nabla_u, r_x, r_w, gammas_blue, gammas_red, tau_blue_l, tau_red_f)

    n = sum(ns);

    xs = {
        x_step(1:ns(1),:);                % leaders
        x_step(ns(1)+1:ns(2)+ns(1),:);        % followers
        x_step(ns(2)+ns(1)+1:n,:)           % uninformed
    };

    vs = {
        v_step(1:ns(1),:);
        v_step(ns(1)+1:ns(2)+ns(1),:);
        v_step(ns(2)+ns(1)+1:n,:)
    };

    ws = {
        w_step(1:ns(1),:);
        w_step(ns(1)+1:ns(2)+ns(1),:);
        w_step(ns(2)+ns(1)+1:n,:)
    };

    v_blue = 1;     % vector +(1,1)
    v_red = -1;     % vector -(1,1)

    w_blue = 1; 
    w_red = -1;

    dvs = {zeros(size(vs{1})), zeros(size(vs{2})), zeros(size(vs{3}))};
    for i = 1:3
        term_1 = (alpha - beta*sum(vs{i}.^2,2)).* vs{i};

        term_2 = 0;
        for j = 1:3
            term_2 = term_2 + 1/ns(j) * potential_sum(xs{i},xs{j},nabla_u);
        end

        term_3_blue = gammas_blue(i) * velocity_alignment(vs{i}, ws{i}, r_w, v_blue, w_blue);
        term_3_red = gammas_red(i) * velocity_alignment(vs{i}, ws{i}, r_w, v_red, w_red);
        term_3 = term_3_blue + term_3_red; % check if the term_3_right_L is needed

        dvs{i} = term_1 + term_2 + term_3;
    end
    dx = [vs{1}; vs{2}; vs{3}];
    dv = [dvs{1}; dvs{2}; dvs{3}];

    % [VVM] why different for interspecie?

    phis = cell(3,3);
    for i = 1:3
        for j = 1:3
            phis{i}{j} = 1/ns(j) * opinion_alignment_sum(xs{i}, xs{j}, ws{i}, ws{j}, ns(i), ns(j), r_x, r_w);
        end
    end
    dw_l = phis{1}{1} + phis{1}{2} + 0*phis{1}{2} - tau_blue_l*(ws{1} - w_blue);       % [VVM]
    dw_f = phis{2}{2} + phis{2}{1} + phis{2}{3}   - tau_red_f*(ws{2} - w_red);
    dw_u = 0*phis{3}{3} + 0*phis{3}{2} + 0*phis{3}{1};

    dw = [dw_l; dw_f; dw_u];
end


function phi = opinion_alignment_endogamic(x_step,w_step,n,r_x,r_w)
    wi = repmat(w_step, 1, n);
    wj = repmat(w_step', n, 1);    
    xi = reshape(x_step, [n, 1, size(x_step, 2)]);
    xj = reshape(x_step, [1, n, size(x_step, 2)]);
    z  = xi - xj;
    d_ij = sqrt(sum((z).^2, 3));   
    bool_phi = (abs(wi - wj) < r_w) & (d_ij < r_x); 
    %bool_phi = (abs(wi - wj) < r_w); 
    phi      = sum(bool_phi .* (wj - wi), 2);
end


function phi = opinion_alignment_sum(x_step, y_step, w_step, q, n_x, n_y, r_x, r_w)   
    % Reshape opinions for pairwise comparison
    wi = reshape(w_step, [n_x, 1]);                     % Size: (n_x, 1)
    wj = reshape(q, [1, n_y]);                          % Size: (1, n_y)
    xi = reshape(x_step, [n_x, 1, size(x_step, 2)]);    % Size: (n_x, 1, D)
    xj = reshape(y_step, [1, n_y, size(y_step, 2)]);    % Size: (1, n_y, D)
    
    z = xi - xj;                                        % Size: (n_x, n_y, D)
    d_ij = sqrt(sum(z.^2, 3));                       % Size: (n_x, n_y)
    
    bool_phi = (abs(wi - wj) < r_w) & (d_ij < r_x); % Size: (n_x, n_y)
    %bool_phi = (abs(wi - wj) < r_w); % Size: (n_x, n_y)
    
    % Compute the alignment factor
    phi = sum(bool_phi .* (wj - wi), 2); % Size: (n_x, 1)
end


function forces = potential_sum_endogamic(x_step,nabla_u)
    xi = x_step;
    xj = reshape(x_step', 1, size(x_step, 2), []); 
    z = xi - xj; 
    d_ij = sqrt(sum(z.^2, 2));
    d_ij(d_ij == 0) = 1;
    forces = -nabla_u(d_ij) .* z ./ d_ij;
    forces = squeeze(sum(forces, 3));    
end


function forces = potential_sum(x_step, y_step, nabla_u)
    % x_step and y_step are inputs for positions of different species (sizes may differ)
    % nabla_u is the derivative of the potential function
    
    % Expand x_step and y_step to match dimensions for pairwise distance computation
    xi = reshape(x_step, size(x_step, 1), 1, size(x_step, 2)); % Reshape xi to (Nx, 1, D)
    xj = reshape(y_step, 1, size(y_step, 1), size(y_step, 2)); % Reshape xj to (1, Ny, D)
    
    % Compute pairwise increments xi - xj
    z = xi - xj; % Resulting size: (Nx, Ny, D)
    
    % Compute pairwise distances
    d_ij = sqrt(sum(z.^2, 3)); % Resulting size: (Nx, Ny)
    d_ij(d_ij == 0) = 1; % Avoid division by zero
    
    % Apply the derivative of the potential function
    potential_term = nabla_u(d_ij); % Evaluate nabla_u at distances, size: (Nx, Ny)
    
    % Compute forces
    forces = -potential_term .* z ./ d_ij; % Size: (Nx, Ny, D)
    
    % Sum over the second dimension (interaction sum for each particle in x_step)
    forces = squeeze(sum(forces, 2)); % Resulting size: (Nx, D)
end


function morse_potential = morse_potential(r, c_rep, c_att, l_rep, l_att)
    morse_potential = -(c_rep/l_rep)*exp(-r./l_rep) + (c_att/l_att)*exp(-r./l_att);
end


function alignment = velocity_alignment(v,w1,r_w,vt,wt)
    bool_alignment = abs(w1 - wt) < r_w;
    alignment = bool_alignment.*(vt-v);
end


% function alignment = velocity_alignment(l,w1,r_w,xt,wt)
%     bool_alignment = abs(w1 - wt) < r_w;
%     alignment = bool_alignment.*(xt-l);
% end

