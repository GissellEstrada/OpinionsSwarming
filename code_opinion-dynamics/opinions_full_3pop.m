clear all;
% close all;


%% -------------------READ ME-----------------------

% The "cube" x stores both spatial coordinates and velocities for each instant in time:
% x = [spatial coord; velocity coord];
% The indexing for each kind of velocity and spatial coord. can be found in
% the line where I plot the solution
% The vector w stores opinions from each subpopulation:
% w = [op];

% The only place where I multiply by the factor 1/n is in the sumphi term,
% which is only found in the equation regarding w

%----------------------------------------------%

% All the parameters used in the model are stated and described right bellow

% Other possible parameters. Changing tau_red and tau_blue also gives very rich dynamics
% alpha = 1; beta = 5;
% c_att = 100; l_att = 1.2; c_rep = 350; l_rep = 0.8; 
% %l_att = 2.2


%% Problem data

n_l = 20;
n_f = 50;
n_u = 50;                                   % we need to change n_u=2, 20, 80
n = n_l + n_f + n_u;

% x = zeros(2*n,2,steps);
% v = zeros(n, 2, steps);
% w = zeros(n, steps)

t_final = 100;                              % final time
dt = 1.0e-2;                                % timestep
steps = floor(t_final/dt);                  % number of time steps

alpha = 1; beta = 0.5;                      % friction parameters
c_att = 50; l_att = 1;
c_rep = 60; l_rep = 0.5;

nabla_u = @(r) morsepotential(r, c_rep, c_att, l_rep, l_att);

r_x = 1;%5;                                 % spatial radius for alignment in velocity 1
r_w = 0.5;%1;                               % preference radius for alignment il velocity 0.5

% blue: preference towards 1 (leaders), red: preference towards -1 (followers)

gamma_l_red = 1;                            % alignment in velocity
gamma_l_blue = 1;                           % alignment in velocity
tau_l_blue = 0.1;                           % alignment in opinion

gamma_f_blue = 1;
gamma_f_red = 1;
tau_f_red = 0.01;

gamma_u_red = 0;
gamma_u_blue = 0;

% Initial conditions

rng(1234);

x_l_0 = -1 + 2*rand(n_l,2);
x_f_0 = -1 + 2*rand(n_f,2); 
x_u_0 = -1 + 2*rand(n_u,2); 

v_l_0 = -1 + 2*rand(n_l,2);
v_f_0 = -1 + 2*rand(n_f,2);
v_u_0 = -1 + 2*rand(n_u,2);

w_l_0 = -1 + 2*rand(n_l,1);
w_f_0 = -1 + 2*rand(n_f,1);
w_u_0 = -0.1 + 0.2*rand(n_u,1);             % the uninformed opinions start very close to zero

x_0 = [x_l_0; x_f_0; x_u_0];
v_0 = [v_l_0; v_f_0; v_u_0];
w_0 = [w_l_0; w_f_0; 0*w_u_0];              % [VVM] 0?
% w_0 = [w_l_0; w_f_0; w_u_0];


%% Integration step

% Preparation: compute the first solution using Euler. Use it to compute the second system.

x = zeros(2*n,2,steps);

x(:,:,1) = [x_0; v_0];
w(:,1) = w_0;

[dx_1, dw_1] = ode_system(x(:,:,1), w(:,1), n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gamma_l_red, gamma_l_blue, tau_l_blue, gamma_f_blue, gamma_f_red, tau_f_red, gamma_u_red, gamma_u_blue);

x(:,:,2) = x(:,:,1) + dt * dx_1; 
w(:,2) = w(:,1) + dt * dw_1; 

[dx_2, dw_2] = ode_system(x(:,:,2), w(:,2), n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gamma_l_red, gamma_l_blue, tau_l_blue, gamma_f_blue, gamma_f_red, tau_f_red, gamma_u_red, gamma_u_blue);

% Adams-Bashforth 2-step method

for i = 3:steps
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * dx_2 - dx_1);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * dw_2 - dw_1);
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')   % [VVM] also v?
        error('NaN or Inf encountered at time step %d', i);
    end

    dx_1 = dx_2;
    dw_1 = dw_2;

    [dx_2, dw_2] = ode_system(x(:,:,i), w(:,i), n_l, n_f, n_u, alpha, beta,nabla_u, r_x, r_w, gamma_l_red, gamma_l_blue, tau_l_blue, gamma_f_blue, gamma_f_red, tau_f_red, gamma_u_red, gamma_u_blue);
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

    quiver(x(1:n,1,end), x(1:n,2,end), x(n+1:2*n,1,end), x(n+1:2*n,2,end), 'r'); % Plot velocities as arrows

    xlabel('X');
    ylabel('Y');

    grid on;
    title('Velocities at the final time');

    hold off;
    axis equal; fontsize(gca, 15,'points');

    filename1 = fullfile(output_folder, ['3PopOp_WithUninformed_FLUInteraction_case2_NL', num2str(n_l),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_taur_',num2str(tau_l_blue),'_taub_',num2str(tau_f_red), '.fig']);
    saveas(gcf, filename1);

% %% Plot for several times

% % Parameters
% time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% % Prepare figure
% figure;

%     % Plot positions as circles in 3D space
%     for tIdx = 1:length(time)
%         t = time(tIdx); % Current time index
%         plot3(x(1:n_l,1,t), x(1:n_l,2,t), t * ones(n_l,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none'); hold on
%         plot3(x(n_l+1:n_l+n_f,1,t), x(n_l+1:n_l+n_f,2,t), t * ones(n_f,1), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');hold on
%         plot3(x(n_l+n_f+1:n,1,t), x(n_l+n_f+1:n,2,t), t * ones(n_u,1), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none');hold on
    
%         hold on;
%     end

%     % Plot velocities as arrows in 3D space
%     for tIdx = 1:length(time)
%         t = time(tIdx); % Current time index
%         quiver3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), ...
%                 x(n+1:2*n,1,t), x(n+1:2*n,2,t), zeros(n,1), 'r');
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
%     for tIdx = 1:length(time)
%         t = time(tIdx); % Current time index
%         plot3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
%         hold on;
%     end

%     % Plot velocities as arrows in 3D space
%     for tIdx = 1:length(time)
%         t = time(tIdx); % Current time index
%         quiver3(x(1:n,1,t), x(1:n,2,t), t * ones(n,1), ...
%                 x(1:n,1,t), x(1:n,2,t), zeros(n,1), 'r');
%     end

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

%     filename2 = fullfile(output_folder, ['OpinionEvol_WithUninformed_FLUInteraction_case2_NL', num2str(n_l),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_taur_',num2str(tau_l_blue),'_taub_',num2str(tau_f_red), '.fig']);
%     saveas(gcf, filename2);


% %% Another figure

% % Get the total number of time steps
% num_time_steps = size(x, 3);

% % Initialize arrays to store mean velocities
% mean_vx_time = zeros(1, num_time_steps);
% mean_vy_time = zeros(1, num_time_steps);

% % Loop through each time step to calculate the mean velocity components
% for t = 1:num_time_steps
%     vx = x(n+1:2*n, 1, t); % x-component of velocity at time t
%     vy = x(n+1:2*n, 2, t); % y-component of velocity at time t
    
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

function [dx, dw] = ode_system(x_step, w_step, n_l, n_f, n_u, alpha, beta, nabla_u, r_x, r_w, gamma_l_red, gamma_l_blue, tau_l_blue, gamma_f_blue, gamma_f_red, tau_f_red, gamma_u_red, gamma_u_blue)

    n = n_l + n_f + n_u;

    x_l = x_step(1:n_l,:);
    x_f = x_step(n_l+1:n_f+n_l,:);
    x_u = x_step(n_f+n_l+1:n,:);

    v_l = x_step(n+1:n+n_l,:);
    v_f = x_step(n+n_l+1:n+n_l+n_f,:);
    v_u = x_step(n+n_l+n_f+1:end,:);

    w_l = w_step(1:n_l);
    w_f = w_step(n_l+1:n_l+n_f);
    w_u = w_step(n_l+n_f+1:end);

    v_blue = 1;
    v_red = -1;
    % v_blue = [1, 0];      [VVM]
    % v_red = [0, -1];

    w_blue = 1; 
    w_red = -1;   
   
    % x_blue = 1;
    % x_red = -1;

    term1_LL     = 1/n_l * computePotentialTerm(x_l,nabla_u); % 
    term1_FF     = 1/n_f * computePotentialTerm(x_f,nabla_u); % 
    term1_UU     = 1/n_u * computePotentialTerm(x_u,nabla_u); % 

    term1_LF     = 1/n_f * computePotentialTermInterSpecie(x_l,x_f,nabla_u);
    term1_FL     = 1/n_l * computePotentialTermInterSpecie(x_f,x_l,nabla_u);

    term1_LU     = 1/n_u * computePotentialTermInterSpecie(x_l,x_u,nabla_u);
    term1_UL     = 1/n_l * computePotentialTermInterSpecie(x_u,x_l,nabla_u);

    term1_FU     = 1/n_u * computePotentialTermInterSpecie(x_f,x_u,nabla_u);
    term1_UF     = 1/n_f * computePotentialTermInterSpecie(x_u,x_f,nabla_u);

    term2_L     = (alpha - beta*sum(v_l.^2,2)).* v_l;
    term2_F     = (alpha - beta*sum(v_f.^2,2)).* v_f;
    term2_U     = (alpha - beta*sum(v_u.^2,2)).* v_u;

    termRight_L  = gamma_l_blue * computeOpinionAlignmentPreference(v_l,w_l,r_w,v_blue,w_blue);
    termLeft_L   = gamma_l_red * computeOpinionAlignmentPreference(v_l,w_l,r_w,v_red,w_red);

%     termRight_L  = gamma_l_blue * computeOpinionAlignmentPreference(x_l,w_l,r_w,x_blue,w_blue);
%     termLeft_L   = gamma_l_red * computeOpinionAlignmentPreference(x_l,w_l,r_w,x_red,w_red);

    termLeft_F  = gamma_f_red * computeOpinionAlignmentPreference(v_f,w_f,r_w,v_red,w_red);
    termRight_F = gamma_f_blue * computeOpinionAlignmentPreference(v_f,w_f,r_w,v_blue,w_blue);

%     termLeft_F  = gamma_f_red * computeOpinionAlignmentPreference(x_f,w_f,r_w,x_red,w_red);
%     termRight_F = gamma_f_blue * computeOpinionAlignmentPreference(x_f,w_f,r_w,x_blue,w_blue);

    termLeft_U  = gamma_u_red * computeOpinionAlignmentPreference(v_u,w_u,r_w,v_red,w_red);
    termRight_U = gamma_u_blue * computeOpinionAlignmentPreference(v_u,w_u,r_w,v_blue,w_blue);

%     termLeft_U  = gamma_u_red * computeOpinionAlignmentPreference(x_u,w_u,r_w,x_red,w_red);
%     termRight_U = gamma_u_blue * computeOpinionAlignmentPreference(x_u,w_u,r_w,x_blue,w_blue);

    dvL = term1_LL + term1_LF + term1_LU + term2_L + termRight_L + termLeft_L; % check if the termLeft_L is needed
    dvF = term1_FL + term1_FF + term1_FU + term2_F + termLeft_F + termRight_F;
    dvU = term1_UL + term1_UF + term1_UU + term2_U + termLeft_U + termRight_U;

    dx = [v_l; v_f; v_u; dvL; dvF; dvU];

    phi_LL = 1/n_l * computeOpinionAligment(x_l, w_l, n_l, r_x, r_w);
    phi_FF = 1/n_f * computeOpinionAligment(x_f, w_f, n_f, r_x, r_w);
    phi_UU = 1/n_u * computeOpinionAligment(x_u, w_u, n_u, r_x, r_w);

    phi_LF = 1/n_f * computeOpinionAlignmentInterSpecie(x_l, x_f, w_l, w_f, n_l, n_f, r_x, r_w);
    phi_LU = 1/n_u * computeOpinionAlignmentInterSpecie(x_l, x_u, w_l, w_u, n_l, n_u, r_x, r_w);

    phi_FL = 1/n_l * computeOpinionAlignmentInterSpecie(x_f, x_l, w_f, w_l, n_f, n_l, r_x, r_w);
    phi_UL = 1/n_l * computeOpinionAlignmentInterSpecie(x_u, x_l, w_u, w_l, n_u, n_l, r_x, r_w);

    phi_FU = 1/n_u * computeOpinionAlignmentInterSpecie(x_f, x_u, w_f, w_u, n_f, n_u, r_x, r_w);
    phi_UF = 1/n_f * computeOpinionAlignmentInterSpecie(x_u, x_f, w_u, w_f, n_u, n_f, r_x, r_w);

    dwL = phi_LL + phi_LF + 0*phi_LU - tau_l_blue*(w_l - w_blue);
    dwF = phi_FF + phi_FL + phi_FU - tau_f_red*(w_f - w_red);
    dwU = 0*phi_UU + 0*phi_UF + 0*phi_UL;

    dw = [dwL; dwF; dwU];

       
end


function phi = computeOpinionAligment(x_step,w_step,n,r_x,r_w)
    wi = repmat(w_step, 1, n);
    wj = repmat(w_step', n, 1);    
    xi = reshape(x_step, [n, 1, size(x_step, 2)]);
    xj = reshape(x_step, [1, n, size(x_step, 2)]);
    incXij  = xi - xj;
    distWij = sqrt(sum((incXij).^2, 3));   
    isClosed = (abs(wi - wj) < r_w) & (distWij < r_x); 
    %isClosed = (abs(wi - wj) < r_w); 
    phi      = sum(isClosed .* (wj - wi), 2);
end


function phi = computeOpinionAlignmentInterSpecie(x_step, y_step, w_step, q, n, NN, r_x, r_w)
    % x_step, y_step: Positions of two different species
    % w_step, q: Opinions (scalars or vectors) corresponding to x_step and y_step
    % n, NN: Number of individuals in each species
    % r_x: Radius for spatial influence
    % r_w: Radius for opinion influence
    
    % Reshape opinions for pairwise comparison
    wi = reshape(w_step, [n, 1]);  % Size: (n, 1)
    wj = reshape(q, [1, NN]); % Size: (1, NN)
    
    % Reshape positions for pairwise computation
    xi = reshape(x_step, [n, 1, size(x_step, 2)]); % Size: (n, 1, D)
    xj = reshape(y_step, [1, NN, size(y_step, 2)]); % Size: (1, NN, D)
    
    % Compute pairwise distances between positions
    incXij = xi - xj; % Size: (n, NN, D)
    distXij = sqrt(sum(incXij.^2, 3)); % Size: (n, NN)
    
    % Identify pairs satisfying both opinion and spatial thresholds
    isClosed = (abs(wi - wj) < r_w) & (distXij < r_x); % Size: (n, NN)
    %isClosed = (abs(wi - wj) < r_w); % Size: (n, NN)
    
    % Compute the alignment factor
    phi = sum(isClosed .* (wj - wi), 2); % Size: (n, 1)
end


function forces = computePotentialTerm(x_step,nabla_u)
    xi = x_step;
    xj = reshape(x_step', 1, size(x_step, 2), []); 
    incXiXj = xi - xj; 
    distXiXj = sqrt(sum(incXiXj.^2, 2));
    distXiXj(distXiXj == 0) = 1;
    forces = -nabla_u(distXiXj) .* incXiXj ./ distXiXj;
    forces = squeeze(sum(forces, 3));    
end


function forces = computePotentialTermInterSpecie(x_step, y_step, nabla_u)
    % x_step and y_step are inputs for positions of different species (sizes may differ)
    % nabla_u is the derivative of the potential function
    
    % Expand x_step and y_step to match dimensions for pairwise distance computation
    xi = reshape(x_step, size(x_step, 1), 1, size(x_step, 2)); % Reshape xi to (Nx, 1, D)
    xj = reshape(y_step, 1, size(y_step, 1), size(y_step, 2)); % Reshape xj to (1, Ny, D)
    
    % Compute pairwise increments xi - xj
    incXiXj = xi - xj; % Resulting size: (Nx, Ny, D)
    
    % Compute pairwise distances
    distXiXj = sqrt(sum(incXiXj.^2, 3)); % Resulting size: (Nx, Ny)
    distXiXj(distXiXj == 0) = 1; % Avoid division by zero
    
    % Apply the derivative of the potential function
    potentialTerm = nabla_u(distXiXj); % Evaluate nabla_u at distances, size: (Nx, Ny)
    
    % Compute forces
    forces = -potentialTerm .* incXiXj ./ distXiXj; % Size: (Nx, Ny, D)
    
    % Sum over the second dimension (interaction sum for each particle in x_step)
    forces = squeeze(sum(forces, 2)); % Resulting size: (Nx, D)
end


function morsepotential = morsepotential(r, c_rep, c_att, l_rep, l_att)
    morsepotential = -(c_rep/l_rep)*exp(-r./l_rep) + (c_att/l_att)*exp(-r./l_att);
end


function Op_align = computeOpinionAlignmentPreference(v,w1,r_w,vt,wt)
    isAligned = abs(w1 - wt) < r_w;
    Op_align = isAligned.*(vt-v);
end


% function Op_align = computeOpinionAlignmentPreference(l,w1,r_w,xt,wt)
%     isAligned = abs(w1 - wt) < r_w;
%     Op_align = isAligned.*(xt-l);
% end

