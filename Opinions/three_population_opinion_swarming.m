clear all;
close all;


%-------------------READ ME-----------------------%

% The "cube" x stores both spatial coordinates and velocities for each instant in time:
% x = [spatial coord; velocity coord];
% The indexing for each kind of velocity and spatial coord. can be found in
% the line where I plot the solution
% The vector w stores opinions from each subpopulation:
% w = [op];

% The only place where I multiply by the factor 1/N is in the sumphi term,
% which is only found in the equation regarding w

% All the parameters used in the model are stated and described right bellow

%----------------------------------------------%

% Problem data
NL = 20;                             % number of individuals from each group (change the number N to get different dynamics)
NF = 50;
NU = 50;                             % we need to chenge NU=2, 20, 80

N = NL + NF + NU;
%%
alpha = 1; beta = 0.5;                        % friction parameters
Ca = 50; la = 1; Cr = 60; lr = 0.5;           % rotating mill

% alpha = 1; beta = 5;
% Ca = 100; la = 1.2; Cr = 350; lr = 0.8; %la = 2.2

%%
T = 100;                             % final time
dt = 1.0e-2;                        % timestep
M = floor(T/dt);                    % number of time steps

rx = 1;%5;                             % spatial radius for alignment in velocity 1
rw = 0.5;%1;                           % preference radius for alignment il velocity 0.5

%% R: preference towards 1 (leaders), L: preference towards -1 (followers)

%
gammaL_left = 1;                   % alignment in velocity
gammaL_right = 1;                  % alignment in velocity
tauL_right = 0.1;                  % alignment in opinion

gammaF_right = 1;                  % alignment in velocity
gammaF_left = 1;                   % alignment in velocity
tauF_left = 0.01;                  % alignment in opinion

gammaU_left = 0;
gammaU_right = 0;


%% Changing taur and taub gives very rich dynamics

% Initial conditions
x0L = -1 + 2*rand(NL,2); v0L = -1 + 2*rand(NL,2);
w0L = -1 + 2*rand(NL,1);

x0F = -1 + 2*rand(NF,2); v0F = -1 + 2*rand(NF,2);
w0F = -1 + 2*rand(NF,1);

x0U = -1 + 2*rand(NU,2); v0U = -1 + 2*rand(NU,2);
w0U = -0.1 + 0.2*rand(NU,1);              % the uninformed opinions start very close to zero
%w0U = -1 + 2*rand(NU,1);

x0 = [x0L;x0F;x0U];
v0 = [v0L;v0F;v0U];
w0 = [w0L;w0F;0*w0U];

dU = @(r) morsepotential(r, Cr, Ca, lr, la);

% Initial condition for Adams-Bashforth
x = zeros(2*N,2,M);
x(:,:,1) = [x0;v0]; % Initial state
w(:,1) = w0;
[F1x, F1w] = F(x(:,:,1), w(:,1), NL, NF, NU, alpha, beta, dU, rx, rw, gammaL_left, gammaL_right, tauL_right, gammaF_right, gammaF_left, tauF_left, gammaU_left, gammaU_right); % Initial F value

% Use Euler for the first step
x(:,:,2) = x(:,:,1) + dt * F1x; 
w(:,2) = w(:,1) + dt * F1w; 
% Compute F for the second step (needed for Adams-Bashforth)
[F2x, F2w] = F(x(:,:,2), w(:,2), NL, NF, NU, alpha, beta, dU, rx, rw, gammaL_left, gammaL_right, tauL_right, gammaF_right, gammaF_left, tauF_left, gammaU_left, gammaU_right);

% Adams-Bashforth 2-step method
for i = 3:M
    % Adams-Bashforth 2-step formula
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * F2x - F1x);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * F2w - F1w);
    % Update F values for the next step
    F1x = F2x;  % F from previous step
    F1w = F2w;
    [F2x, F2w] = F(x(:,:,i), w(:,i), NL, NF, NU, alpha, beta,dU, rx, rw, gammaL_left, gammaL_right, tauL_right, gammaF_right, gammaF_left, tauF_left, gammaU_left, gammaU_right);  % F for current step
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')
        error('NaN or Inf encountered at time step %d', i);
    end
end


%% Save figure

outputFolder = 'Figures_Mill_allcases';      % Folder to save the figures
% Create the folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


%% Plot of the final velocity configuration
figure()
plot(x(1:NL,1,end), x(1:NL,2,end), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');  % Plot positions as circles
hold on;
plot(x(NL+1:NL+NF,1,end), x(NL+1:NL+NF,2,end), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
hold on;
plot(x(NL+NF+1:N,1,end), x(NL+NF+1:N,2,end), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');

quiver(x(1:N,1,end), x(1:N,2,end), x(N+1:2*N,1,end), x(N+1:2*N,2,end), 'r'); % Plot velocities as arrows

xlabel('X');
ylabel('Y');

grid on;
title('Velocities at the final time');

hold off;
axis equal; fontsize(gca, 15,'points');

filename1 = fullfile(outputFolder, ['3PopOp_WithUninformed_FLUInteraction_case2_NL', num2str(NL),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(Ca),'_la_',num2str(la),'_Cr_',num2str(Cr),'_lr_',num2str(lr),'_rx_',num2str(rx),'_rw_',num2str(rw),'_taur_',num2str(tauL_right),'_taub_',num2str(tauF_left), '.fig']);
saveas(gcf, filename1);

%% Plot for several times

% Parameters
time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% Prepare figure
figure;

% Plot positions as circles in 3D space
for tIdx = 1:length(time)
    t = time(tIdx); % Current time index
    plot3(x(1:NL,1,t), x(1:NL,2,t), t * ones(NL,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none'); hold on
    plot3(x(NL+1:NL+NF,1,t), x(NL+1:NL+NF,2,t), t * ones(NF,1), 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');hold on
    plot3(x(NL+NF+1:N,1,t), x(NL+NF+1:N,2,t), t * ones(NU,1), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none');hold on
   
    hold on;
end

% Plot velocities as arrows in 3D space
for tIdx = 1:length(time)
    t = time(tIdx); % Current time index
    quiver3(x(1:N,1,t), x(1:N,2,t), t * ones(N,1), ...
            x(N+1:2*N,1,t), x(N+1:2*N,2,t), zeros(N,1), 'r');
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
hold off; fontsize(gca, 15,'points');

% filename2 = fullfile(outputFolder, '3PopulationVelocOverTime_FLU3.fig');
% saveas(gcf, filename2);


%% Plot of the trajectories for some times

% Parameters
time = (1:1000:size(x, 3)); % Time indices corresponding to the 1:1000:end sampling

% Prepare figure
figure;

% Plot positions as circles in 3D space
for tIdx = 1:length(time)
    t = time(tIdx); % Current time index
    plot3(x(1:N,1,t), x(1:N,2,t), t * ones(N,1), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');
    hold on;
end

% Plot velocities as arrows in 3D space
for tIdx = 1:length(time)
    t = time(tIdx); % Current time index
    quiver3(x(1:N,1,t), x(1:N,2,t), t * ones(N,1), ...
            x(1:N,1,t), x(1:N,2,t), zeros(N,1), 'r');
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
hold off; fontsize(gca, 15,'points');

% filename3 = fullfile(outputFolder, '3PopulationPositOverTime_FLU3.fig');
% saveas(gcf, filename3);

 %%


figure();
plot(1:M,w(1:NL,:), 'b', 'LineWidth', 1.5); hold on
plot(1:M,w(NL+1:NL+NF,:),'r', 'LineWidth', 1.5); hold on
plot(1:M,w(NL+NF+1:end,:),'k', 'LineWidth', 1.5)
xlabel('Time');
ylabel('Opinion');
title('Opinion Over Time'); fontsize(gca, 15,'points');

filename2 = fullfile(outputFolder, ['OpinionEvol_WithUninformed_FLUInteraction_case2_NL', num2str(NL),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(Ca),'_la_',num2str(la),'_Cr_',num2str(Cr),'_lr_',num2str(lr),'_rx_',num2str(rx),'_rw_',num2str(rw),'_taur_',num2str(tauL_right),'_taub_',num2str(tauF_left), '.fig']);
saveas(gcf, filename2);



% Get the total number of time steps
num_time_steps = size(x, 3);

% Initialize arrays to store mean velocities
mean_vx_time = zeros(1, num_time_steps);
mean_vy_time = zeros(1, num_time_steps);

% Loop through each time step to calculate the mean velocity components
for t = 1:num_time_steps
    vx = x(N+1:2*N, 1, t); % x-component of velocity at time t
    vy = x(N+1:2*N, 2, t); % y-component of velocity at time t
    
    mean_vx_time(t) = mean(vx); % Mean x-component
    mean_vy_time(t) = mean(vy); % Mean y-component
end

% Time vector
time = 1:num_time_steps; % Modify if you have actual time values

% Plot mean velocity components over time
figure;

subplot(2, 1, 1);
plot(time, mean_vx_time, 'b-', 'LineWidth', 2);
xlabel('Time');
ylabel('Mean V_x');
title('Mean Velocity Component V_x Over Time');
grid on; fontsize(gca, 15,'points');

subplot(2, 1, 2);
plot(time, mean_vy_time, 'r-', 'LineWidth', 2);
xlabel('Time');
ylabel('Mean V_y');
title('Mean Velocity Component V_y Over Time');
grid on; fontsize(gca, 15,'points');

% filename5 = fullfile(outputFolder, 'VelocityComponents_FLU2.fig');
% saveas(gcf, filename5);


figure();
 for t = 1:num_time_steps
    wmean = w(1:N, t); % x-component of velocity at time t
    mean_opinion(t) = mean(wmean); % Mean x-component
 end
 plot(time,mean_opinion, 'b', 'LineWidth', 1.5); hold on
xlabel('Time');
ylabel('Mean Opnion');
title('Mean opinion of the whole population');
grid on; fontsize(gca, 15,'points');

% filename6 = fullfile(outputFolder, 'MeanOpinion_FLU2.fig');
% saveas(gcf, filename6);


% Combine velocity components into magnitude (optional)
figure;
mean_velocity_other = (mean_vx_time + mean_vy_time)/2;
plot(time, mean_velocity_other, 'k-', 'LineWidth', 2);
xlabel('Time');
ylabel('Mean Velocity');
title('Mean Velocity Over Time');
grid on; fontsize(gca, 15,'points');

% filename7 = fullfile(outputFolder, 'MeanVelocityOther_FLU2.fig');
% saveas(gcf, filename7);

figure;
mean_velocity_magnitude = sqrt(mean_vx_time.^2 + mean_vy_time.^2);
plot(time, mean_velocity_magnitude, 'k-', 'LineWidth', 2);
xlabel('Time');
ylabel('Mean Velocity Magnitude');
title('Mean Velocity Magnitude Over Time');
grid on; fontsize(gca, 15,'points');

% filename8 = fullfile(outputFolder, 'MeanVelocityMagnitude_FLU2.fig');
% saveas(gcf, filename8);


% -------------------PROBLEM-RELATED FUNCTIONS------------------- %

function [dx, dw] = F(x, w, NL, NF, NU, alpha, beta, dU, rx, rw, gammaL_left, gammaL_right, tauL_right, gammaF_right, gammaF_left, tauF_left, gammaU_left, gammaU_right)

    N = NL + NF + NU;

    xL = x(1:NL,:);
    xF = x(NL+1:NF+NL,:);
    xU = x(NF+NL+1:N,:);

    vL = x(N+1:N+NL,:);
    vF = x(N+NL+1:N+NL+NF,:);
    vU = x(N+NL+NF+1:end,:);

    wL = w(1:NL);
    wF = w(NL+1:NL+NF);
    wU = w(NL+NF+1:end);

    vRight = 1;
    wRight = 1;    
    xRight = 1;

    vLeft = -1;
    wLeft = -1;   
    xLeft = -1;

    term1_LL     = 1/NL * computePotentialTerm(xL,dU); % 
    term1_FF     = 1/NF * computePotentialTerm(xF,dU); % 
    term1_UU     = 1/NU * computePotentialTerm(xU,dU); % 

    term1_LF     = 1/NF * computePotentialTermInterSpecie(xL,xF,dU);
    term1_FL     = 1/NL * computePotentialTermInterSpecie(xF,xL,dU);

    term1_LU     = 1/NU * computePotentialTermInterSpecie(xL,xU,dU);
    term1_UL     = 1/NL * computePotentialTermInterSpecie(xU,xL,dU);

    term1_FU     = 1/NU * computePotentialTermInterSpecie(xF,xU,dU);
    term1_UF     = 1/NF * computePotentialTermInterSpecie(xU,xF,dU);

    term2_L     = (alpha - beta*sum(vL.^2,2)).* vL;
    term2_F     = (alpha - beta*sum(vF.^2,2)).* vF;
    term2_U     = (alpha - beta*sum(vU.^2,2)).* vU;

    termRight_L  = gammaL_right * computeOpinionAlignmentPreference(vL,wL,rw,vRight,wRight);
    termLeft_L   = gammaL_left * computeOpinionAlignmentPreference(vL,wL,rw,vLeft,wLeft);

%     termRight_L  = gammaL_right * computeOpinionAlignmentPreference(xL,wL,rw,xRight,wRight);
%     termLeft_L   = gammaL_left * computeOpinionAlignmentPreference(xL,wL,rw,xLeft,wLeft);

    termLeft_F  = gammaF_left * computeOpinionAlignmentPreference(vF,wF,rw,vLeft,wLeft);
    termRight_F = gammaF_right * computeOpinionAlignmentPreference(vF,wF,rw,vRight,wRight);

%     termLeft_F  = gammaF_left * computeOpinionAlignmentPreference(xF,wF,rw,xLeft,wLeft);
%     termRight_F = gammaF_right * computeOpinionAlignmentPreference(xF,wF,rw,xRight,wRight);

    termLeft_U  = gammaU_left * computeOpinionAlignmentPreference(vU,wU,rw,vLeft,wLeft);
    termRight_U = gammaU_right * computeOpinionAlignmentPreference(vU,wU,rw,vRight,wRight);

%     termLeft_U  = gammaU_left * computeOpinionAlignmentPreference(xU,wU,rw,xLeft,wLeft);
%     termRight_U = gammaU_right * computeOpinionAlignmentPreference(xU,wU,rw,xRight,wRight);

    dvL = term1_LL + term1_LF + term1_LU + term2_L + termRight_L + termLeft_L; % check if the termLeft_L is needed
    dvF = term1_FF + term1_FL + term1_FU + term2_F + termLeft_F + termRight_F;
    dvU = term1_UU + term1_UL + term1_UF + term2_U + termLeft_U + termRight_U;

    dx = [vL; vF; vU; dvL; dvF; dvU];

    phi_LL = 1/NL * computeOpinionAligment(xL, wL, NL, rx, rw);
    phi_FF = 1/NF * computeOpinionAligment(xF, wF, NF, rx, rw);
    phi_UU = 1/NU * computeOpinionAligment(xU, wU, NU, rx, rw);

    phi_LF = 1/NF * computeOpinionAlignmentInterSpecie(xL, xF, wL, wF, NL, NF, rx, rw);
    phi_LU = 1/NU * computeOpinionAlignmentInterSpecie(xL, xU, wL, wU, NL, NU, rx, rw);

    phi_FL = 1/NL * computeOpinionAlignmentInterSpecie(xF, xL, wF, wL, NF, NL, rx, rw);
    phi_UL = 1/NL * computeOpinionAlignmentInterSpecie(xU, xL, wU, wL, NU, NL, rx, rw);

    phi_FU = 1/NU * computeOpinionAlignmentInterSpecie(xF, xU, wF, wU, NF, NU, rx, rw);
    phi_UF = 1/NF * computeOpinionAlignmentInterSpecie(xU, xF, wU, wF, NU, NF, rx, rw);

    dwL = phi_LL + phi_LF + 0*phi_LU - tauL_right*(wL - wRight);
    dwF = phi_FF + phi_FL + phi_FU - tauF_left*(wF - wLeft);
    dwU = 0*phi_UU + 0*phi_UF + 0*phi_UL;

    dw = [dwL; dwF; dwU];

       
end

function phi = computeOpinionAligment(x,w,N,rx,rw)
    wi = repmat(w, 1, N);
    wj = repmat(w', N, 1);    
    xi = reshape(x, [N, 1, size(x, 2)]);
    xj = reshape(x, [1, N, size(x, 2)]);
    incXij  = xi - xj;
    distWij = sqrt(sum((incXij).^2, 3));   
    isClosed = (abs(wi - wj) < rw) & (distWij < rx); 
    %isClosed = (abs(wi - wj) < rw); 
    phi      = sum(isClosed .* (wj - wi), 2);
end

function phi = computeOpinionAlignmentInterSpecie(x, y, w, q, N, NN, rx, rw)
    % x, y: Positions of two different species
    % w, q: Opinions (scalars or vectors) corresponding to x and y
    % N, NN: Number of individuals in each species
    % rx: Radius for spatial influence
    % rw: Radius for opinion influence
    
    % Reshape opinions for pairwise comparison
    wi = reshape(w, [N, 1]);  % Size: (N, 1)
    wj = reshape(q, [1, NN]); % Size: (1, NN)
    
    % Reshape positions for pairwise computation
    xi = reshape(x, [N, 1, size(x, 2)]); % Size: (N, 1, D)
    xj = reshape(y, [1, NN, size(y, 2)]); % Size: (1, NN, D)
    
    % Compute pairwise distances between positions
    incXij = xi - xj; % Size: (N, NN, D)
    distXij = sqrt(sum(incXij.^2, 3)); % Size: (N, NN)
    
    % Identify pairs satisfying both opinion and spatial thresholds
    isClosed = (abs(wi - wj) < rw) & (distXij < rx); % Size: (N, NN)
    %isClosed = (abs(wi - wj) < rw); % Size: (N, NN)
    
    % Compute the alignment factor
    phi = sum(isClosed .* (wj - wi), 2); % Size: (N, 1)
end

function forces = computePotentialTerm(x,dU)
    xi = x;
    xj = reshape(x', 1, size(x, 2), []); 
    incXiXj = xi - xj; 
    distXiXj = sqrt(sum(incXiXj.^2, 2));
    distXiXj(distXiXj == 0) = 1;
    forces = -dU(distXiXj) .* incXiXj ./ distXiXj;
    forces = squeeze(sum(forces, 3));    
end

function forces = computePotentialTermInterSpecie(x, y, dU)
    % x and y are inputs for positions of different species (sizes may differ)
    % dU is the derivative of the potential function
    
    % Expand x and y to match dimensions for pairwise distance computation
    xi = reshape(x, size(x, 1), 1, size(x, 2)); % Reshape xi to (Nx, 1, D)
    xj = reshape(y, 1, size(y, 1), size(y, 2)); % Reshape xj to (1, Ny, D)
    
    % Compute pairwise increments xi - xj
    incXiXj = xi - xj; % Resulting size: (Nx, Ny, D)
    
    % Compute pairwise distances
    distXiXj = sqrt(sum(incXiXj.^2, 3)); % Resulting size: (Nx, Ny)
    distXiXj(distXiXj == 0) = 1; % Avoid division by zero
    
    % Apply the derivative of the potential function
    potentialTerm = dU(distXiXj); % Evaluate dU at distances, size: (Nx, Ny)
    
    % Compute forces
    forces = -potentialTerm .* incXiXj ./ distXiXj; % Size: (Nx, Ny, D)
    
    % Sum over the second dimension (interaction sum for each particle in x)
    forces = squeeze(sum(forces, 2)); % Resulting size: (Nx, D)
end

function morsepotential = morsepotential(r, Cr, Ca, lr, la)
    morsepotential = -(Cr/lr)*exp(-r./lr) + (Ca/la)*exp(-r./la);
end

function Op_align = computeOpinionAlignmentPreference(v,w1,rw,vt,wt)
    isAligned = abs(w1 - wt) < rw;
    Op_align = isAligned.*(vt-v);
end

% function Op_align = computeOpinionAlignmentPreference(l,w1,rw,xt,wt)
%     isAligned = abs(w1 - wt) < rw;
%     Op_align = isAligned.*(xt-l);
% end

