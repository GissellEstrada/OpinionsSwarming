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
NL = 150;                             % number of individuals from each group (change the number N to get different dynamics)
%%
alpha = 1; beta = 0.5;                        % friction parameters
Ca = 100; la = 1; Cr = 60; lr = 0.5;          % clusters moving in opposite direction (rotating)
%Ca = 100; la = 1; Cr = 50; lr = 0.5;          % circle formation
%Ca = 100; la = 1; Cr = 40; lr = 6;            % compact clusters moving in the same direction (rotating)
%Ca = 100; la = 1; Cr = 50; lr = 1.2;          % compact clusters moving in the same direction (rotating)
%Ca = 100; la = 1; Cr = 50; lr = 1.2;
%Ca = 50; la = 1; Cr = 60; lr = 0.5;           % rotating mill
%Ca = 100; la = 1; Cr = 60; lr = 0.7;          % (small) clusters moving in opposite direction (rotating)
%%
% alpha = 0.1; beta = 5;
% Ca = 100; la = 1.2; Cr = 350; lr = 0.8;       % all moving in the same direction as a flock

%%
% alpha = 1; beta = 5;
% Ca = 100; la = 1.2; Cr = 350; lr = 0.8;

%%

T = 100;                             % final time
dt = 1.0e-2;                        % timestep
M = floor(T/dt);                    % number of time steps

N = NL;                   % total number of individuals
rx = 0.05;                             % spatial radius for alignment in velocity
rw = 0.05;                           % preference radius for alignment il velocity
taur = 0.05;                         % strenght of preference towards 1
taub = 0;                        % strenght of preference towards -1

%% Changing taur and taub gives very rich dynamics

% Initial conditions
x0L = -1 + 2*rand(NL,2); v0L = -1 + 2*rand(NL,2);
w0L = -1 + 2*rand(NL,1);

x0 = x0L;
v0 = v0L;
w0 = w0L;

dU = @(r) morsepotential(r, Cr, Ca, lr, la);

% Initial condition for Adams-Bashforth
x = zeros(2*N,2,M);
x(:,:,1) = [x0;v0]; % Initial state
w(:,1) = w0;
[F1x, F1w] = F(x(:,:,1), w(:,1), NL, alpha, beta, dU, rx, rw, taur, taub); % Initial F value

% Use Euler for the first step
x(:,:,2) = x(:,:,1) + dt * F1x; 
w(:,2) = w(:,1) + dt * F1w; 
% Compute F for the second step (needed for Adams-Bashforth)
[F2x, F2w] = F(x(:,:,2), w(:,2), NL, alpha, beta, dU, rx, rw, taur, taub);

% Adams-Bashforth 2-step method
for i = 3:M
    % Adams-Bashforth 2-step formula
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * F2x - F1x);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * F2w - F1w);
    % Update F values for the next step
    F1x = F2x;  % F from previous step
    F1w = F2w;
    [F2x, F2w] = F(x(:,:,i), w(:,i), NL, alpha, beta,dU, rx, rw, taur, taub);  % F for current step
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')
        error('NaN or Inf encountered at time step %d', i);
    end
end


%% Save figure

outputFolder = 'Figures';      % Folder to save the figures
% Create the folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


%% Plot of the final velocity configuration
figure()
plot(x(1:N,1,end), x(1:N,2,end), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');  % Plot positions as circles
hold on;

quiver(x(1:N,1,end), x(1:N,2,end), x(N+1:2*N,1,end), x(N+1:2*N,2,end), 'r'); % Plot velocities as arrows

xlabel('X');
ylabel('Y');

grid on;
title('Velocities at the final time');

hold off;
axis equal;
filename1 = fullfile(outputFolder, ['NoOpFinalVelocity_NL', num2str(NL),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(Ca),'_la_',num2str(la),'_Cr_',num2str(Cr),'_lr_',num2str(lr),'_rx_',num2str(rx),'_rw_',num2str(rw),'_taur_',num2str(taur),'_taub_',num2str(taub), '.fig']);
saveas(gcf, filename1);

%% Plot for several times

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
hold off;

filename2 = fullfile(outputFolder, ['NoOpVelocityOverTime_NL', num2str(NL),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(Ca),'_la_',num2str(la),'_Cr_',num2str(Cr),'_lr_',num2str(lr),'_rx_',num2str(rx),'_rw_',num2str(rw),'_taur_',num2str(taur),'_taub_',num2str(taub), '.fig']);
saveas(gcf, filename2);


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
hold off;

filename3 = fullfile(outputFolder, ['NoOpPositionOverTime_NL', num2str(NL),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(Ca),'_la_',num2str(la),'_Cr_',num2str(Cr),'_lr_',num2str(lr),'_rx_',num2str(rx),'_rw_',num2str(rw),'_taur_',num2str(taur),'_taub_',num2str(taub), '.fig']);
saveas(gcf, filename3);

 %%

% figure();
% plot(1:M,w, 'k');
% xlabel('Time');
% ylabel('Opinion');
% title('Opinion Over Time');
% 
% filename4 = fullfile(outputFolder, ['OpinionOverTime_', num2str(params), '.fig']);
% saveas(gcf, filename3);


% -------------------PROBLEM-RELATED FUNCTIONS------------------- %

function [dx,dw] = F(x, w, NL, alpha, beta, dU, rx, rw, tauL, tauR)
    N = NL;
    l = x(1:NL,:);
    v = x(N+1:(N+NL),:);
    wu = w;

    vR = 1;
    wR = 1;    

    vL = -1;
    wL = -1;   

    term1     = 1/N*computePotentialTerm(l,dU); % there is 1/N still missing here
    term2     = (alpha - beta*sum(v.^2,2)).*v;
    termLeft  = tauL*computeOpinionAlignmentPreference_left(v,wu,rw,vL,wL);
    termRight = tauR*computeOpinionAlignmentPreference_right(v,wu,rw,vR,wR);

    dv = term1 + term2 + termLeft + termRight;
    dx = [v; dv];

    phi = computeOpinionAligment(l, wu, NL, rx, rw);
    dw = 1/N*phi + tauL*(wL - wu) + tauR*(wR - wu); % notice the 1/N here
       
end

function phi = computeOpinionAligment(x,w,N,rx,rw)
    wi = repmat(w, 1, N);
    wj = repmat(w', N, 1);    
    xi = reshape(x, [N, 1, size(x, 2)]);
    xj = reshape(x, [1, N, size(x, 2)]);
    incXij  = xi - xj;
    distWij = sqrt(sum((incXij).^2, 3));   
    isClosed = (abs(wi - wj) < rw) & (distWij < rx);    
    phi      = sum(isClosed .* (wi - wj), 2);
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

function morsepotential = morsepotential(r, Cr, Ca, lr, la)
    morsepotential = -(Cr/lr)*exp(-r./lr) + (Ca/la)*exp(-r./la);
end

function Op_align = computeOpinionAlignmentPreference_left(v,w1,rw,vt,wt)
    if w1 < 0
       isAligned = w1;
       Op_align = isAligned.*(vt-v);
    else 
        isAligned = 0;
       Op_align = isAligned.*(vt-v);
    end
end

function Op_align = computeOpinionAlignmentPreference_right(v,w1,rw,vt,wt)
    if w1 > 0
       isAligned = w1;
       Op_align = isAligned.*(vt-v);
    else 
        isAligned = 0;
       Op_align = isAligned.*(vt-v);
    end
end




