clear all;
close all;

%% -------------------READ ME-----------------------

% The "cube" x stores both spatial coordinates and velocities for each instant in time:
% x = [spatial coord; velocity coord];
% The indexing for each kind of velocity and spatial coord. can be found in
% the line where I plot the solution
% The vector w stores opinions from each subpopulation:
% w = [op];

% The only place where I multiply by the factor 1/n is in the sumphi term,
% which is only found in the equation regarding w

% All the parameters used in the model are stated and described right below

%----------------------------------------------%


%% Problem data

% alpha = 1; beta = 0.5;                                % friction parameters
% c_att = 100; l_att = 1; c_rep = 60; l_rep = 0.5;      % clusters moving in opposite direction (rotating)
% c_att = 100; l_att = 1; c_rep = 50; l_rep = 0.5;      % circle formation
% c_att = 100; l_att = 1; c_rep = 40; l_rep = 6;        % compact clusters moving in the same direction (rotating)
% c_att = 100; l_att = 1; c_rep = 50; l_rep = 1.2;      % compact clusters moving in the same direction (rotating)
% c_att = 50; l_att = 1; c_rep = 60; l_rep = 0.5;       % rotating mill
% c_att = 1; l_att = 1; c_rep = 2; l_rep = 0.5;
% c_att = 100; l_att = 1; c_rep = 60; l_rep = 0.7;      % (small) clusters moving in opposite direction (rotating)

% Changing tau_red and tau_blue gives very rich dynamics

alpha = 1; beta = 5;
c_att = 100; l_att = 1.2;
c_rep = 350; l_rep = 0.8;

t_final = 100;                      % final time
dt = 1.0e-2;                        % timestep
steps = floor(t_final/dt);          % number of time steps

n = 100;                            % total number of individuals
r_x = 0.5;                          % spatial radius for alignment in velocity
r_w = 1;                            % preference radius for alignment il velocity
tau_red = 0.1;                      % strength of preference towards -1, red target
tau_blue = 0.1;                     % strength of preference towards 1, blue target

% Initial conditions
rng(1234);

x_0 = -1 + 2*rand(n,2);     % n*2, initial coordinates
v_0 = -1 + 2*rand(n,2);
w_0 = -1 + 2*rand(n,1);     % n*1, initial values

dU = @(r) morse_potential(r, c_rep, c_att, l_rep, l_att);

% Initial condition for Adams-Bashforth
x = zeros(2*n, 2, steps);   % 2n*2 vector for each step, first n rows for x, rest for v
x(:,:,1) = [x_0; v_0];      % Initial state
w(:,1) = w_0;
[F1x, F1w] = F(x(:,:,1), w(:,1), n, alpha, beta, dU, r_x, r_w, tau_red, tau_blue); % Initial F value

% Use Euler for the first step
x(:,:,2) = x(:,:,1) + dt * F1x; 
w(:,2) = w(:,1) + dt * F1w; 

% Compute F for the second step (needed for Adams-Bashforth)
[F2x, F2w] = F(x(:,:,2), w(:,2), n, alpha, beta, dU, r_x, r_w, tau_red, tau_blue);

% Adams-Bashforth 2-step method
for i = 3:steps
    % Adams-Bashforth 2-step formula
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * F2x - F1x);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * F2w - F1w);
    % Update F values for the next step
    F1x = F2x;  % F from previous step
    F1w = F2w;
    [F2x, F2w] = F(x(:,:,i), w(:,i), n, alpha, beta,dU, r_x, r_w, tau_red, tau_blue);  % F for current step
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')
        error('NaN or Inf encountered at time step %d', i);
    end
end


%% Plot final velocity configuration

% Create the folder if it doesn't exist

outputFolder = 'Figures_onePopNoMill';      % Folder to save the figures
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Plot
figure()
    plot(x(1:n,1,end), x(1:n,2,end), 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b','LineWidth', 1);  % Plot positions as circles
    hold on;

    quiver(x(1:n,1,end), x(1:n,2,end), x(n+1:2*n,1,end), x(n+1:2*n,2,end), 'r','LineWidth', 1); % Plot velocities as arrows

    xlabel('X');
    ylabel('Y');

    grid on;
    title('Velocities at the final time');

    hold off;
    axis equal;
    set(gca,'FontSize',16)
    % filename1 = fullfile(outputFolder, ['FinalVelocityNoMillNoPreference_NL', num2str(n),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_tau_blue_',num2str(tau_blue),'_tau_red_',num2str(tau_red), '.fig']);
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
            x(n+1:2*n,1,t), x(n+1:2*n,2,t), zeros(n,1), 'r');
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

% filename2 = fullfile(outputFolder, ['VelocityOverTime_NL', num2str(n),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_tau_blue_',num2str(tau_blue),'_tau_red_',num2str(tau_red), '.fig']);
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

% filename3 = fullfile(outputFolder, ['PositionOverTime_NL', num2str(n),'_alpha_',num2str(alpha),'_beta_',num2str(beta),'_Ca_',num2str(c_att),'_la_',num2str(l_att),'_Cr_',num2str(c_rep),'_lr_',num2str(l_rep),'_rx_',num2str(r_x),'_rw_',num2str(r_w),'_tau_blue_',num2str(tau_blue),'_tau_red_',num2str(tau_red), '.fig']);
% saveas(gcf, filename3);

figure();
    plot(1:steps,w, 'k','LineWidth', 1);
    xlabel('Time');
    ylabel('Opinion');
    title('Opinion Over Time');
    set(gca,'FontSize',16)
    % 
    % filename4 = fullfile(outputFolder, ['OpinionOverTimeNoMillNoPreference', '.fig']);
    % saveas(gcf, filename4);



%% -------------------PROBLEM-RELATED FUNCTIONS------------------- %

function [dx,dw] = F(x, w, n, alpha, beta, dU, r_x, r_w, tauL, tauR)
    n = n;
    l = x(1:n,:);
    v = x(n+1:(n+n),:);
    wu = w;

    vR = 1;
    wR = 1; 
    xR = 1;

    vL = -1;
    wL = -1; 
    xL = -1;

    term1     = 1/n * computePotentialTerm(l,dU); % there is 1/n still missing here
    term2     = (alpha - beta*sum(v.^2,2)).*v;

    termLeft  = tauL*computeOpinionAlignmentPreference(v,wu,r_w,vL,wL);
    termRight = tauR*computeOpinionAlignmentPreference(v,wu,r_w,vR,wR);

%     termLeft  = tauL*computeOpinionAlignmentPreference(l,wu,r_w,xL,wL);
%     termRight = tauR*computeOpinionAlignmentPreference(l,wu,r_w,xR,wR);

    termVelAlign = 1/n * computeVelocityAligment(l,wu,v,n,r_x,r_w);

    dv = term1 + term2 + termLeft + termRight + 0*termVelAlign;
    dx = [v; dv];

    phi = computeOpinionAligment(l, wu, n, r_x, r_w);
    dw = 1/n * phi - tauL * (wu - wL) - tauR * (wu - wR); % notice the 1/n here
       
end


function phi = computeOpinionAligment(x,w,n,r_x,r_w)
    wi = repmat(w, 1, n);
    wj = repmat(w', n, 1);    
    xi = reshape(x, [n, 1, size(x, 2)]);
    xj = reshape(x, [1, n, size(x, 2)]);
    incXij  = xi - xj;
    distWij = sqrt(sum((incXij).^2, 3));   
    isClosed = (abs(wi - wj) < r_w) & (distWij < r_x);
    %isClosed = (abs(wi - wj) < r_w);
    phi      = sum(isClosed .* (wj - wi), 2);
end


function U = computeVelocityAligment(x,w,v,n,r_x,r_w) % this is velocity alignment as in Cucker-Smale
    wi = repmat(w, 1, n);
    wj = repmat(w', n, 1);    
    xj = reshape(x, [n, 1, size(x, 2)]);
    xi = reshape(x, [1, n, size(x, 2)]);
    vj = reshape(v, [n, 1, size(v, 2)]);
    vi = reshape(v, [1, n, size(v, 2)]);
    incXij  = xi - xj;
    distWij = sqrt(sum((incXij).^2, 3));   
    isClosed = (abs(wi - wj) < r_w) & (distWij < r_x);  
    velDiff = vj - vi; % Pairwise velocity difference, 150x150x2
    U = squeeze(sum(isClosed .* velDiff, 2));
    %U      = sum(isClosed .* (vj - vi), 2);
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


function morse_potential = morse_potential(r, c_rep, c_att, l_rep, l_att)
    morse_potential = -(c_rep/l_rep)*exp(-r./l_rep) + (c_att/l_att)*exp(-r./l_att);
end


function Op_align = computeOpinionAlignmentPreference(v,w1,r_w,vt,wt)
    isAligned = abs(w1 - wt) < r_w;
    Op_align = isAligned.*(vt-v);
end


% function Op_align = computeOpinionAlignmentPreference(l,w1,r_w,xt,wt) 
%     isAligned = abs(w1 - wt) < r_w;
%     Op_align = isAligned.*(xt-l);
% end




