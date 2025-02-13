clear all;
close all;
format long

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
NL = 90;                             % number of individuals from each group
Ca = 1; la = 1; Cr = 0.6; lr = 0.5; % attraction-repulsion parameters
%Ca = 1; la = 1; Cr = 0.6; lr = 0.5;
T = 60;                             % final time
dt = 1.0e-2;                        % timestep
M = floor(T/dt);                    % number of time steps
alpha = 1; beta = 0.5;              % friction parameters
N = NL;                   % total number of individuals

%% Changing taur and taub gives very rich dynamics

% Initial conditions
x0L = -1 + 2*rand(NL,2); v0L = -1 + 2*rand(NL,2);

x0 = x0L;
v0 = v0L;

% Initial condition for Adams-Bashforth
x = zeros(2*N,2,M);
x(:,:,1) = [x0;v0]; % Initial state
[F1x] = F(x(:,:,1), NL, alpha, beta, Cr, Ca, lr, la); % Initial F value

% Use Euler for the first step
x(:,:,2) = x(:,:,1) + dt * F1x; 
% Compute F for the second step (needed for Adams-Bashforth)
[F2x] = F(x(:,:,2), NL, alpha, beta, Cr, Ca, lr, la);

% Adams-Bashforth 2-step method
for i = 3:M
    % Adams-Bashforth 2-step formula
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * F2x - F1x);  

    % Update F values for the next step
    F1x = F2x;  % F from previous step
 
    [F2x] = F(x(:,:,i), NL, alpha, beta, Cr, Ca, lr, la);  % F for current step
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')
        error('NaN or Inf encountered at time step %d', i);
    end
end

% Plot solution at last instant in time


figure;
hold on;
for i = 1:N
   plot(squeeze(x(i, 1, 5998:5999)),  squeeze(x(i, 2, 5998:5999)), 'r'); 
end
hold on

for i = 1:N
   plot(squeeze(x(i, 1, 5999:end)),  squeeze(x(i, 2, 5999:end)), 'b');
end



% % plots uninformed (preference towards -1) in red and leaders and followers
% % (preference towards 1) in blue
quiver(x(1:N,1,end),x(1:N,2,end),x(1:N,1,end),x(1:N,2,end),'r');


% figure()
% quiver(x(1:N,1,:),x(1:N,2,:),x(N+1:N+N,1,:),x(N+1:N+N,2,:),'r');
% quiver(x(N+1:N,1,end),x(N+1:N,2,end),x(N+N+1:end,1,end),x(N+N+1:end,2,end),'b');



% -------------------PROBLEM-RELATED FUNCTIONS------------------- %

function [Fx] = F(x, NL, alpha, beta, Cr, Ca, lr, la)
    N = NL;
    l = x(1:NL,:);
    v = x(N+1:(N+NL),:);
    Fx = [v; ...
          compute_morse_acceleration(l, v, Ca, la, Cr, lr, NL, alpha, beta)];
    
end

function accel = compute_morse_acceleration(x, v, Ca, la, Cr, lr, N, alpha, beta)
    accel = zeros(N, 2);
    for i = 1:N
        interaction_sum = zeros(1, 2);
        for j = 1:N
            if i ~= j
                dist_ij = norm(x(i, :) - x(j, :));
                % Compute the gradient of the Morse potential
                force = (-(Cr/lr) * exp(- (dist_ij)/lr) + (Ca/la) * exp(- (dist_ij)/la)) * (x(i, :) - x(j, :)) / dist_ij;
                interaction_sum = interaction_sum - force; % Subtract to account for the direction
            end
        end
        % Compute the friction term: (alpha - beta * |v|^2) * v
        v_magnitude_sq = norm(v(i, :))^2;
        friction = (alpha - beta * v_magnitude_sq) * v(i, :);
        
        % Total acceleration is the sum of the interaction and friction terms
        accel(i, :) = interaction_sum + friction;
        
    end
end







