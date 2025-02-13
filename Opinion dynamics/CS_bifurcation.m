
% Cucker-Smale System using Adams-Bashforth Method with Morse Potential
clear; 

% Number of agents
N = 100;

% Dimension of space (e.g., 2 for 2D)
d = 2;

% Morse potential parameters
D_e = 1.0;    % Depth of the potential well (interaction strength)
alpha = 0.5;  % Range parameter (controls decay)
r_e = 0;    % Equilibrium distance

Ca = 1;
Cr = 0.5;
la = 1;
lr = 0.5;

% Friction term parameters
alpha_friction = 1;  % Linear friction coefficient
beta_friction = 0.5;  % Nonlinear friction coefficient

% Initial positions (random)
 x_position = rand(N, d);
% save('x_position.mat','x_position')
%load('x_position.mat');

x = x_position;

% Initial velocities (random)
v = rand(N, d);

% Time parameters
T = 60;       % Total simulation time
dt = 0.01;    % Time step
num_steps = T / dt;

% Initialize positions and velocities
x_current = x;
v_current = v;

% Preallocate arrays to store solutions
x_history = zeros(N, d, num_steps);
v_history = zeros(N, d, num_steps);

% Store initial positions and velocities
x_history(:, :, 1) = x_current;
v_history(:, :, 1) = v_current;

% Adams-Bashforth requires previous step(s), so use Euler for the first step
% Compute initial acceleration using Morse potential
accel_prev = compute_morse_acceleration(x_current, v_current, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);

% Perform Euler step for the first time step
v_next = v_current + dt * accel_prev;
x_next = x_current + dt * v_current;

% Store the results of the first step
x_history(:, :, 2) = x_next;
v_history(:, :, 2) = v_next;

% Compute the acceleration for the next step
accel_current = compute_morse_acceleration(x_next, v_next, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);

% Adams-Bashforth 2-step method loop
for t = 2:num_steps - 1
    % Adams-Bashforth 2-step update for velocities
    v_next = v_history(:, :, t) + dt/2 * (3 * accel_current - accel_prev);
    
    % Update positions using the updated velocities
    x_next = x_history(:, :, t) + dt/2 * (3 * v_history(:, :, t) - v_history(:, :, t - 1));
    
    % Store the new positions and velocities
    x_history(:, :, t + 1) = x_next;
    v_history(:, :, t + 1) = v_next;
    
    % Update the accelerations for the next step
    accel_prev = accel_current;
    accel_current = compute_morse_acceleration(x_next, v_next, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);
end

% Plot the trajectories of agents

t_frac = t(1000:end);

figure;
hold on;
for i = 1:N
    
   plot(squeeze(x_history(i, 1, 5998:5999)),  squeeze(x_history(i, 2, 5998:5999)), 'r'); 
   %quiver(x(1:N,1,end),x(1:N,2,end),x(N+1:end,1,end),x(N+1:end,2,end));
   
   
end
hold on

for i = 1:N
    
   plot(squeeze(x_history(i, 1, 5999:end)),  squeeze(x_history(i, 2, 5999:end)), 'b'); 
   %quiver(x(1:N,1,end),x(1:N,2,end),x(N+1:end,1,end),x(N+1:end,2,end));
   
   
end


quiver(x_history(1:N,1,end),x_history(1:N,2,end),x_history(1:N,1,end),x_history(1:N,2,end));
xlabel('x');
ylabel('y');
title('Trajectories of Agents in Cucker-Smale System with Morse Potential');
grid on;
hold off;

% Function to compute acceleration based on the Morse potential interaction
function accel = compute_morse_acceleration(x, v, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction)
    accel = zeros(N, d);
    for i = 1:N
        interaction_sum = zeros(1, d);
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
        friction = (alpha_friction - beta_friction * v_magnitude_sq) * v(i, :);
        
        % Total acceleration is the sum of the interaction and friction terms
        accel(i, :) = interaction_sum + friction;
        %accel(i, :) = interaction_sum;
    end
end
