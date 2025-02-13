% Non-dimensionalized Flocking Dynamics with Exponential Interaction
clear; clc; close all;

% Parameters for non-dimensionalization
alpha = 1;       % Original coefficient for self-propulsion
beta = 1;        % Coefficient for time scaling
mi = 1;          % Particle mass
Ca = 1;          % Characteristic attraction constant
Cr = 1;          % Characteristic repulsion constant
la = 1;          % Characteristic length for attraction
lr = 1;          % Characteristic length for repulsion

% Derived non-dimensional parameters
alpha_prime = (alpha * beta * la^2) / mi^2;
mi_prime = (mi^3) / (beta^2 * Ca * la^2);
C = Cr / Ca;
l_ratio = lr / la;

% Simulation parameters
N = 100;                  % Number of particles
dt_prime = 0.01;          % Non-dimensional time step
T_prime = 1000;           % Total non-dimensional simulation steps
v0_prime = 1;             % Non-dimensional initial velocity magnitude
interaction_radius_prime = 5;  % Non-dimensional interaction radius

% Initial conditions for non-dimensionalized positions and velocities
positions_prime = rand(N, 2);
angles = 2 * pi * rand(N, 1);
velocities_prime = v0_prime * [cos(angles), sin(angles)];



% Simulation loop
for t_prime = 1:T_prime
    new_positions_prime = positions_prime;
    new_velocities_prime = velocities_prime;
    
    % Calculate interactions and update velocities
    for i = 1:N
        force_total = [0, 0];
        
        for j = 1:N
            if i ~= j
                % Compute force due to interaction
                force_total = force_total + interaction_force(positions_prime, i, j, C, l_ratio);
            end
        end
        
        % Self-propulsion and friction terms
        vel_prime = velocities_prime(i, :);
        propulsion = alpha_prime * vel_prime;
        friction = -norm(vel_prime)^2 * vel_prime;
        
        % Update non-dimensional velocity
        new_velocities_prime(i, :) = vel_prime + dt_prime * (propulsion + friction + force_total / mi_prime);
        
        % Update non-dimensional position
        new_positions_prime(i, :) = positions_prime(i, :) + dt_prime * new_velocities_prime(i, :);
    end
    
    % Update all positions and velocities for next step
    positions_prime = new_positions_prime;
    velocities_prime = new_velocities_prime;
    
    % Plot the current state every 10 steps
    if mod(t_prime, 10) == 0
        clf;
        quiver(positions_prime(:, 1), positions_prime(:, 2), velocities_prime(:, 1), velocities_prime(:, 2), 0.5, 'b');
        axis([0 1 0 1]);  % Adjust axis limits as needed
        title(['Non-dimensional time step: ', num2str(t_prime)]);
        drawnow;
    end
end

% Function for non-dimensionalized potential
function force = interaction_force(positions_prime, i, j, C, l_ratio)
    diff_prime = positions_prime(j, :) - positions_prime(i, :);
    dist_prime = norm(diff_prime);
    if dist_prime == 0
        force = [0, 0];  % Avoid division by zero for identical positions
    else
        % Interaction force from Eq. (6)
        interaction_potential = -exp(-dist_prime) + C * exp(-dist_prime / l_ratio);
        force = (interaction_potential / dist_prime) * diff_prime;
    end
end
