% Flocking model with Morse potential, scaled by N, with friction term
% Parameters
N = 50;                   % Number of agents
d = 2;                    % Dimension (2D space)
T = 10;                   % Total simulation time
dt = 0.01;                % Time step
steps = T/dt;             % Number of time steps

% Morse potential parameters
D_e = 0.5;                % Depth of the potential well (attraction strength)
a = 1.0;                  % Decay rate of attraction
b = 0.5;                  % Decay rate of repulsion
r0 = 1.0;                 % Equilibrium distance

% Friction coefficient
sigma = 0.1;              % Damping factor

% Initial conditions
x = rand(d, N) * 10;      % Initial positions
v = rand(d, N) * 2 - 1;   % Initial velocities

% Main simulation loop
for t = 1:steps
    % Initialize force matrix
    force_matrix = zeros(d, N);
    
    % Compute pairwise distances and forces
    for i = 1:N
        for j = 1:N
            if i ~= j
                % Distance and direction between agents i and j
                r_ij = norm(x(:, j) - x(:, i));      % Distance between i and j
                r_vec = (x(:, j) - x(:, i)) / r_ij;  % Unit vector from i to j

                % Morse potential force calculation
                F_repulsion = b * exp(-b * (r_ij - r0));        % Repulsive component
                F_attraction = -a * exp(-a * (r_ij - r0));      % Attractive component
                F_ij = (F_repulsion + F_attraction) * r_vec;

                % Accumulate force contributions, scaled by 1/N
                force_matrix(:, i) = force_matrix(:, i) + F_ij / N;
            end
        end
    end
    
    % Friction term, potentially scaled by 1/N
    friction_term = -sigma * v / N;
    
    % Update velocities with Morse forces and friction
    v_update = force_matrix + friction_term;
    v = v + dt * v_update;
    
    % Update positions based on velocities
    x = x + dt * v;
    
    % (Optional) Plotting for visualization
    if mod(t, 100) == 0 || t == steps
        clf; hold on;
        plot(x(1, :), x(2, :), 'bo', 'MarkerFaceColor', 'b'); % Plot positions
        quiver(x(1, :), x(2, :), v(1, :), v(2, :), 0.5, 'r'); % Plot velocities
        xlim([0 10]); ylim([0 10]);
        title(['Time: ', num2str(t * dt)]);
        pause(0.05);
    end
end