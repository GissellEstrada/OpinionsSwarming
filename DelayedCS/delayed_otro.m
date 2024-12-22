
    %% Parameters
    N = 5;               % Number of agents
    beta = 0.1;            % Decay rate for communication
    t_final = 10;        % Final simulation time
    dt = 0.01;           % Time step for integration
    tau_max = 20;         % Maximum delay
%     phi = @(tau) exp(-tau); % Exponential kernel function

    phi_exponent = 1.5;
    phi = @(tau) 1/(1+tau.^phi_exponent); % Levy kernel
    %phi = @(tau) exp(-tau); % Levy kernel

    Ca = 1; la = 1; Cr = 1; lr = 0.5;
    rng(0);              % Seed for reproducibility

    %% Initial Conditions
    x0 = rand(3, N);     % Initial positions (3D)
    v0 = rand(3, N);     % Initial velocities (3D)
    z0 = [x0(:); v0(:)]; % Flattened initial state vector

    %% Solve ODE
    [t, z] = ode45(@(t, z) odesystem_with_morse(t, z, N, beta, tau_max, dt, phi, Ca, la, Cr, lr), [0 t_final], z0);

    %% Post-process and Plot Results
    x = reshape(z(:, 1:3*N)', [3, N, length(t)]);       % Extract positions (3D)
    v = reshape(z(:, 3*N+1:end)', [3, N, length(t)]);   % Extract velocities (3D)


%     % Plot Position Magnitude vs Time
%     figure;
%     hold on;
%     for i = 1:N
%         plot(t, squeeze(sqrt(x(1, i, :).^2 + x(2, i, :).^2 + x(3, i, :).^2)), 'LineWidth', 1.5); % Position Magnitude
%     end
%     xlabel('Time (t)'); ylabel('Magnitude of Position');
%     title('Position Magnitude Over Time');
%     legend(arrayfun(@(i) sprintf('Agent %d', i), 1:N, 'UniformOutput', false));
%     grid on;
%     hold off;




%%

% % Create VideoWriter Object
% video_filename = 'trajectories_movie.avi';
% video_writer = VideoWriter(video_filename);
% video_writer.FrameRate = 30;
% open(video_writer);
% 
% % Create a figure for plotting
% figure;
% hold on;
% axis equal;
% xlabel('X'); ylabel('Y');
% title('2D Trajectories of Agents');
% grid on;
% 
% % Plot and record movie frames
% for frame = 1:length(t)
%     cla; % Clear axes
%     for i = 1:N
%         % Plot the current positions of the agents
%         plot(squeeze(x(1, i, 1:frame)), squeeze(x(2, i, 1:frame)), 'LineWidth', 1.5); % Trajectory
%         scatter(squeeze(x(1, i, frame)), squeeze(x(2, i, frame)), 50, 'filled'); % Current position
%     end
%     % Add a legend (optional for clarity)
%     legend(arrayfun(@(i) sprintf('Agent %d', i), 1:N, 'UniformOutput', false), 'Location', 'bestoutside');
%     drawnow; % Update the figure
%     % Capture the frame for the video
%     frame_data = getframe(gcf);
%     writeVideo(video_writer, frame_data);
% end
% 
% % Close the video file
% close(video_writer);
% 
% disp('Movie saved as trajectories_movie.avi');



%%



   % Plot Trajectories
    figure;
    hold on;
    for i = 1:N
        plot(squeeze(x(1, i, :)), squeeze(x(2, i, :)), 'LineWidth', 1.5); % Plot trajectories
    end
    xlabel('x'); ylabel('y');
    title('Trajectories of Agents');
    %legend(arrayfun(@(i) sprintf('Agent %d', i), 1:N, 'UniformOutput', false));
    grid on;
    hold off;

    
    % Plot Velocity Magnitude vs Time
    figure;
    hold on;
    for i = 1:N
        plot(t, squeeze(sqrt(v(1, i, :).^2 + v(2, i, :).^2 + v(3, i, :).^2)), 'LineWidth', 1.5); % Velocity Magnitude
    end
    xlabel('Time (t)'); ylabel('Magnitude of Velocity');
    title('Velocity Magnitude Over Time');
    %legend(arrayfun(@(i) sprintf('Agent %d', i), 1:N, 'UniformOutput', false));
    grid on;
    hold off;

    % Plot 3D Trajectories (Position vs Time)
    figure;
    hold on;
    for i = 1:N
        plot3(squeeze(x(1, i, :)), squeeze(x(2, i, :)), squeeze(x(3, i, :)), 'LineWidth', 1.5); % 3D Position
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('3D Trajectories of Agents');
    %legend(arrayfun(@(i) sprintf('Agent %d', i), 1:N, 'UniformOutput', false));
    grid on;
    hold off;

    % Plot 3D Velocity Components (Velocity vs Time)
    figure;
    hold on;
    for i = 1:N
        % Plot the 3 components of the velocity (x, y, z) as separate curves
        plot3(squeeze(v(1, i, :)), squeeze(v(2, i, :)), squeeze(v(3, i, :)), 'LineWidth', 1.5);
    end
    xlabel('Time (t)');
    ylabel('Velocity X');
    zlabel('Velocity Y');
    title('Velocity Components of Agents in 3D');
    %legend(arrayfun(@(i) sprintf('Agent %d', i), 1:N, 'UniformOutput', false));
    grid on;
    hold off;
  

%% System of ODEs with Morse Potential
function dzdt = odesystem_with_morse(t, z, N, beta, tau_max, dt, phi, Ca, la, Cr, lr)
    % Extract Positions and Velocities
    x = reshape(z(1:3*N), [3, N]);   % Positions (3D)
    v = reshape(z(3*N+1:end), [3, N]); % Velocities (3D)

    % Initialize Derivatives
    dxdt = v;                % dx/dt = v
    dvdt = zeros(size(v));   % dv/dt = 0 initially

    % Loop over agents
    for i = 1:N
        for j = 1:N
            if i ~= j
                % Trapezoidal rule for the integral
                tau_values = 0:dt:tau_max; % Discretize tau
                integral_term = zeros(3, 1); % Initialize integral
                for k = 1:length(tau_values)
                    tau = tau_values(k);

                    % Past positions and velocities
                    x_i_tau = delayed_state(t - tau, z, N, i, 'x');
                    x_j_tau = delayed_state(t - tau, z, N, j, 'x');
                    v_i_tau = delayed_state(t - tau, z, N, i, 'v');
                    v_j_tau = delayed_state(t - tau, z, N, j, 'v');

%                     x_i_tau = delayed_state(t, z, N, i, 'x');
%                     x_j_tau = delayed_state(t, z, N, j, 'x');
%                     v_i_tau = delayed_state(t, z, N, i, 'v');
%                     v_j_tau = delayed_state(t, z, N, j, 'v');

                    % Compute Communication Rate H
                    d = norm(x_i_tau - x_j_tau)^2;
                    H = 1 / (1 + d)^beta;

                    % Compute the integrand
                    integrand = phi(tau) * H * (v_j_tau - v_i_tau);

                    % Apply Trapezoidal Rule Weight
                    if k == 1 || k == length(tau_values)
                        weight = 0.5; % Half weight at boundaries
                    else
                        weight = 1;
                    end
                    integral_term = integral_term + weight * integrand * dt;
                end
                dvdt(:, i) = dvdt(:, i) + (1 / N) * integral_term; % Add interaction term
            end
        end


        % Add the Morse potential term (no delay)
        for j = 1:N
            if i ~= j
                % Distance between agents
                r = norm(x(:, i) - x(:, j));
                % Compute the force using the Morse potential
                force_morse = morse_force(r, Cr, Ca, lr, la);
                % Update the velocity using the force (No delay term here)
                dvdt(:, i) = dvdt(:, i) + (1 / N) * force_morse;
            end
        end
    end

    % Combine Derivatives
    dzdt = [dxdt(:); dvdt(:)];
end

%% Delayed State Function
function delayed_state = delayed_state(tau, z, N, agent_idx, var_type)
    if tau < 0
        delayed_state = zeros(3, 1); % If time is in the past, return zeros
        return;
    end

    % Extract current positions/velocities
    if strcmp(var_type, 'x')
        state = z(1:3*N); % Positions (3D)
    elseif strcmp(var_type, 'v')
        state = z(3*N+1:end); % Velocities (3D)
    else
        error('Invalid variable type: Choose "x" or "v"');
    end

    % Reshape and extract the desired agent's state
    state = reshape(state, [3, N]);
    delayed_state = state(:, agent_idx);
end

%% Morse Potential Force
% function force = morse_force(r, D_e, r_e, alpha)
%     % Compute the force from the Morse potential
%     V = D_e * (2 * exp(-alpha * (r - r_e)) - exp(-2 * alpha * (r - r_e)));
%     force = -V * (r - r_e) / r; % Force is the negative gradient of potential
% end


function force = morse_force(r, Cr, Ca, lr, la)
    force = -(Cr/lr)*exp(-r./lr) + (Ca/la)*exp(-r./la);
end


