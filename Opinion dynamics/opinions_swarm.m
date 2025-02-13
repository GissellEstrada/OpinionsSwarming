clear all;
close all;

T = 60;              % final time
dt = 1.0e-2;          % timestep
num_steps = floor(T/dt);      % number of time steps
L = 10;                % computational domain (opinion space)

N = 90;              % individuals with preference for a (opinion is 1)
mu = 0.05;                   
mu1 = 0;           % strength of b towards opinion 1

%% Swarm parameters
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
 x = rand(N, d);
% save('x_position.mat','x_position')
%load('x_position.mat');
%x = x_position;

% Initial velocities (random)
v = rand(N, d);

% Initialise positions and speeds randomly
Icon = 1;            % starting point of trajectories from -1 to 1
w = (-1 + (1+1)*sort(rand(1,N)));

% Initialize positions and velocities
x_current = x;
v_current = v;
w_current = w';

% Preallocate arrays to store solutions
x_history = zeros(N, d, num_steps);
v_history = zeros(N, d, num_steps);
w_history = zeros(N,num_steps); 

% Store initial positions and velocities
x_history(:, :, 1) = x_current;
v_history(:, :, 1) = v_current;
w_history(:,1) = w_current; 



% Adams-Bashforth requires previous step(s), so use Euler for the first step
% Compute initial acceleration using Morse potential
accel_prev = compute_morse_acceleration(x_current, v_current, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);
Psi_prev = psi1(w_current,N);
opinion_veloc_prev = Op(w_current,x_current,v_current,N);

% Perform Euler step for the first time step
v_next = v_current + dt * accel_prev + dt * opinion_veloc_prev;
x_next = x_current + dt * v_current;
w_next = w_current + mu * dt * Psi_prev + dt * mu1 *(-w_current+ones(N,1));

% Store the results of the first step
x_history(:, :, 2) = x_next;
v_history(:, :, 2) = v_next;
w_history(:,2) = w_next;

% Compute the acceleration for the next step
accel_current = compute_morse_acceleration(x_next, v_next, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);
Psi_current = psi1(w_next,N);
opinion_veloc_current = Op(w_next,x_next,v_next,N);



% Adams-Bashforth 2-step method loop
for t = 2:num_steps - 1
    % Adams-Bashforth 2-step update for velocities
    v_next = v_history(:, :, t) + dt/2 * (3 * accel_current - accel_prev) + dt/2 * (3 * opinion_veloc_current - opinion_veloc_prev);
    
    % Update positions using the updated velocities
    x_next = x_history(:, :, t) + dt/2 * (3 * v_history(:, :, t) - v_history(:, :, t - 1));

     % Update opinion using the updated velocities
    w_next = w_history(:, t) + mu * dt/2 * (3 * Psi_current - Psi_prev) + mu1 * (dt/2)*(-w_history(:, t)+ones(N,1));
    
    % Store the new positions and velocities
    x_history(:, :, t + 1) = x_next;
    v_history(:, :, t + 1) = v_next;
    w_history(:, t + 1) = w_next;
    
    % Update the accelerations, opinion and acc+op for the next step
    accel_prev = accel_current;
    accel_current = compute_morse_acceleration(x_next, v_next, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);

    Psi_prev = Psi_current;
    Psi_current = psi1(w_next,N);

    opinion_veloc_prev = opinion_veloc_current;
    opinion_veloc_current = Op(w_next,x_next,v_next,N);

end


t_frac = t(1000:end);


%% Plot final trajectories
figure;

hold on;
for i = 1:N
   plot(squeeze(x_history(i, 1, end-4:end-2)),  squeeze(x_history(i, 2, end-4:end-2)), 'r'); 

end
% xlim([80 90])
% ylim([0,3])
hold on

for i = 1:N
   plot(squeeze(x_history(i, 1, end-2:end)),  squeeze(x_history(i, 2, end-2:end)), 'b'); 
  
end
% xlim([80 90])
% ylim([0,3])

%% Plot all trajectories
% hold on;
% for i = 1:N
%    plot(squeeze(x_history(i, 1, :)),  squeeze(x_history(i, 2, :))); 
% 
% end


%% Plot opinion
figure
plot(1:num_steps,w_history,'r'); hold on;



%save ('Trajectories_repulsive.mat','x');

function Psi = psi1(X,N) % function to evaluate grad W (Xi-Xj)
Psi = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if i ~= j %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(X(i)-X(j))<0.5
                    
                    Psi(i) = Psi(i) +  (X(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end
                 %
         
            end
        end
    end    
end

function Op_vel = Op(w,x,v,N) % function to evaluate grad W (Xi-Xj)
Op_vel = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if i ~= j %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(w(i)-w(j))<0.5 && abs(x(i)-x(j))<1
                    
                    Op_vel(i) = Op_vel(i) +  (v(j)-v(i));

                else
                    Op_vel(i) = Op_vel(i);
                end
                 %
         
            end
        end
    end    
end

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
        
    end
end







