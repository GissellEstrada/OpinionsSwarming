
% Cucker-Smale System using Adams-Bashforth Method with Morse Potential
clear; 

% Number of agents
N = 50;

% Dimension of space (e.g., 2 for 2D)
d = 2;

% Morse potential parameters
D_e = 1.0;    % Depth of the potential well (interaction strength)
alpha = 0.5;  % Range parameter (controls decay)
r_e = 0;    % Equilibrium distance

Ca = 1;
Cr = 0.6;
la = 1;
lr = 0.5;

% Friction term parameters
alpha_friction = 1;  % Linear friction coefficient
beta_friction = 0.5;  % Nonlinear friction coefficient

% Initial positions (random)
x = rand(N, d);

% Initial velocities (random)
v = rand(N, d);




% Time parameters
T = 20;       % Total simulation time
dt = 0.01;    % Time step
num_steps = T / dt;

% Initialize positions and velocities
x_current = x;
v_current = v;
w_current = w;



% Initial opinions (random)
L = 30;                % computational domain (opinion space)
w = zeros(N, num_steps);
w(:,1) = sort(L*rand(1,N));

% Preallocate arrays to store solutions
x_history = zeros(N, d, num_steps);
v_history = zeros(N, d, num_steps);
w = zeros(N, num_steps);

for k = 1:N-1
    Psi_aa = opinion1(w,N);
    w(:,k+1) = w(:,k) + dt*Psi_aa(:)/N; 
    
end


% Store initial positions and velocities
x_history(:, :, 1) = x_current;
v_history(:, :, 1) = v_current;
w_history(:, 1) = w_current;

% Adams-Bashforth requires previous step(s), so use Euler for the first step
% Compute initial acceleration using Morse potential
accel_prev = compute_morse_acceleration(x_current, v_current, w_current, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);
pure_opinion = compute_pure_opinion(w_current, x_current, N);


% Perform Euler step for the first time step
w_next = w_current' + dt * pure_opinion;
v_next = v_current + dt * accel_prev;
x_next = x_current + dt * v_current;

% Store the results of the first step
x_history(:, :, 2) = x_next;
v_history(:, :, 2) = v_next;
w_history(:, 1) = w_next;

% Compute the acceleration for the next step
accel_current = compute_morse_acceleration(x_next, v_next, w_next, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);





% Adams-Bashforth 2-step method loop
for t = 2:num_steps - 1
    % Adams-Bashforth 2-step update for velocities
    v_next = v_history(:, :, t) + dt/2 * (3 * accel_current - accel_prev);
    
    % Update positions using the updated velocities
    x_next = x_history(:, :, t) + dt/2 * (3 * v_history(:, :, t) - v_history(:, :, t - 1));

    % Update opinion using the updated velocities
    w_next = w_history(:, t) + dt/2 * (3 * w_history(:, t) - w_history(:, t - 1));
    
    
    % Store the new positions and velocities
    x_history(:, :, t + 1) = x_next;
    v_history(:, :, t + 1) = v_next;
    w_history(:, t + 1) = w_next;
    
    % Update the accelerations for the next step
    accel_prev = accel_current;
    accel_current = compute_morse_acceleration(x_next, v_next, w_next, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction);
end

% Plot the trajectories of agents

t_frac = t(1000:end);

figure;
hold on;
for i = 1:N
    plot(squeeze(x_history(i, 1, 150:200)), squeeze(x_history(i, 2, 150:200)));
end
xlabel('x');
ylabel('y');
title('Trajectories of Agents in Cucker-Smale System with Morse Potential');
grid on;
hold off;

for i = 1:N
    plot(w_history(i, 1:10)); hold on
end



% figure;
% hold on;
% for i = 1:N
%     plot(squeeze(x_history(i, 1, :)), squeeze(x_history(i, 2, :)));
% end
% xlabel('x');
% ylabel('y');
% title('Trajectories of Agents in Cucker-Smale System with Morse Potential');
% grid on;
% hold off;

% Function to compute acceleration based on the Morse potential, friction
% term and opinions
function accel = compute_morse_acceleration(x, v, w, Ca, la, Cr, lr, r_e, N, d, alpha_friction, beta_friction)
    accel = zeros(N, d);
    for i = 1:N
        interaction_sum = zeros(1, d);
        for j = 1:N
            if i ~= j
                dist_ij = norm(x(i, :) - x(j, :));
                % Compute the gradient of the Morse potential
                force = (-(Cr/lr) * exp(- (dist_ij - r_e)/lr) + (Ca/la) * exp(- (dist_ij - r_e)/la)) * (x(i, :) - x(j, :)) / dist_ij;
                interaction_sum = interaction_sum - force; % Subtract to account for the direction

                opinion_term1 = opinion2(w,N) * (v(j,:) - v(i, :));
                %opinion_term2 = opinion2(w,N) * (v(i, :) - ones(1,2));
            end
        end

        % Compute the friction term: (alpha - beta * |v|^2) * v
        v_magnitude_sq = norm(v(i, :))^2;
        friction = (alpha_friction - beta_friction * v_magnitude_sq) * v(i, :);
        
        % Total acceleration is the sum of the interaction and friction terms
        accel(i, :) = interaction_sum + friction + opinion_term1(i,:);
        %accel(i, :) = interaction_sum;
    end
end


% Function for the opinion. Bounded confidence type
function Op1 = opinion1(w,N) % function to evaluate grad W (wi-wj)
Op1 = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if i ~= j %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(w(i)-w(j))<1
                    
                    Op1(i) = Op1(i) +  (w(j)-w(i));

                else
                    Op1(i) = Op1(i);
                end
                 %
         
            end
        end
    end    
end



function Op2 = opinion2(w,N) % function to evaluate grad W (wi-wj)
Op2 = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if i ~= j %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(w(i)-w(j))>1 && abs(w(i)-w(j))<2
                    
                    Op2(i) = Op2(i) +  (w(j)-w(i));

                else
                    Op2(i) = Op2(i);
                end
                 %
         
            end
        end
    end    
end


function PO = compute_pure_opinion(w,x,N) % function to evaluate grad W (wi-wj)
PO = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if i ~= j %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(w(i)-w(j))<1 && abs(x(i)-x(j))<1
                    
                    PO(i) = PO(i) +  (w(j)-w(i));

                else
                    PO(i) = PO(i);
                end
                 %
         
            end
        end
    end    
end
