
% Hegselmann-Krause model for opinion dynamics

% Parameters
N = 100;            % Number of agents
T = 100;            % Number of iterations
epsilon = 0.1;      % Communication threshold
delta = 0.01;       % Tolerance for convergence
xmin = 0; xmax = 10;   % x-coordinate bounds
ymin = 0; ymax = 10;   % y-coordinate bounds

% Initialize agents' opinions randomly
opinions = rand(N, 1) * (ymax - ymin) + ymin;

% Generate random initial positions for agents
positions = rand(N, 2) .* [(xmax-xmin) (ymax-ymin)];
positions(:,1) = positions(:,1) + xmin;

% Main loop
for t = 1:T
    updated_opinions = zeros(N, 1);
    for i = 1:N
        neighbors = find(abs(positions(:,2) - positions(i,2)) < epsilon);
        updated_opinions(i) = mean(opinions(neighbors));
    end
    opinions = updated_opinions;
    
    % Check for convergence
    if max(abs(diff(opinions))) < delta
        disp(['Converged at iteration ' num2str(t)]);
        break;
    end
end

% Plot final state
scatter(positions(:,1), positions(:,2), 50, opinions, 'filled');
colorbar;
xlabel('X-coordinate');
ylabel('Y-coordinate');
title('Opinion Dynamics in 2D Space');