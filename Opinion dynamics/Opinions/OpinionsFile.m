function OpinionsFile
% Parameters for types of "fish"
N1 = 5; % leader fishy
N2 = 6; % follower fishy
N3 = 10; % undecided fishy
N = N1 + N2 + N3; % Total number of "fish"
Nt = 5000; % Number of time steps
dt = 0.01; % Time step size
tlist = (0:Nt-1) * dt; % Time list

% Prefactors for each group of fish
prefac1 = dt / N1;
prefac2 = dt / N2;
prefac3 = dt / N3;

% Parameters for saving results
save_every = 50;
S = floor(Nt / save_every);

% Radius and other parameters
Rx = 2.0;
rx = 2.0;
rw = 2.0;

% Morse potential parameters
CR = 1.5;
LR = 0.25;
CA = 1.0;
LA = 0.75;

% Other constants
alpha = 1.0;
beta = 0.5;

% Directions to different targets
v_r = [1, 0];
v_b = [-1, 0];
tau_r = 0.9;
tau_b = 0.9;
muL = 0.001 * dt;
muF = 0.001 * dt;

% Initialize variables
z1 = zeros(S + 1, 5, N1);
z2 = zeros(S + 1, 5, N2);
z = zeros(S + 1, 5, N3);

% Leaders
preference1 = ones(1, N1);
position1 = rand(2, N1) * 2 - 1;  % Random positions between -1 and 1
velocity1 = rand(2, N1) * 2 - 1;  % Random velocities between -1 and 1

% Followers
preference2 = -1 * ones(1, N2);
position2 = rand(2, N2) * 2 - 1;  % Random positions between -1 and 1
velocity2 = rand(2, N2) * 2 - 1;  % Random velocities between -1 and 1

% Undecided
preference = zeros(1, N3);
position = rand(2, N3) * 2 - 1;  % Random positions between -1 and 1
velocity = rand(2, N3) * 2 - 1;  % Random velocities between -1 and 1

mean_vel = zeros(Nt + 1, 2);
dev_vel = zeros(Nt + 1, 1);
mean_pref = zeros(Nt + 1, 1);
dev_pref = zeros(Nt + 1, 1);

z(1, 1:2, :) = position;
z(1, 3:4, :) = velocity;
z(1, 5, :) = preference;

z1(1, 1:2, :) = position1;
z1(1, 3:4, :) = velocity1;
z1(1, 5, :) = preference1;

z2(1, 1:2, :) = position2;
z2(1, 3:4, :) = velocity2;
z2(1, 5, :) = preference2;

mean_vel(1, :) = (sum(velocity, 2) / N3) + (sum(velocity1, 2) / N1) + (sum(velocity2, 2) / N2);
dev_vel(1) = (sum((velocity(1, :) - mean_vel(1, 1)).^2 + (velocity(2, :) - mean_vel(1, 2)).^2) + ...
              sum((velocity1(1, :) - mean_vel(1, 1)).^2 + (velocity1(2, :) - mean_vel(1, 2)).^2) + ...
              sum((velocity2(1, :) - mean_vel(1, 1)).^2 + (velocity2(2, :) - mean_vel(1, 2)).^2)) / N;
mean_pref(1) = (sum(preference) + sum(preference1) + sum(preference2)) / N;
dev_pref(1) = (sum((preference - mean_pref(1)).^2) + sum((preference1 - mean_pref(1)).^2) + sum((preference2 - mean_pref(1)).^2)) / N;

count = 1;

for i = 2:Nt
    % Velocities squared
    vel = velocity(1, :).^2 + velocity(2, :).^2;
    vel1 = velocity1(1, :).^2 + velocity1(2, :).^2;
    vel2 = velocity2(1, :).^2 + velocity2(2, :).^2;

    % Difference in preference
    prefdiffUU = opinion_distance(preference, preference);  % undecided - undecided  
    prefdiffUL = opinion_distance(preference, preference1); % undecided - leader
    prefdiffUF = opinion_distance(preference, preference2); % undecided - followers
    prefdiffLL = opinion_distance(preference1, preference1); % leader - leader
    prefdiffFF = opinion_distance(preference2, preference2); % follower - follower
    prefdiffLF = opinion_distance(preference1, preference2); % leader - follower

    % Difference among all fish
    [diffUU, distUU] = euclidean_distance(position', position'); % undecided - undecided
    [diffUL, distUL] = euclidean_distance(position', position1'); % undecided - leader
    [diffUF, distUF] = euclidean_distance(position', position2'); % undecided - follower
    [diffLL, distLL] = euclidean_distance(position1', position1'); % leader - leader
    [diffFF, distFF] = euclidean_distance(position2', position2'); % follower - follower
    [diffLF, distLF] = euclidean_distance(position1', position2'); % leader - follower

    % Interaction calculations (if close enough in position and preference)
    interactionUU = double(abs(prefdiffUU) <= rw) .* (distUU <= rx);
    interactionUL = double(abs(prefdiffUL) <= rw) .* (distUL <= rx);
    interactionUF = double(abs(prefdiffUF) <= rw) .* (distUF <= rx);
    interactionLL = double(abs(prefdiffLL) <= rw) .* (distLL <= rx);
    interactionFF = double(abs(prefdiffFF) <= rw) .* (distFF <= rx);
    interactionLF = double(abs(prefdiffLF) <= rw) .* (distLF <= rx);

    go_to_r_U = double(preference <= 0);
    go_to_b_U = double(preference > 0);
    go_to_r_L = double(preference1 <= 0);
    go_to_b_L = double(preference1 > 0);
    go_to_r_F = double(preference2 <= 0);
    go_to_b_F = double(preference2 > 0);

    % Pre-calculations for the Morse potentials - R for repulsion, A for attraction
    prefacRUU = exp(-distUU / LR) ./ distUU;
    prefacAUU = exp(-distUU / LA) ./ distUU;
    prefacRUU(prefacRUU == Inf) = 0;
    prefacAUU(prefacAUU == Inf) = 0;

    prefacRUL = exp(-distUL / LR) ./ distUL;
    prefacAUL = exp(-distUL / LA) ./ distUL;
    prefacRUL(prefacRUL == Inf) = 0;
    prefacAUL(prefacAUL == Inf) = 0;

    prefacRUF = exp(-distUF / LR) ./ distUF;
    prefacAUF = exp(-distUF / LA) ./ distUF;
    prefacRUF(prefacRUF == Inf) = 0;
    prefacAUF(prefacAUF == Inf) = 0;

    prefacRLL = exp(-distLL / LR) ./ distLL;
    prefacALL = exp(-distLL / LA) ./ distLL;
    prefacRLL(prefacRLL == Inf) = 0;
    prefacALL(prefacALL == Inf) = 0;

    prefacRFF = exp(-distFF / LR) ./ distFF;
    prefacAFF = exp(-distFF / LA) ./ distFF;
    prefacRFF(prefacRFF == Inf) = 0;
    prefacAFF(prefacAFF == Inf) = 0;

    prefacRLF = exp(-distLF / LR) ./ distLF;
    prefacALF = exp(-distLF / LA) ./ distLF;
    prefacRLF(prefacRLF == Inf) = 0;
    prefacALF(prefacALF == Inf) = 0;

    % Morse potential calculations
    morseUU = CR * prefacRUU .* diffUU - CA * prefacAUU .* diffUU;
    morseUL = CR * prefacRUL .* diffUL - CA * prefacAUL .* diffUL;
    morseUF = CR * prefacRUF .* diffUF - CA * prefacAUF .* diffUF;
    morseLL = CR * prefacRLL .* diffLL - CA * prefacALL .* diffLL;
    morseFF = CR * prefacRFF .* diffFF - CA * prefacAFF .* diffFF;
    morseLF = CR * prefacRLF .* diffLF - CA * prefacALF .* diffLF;

    % Update for undecided (U)
    velocity = velocity + prefac3 * sum(morseUU, 2)' + prefac1 * sum(morseUL, 2)' + prefac2 * sum(morseUF, 2)' ...
              + dt * (alpha - beta * vel) .* velocity - dt * go_to_r_U / tau_r * (velocity - v_r) - dt * go_to_b_U / tau_b * (velocity - v_b);
    position = position + prefac3 * velocity;
    preference = preference - prefac3 * sum(interactionUU .* prefdiffUU, 2)' - prefac2 * sum(interactionUL .* prefdiffUL, 2)' ...
                 - prefac3 * sum(interactionUF .* prefdiffUF, 2)';

    % Update for leaders (L)
    velocity1 = velocity1 - prefac3 * sum(morseUL, 1)' + prefac2 * sum(morseLF, 2)' + prefac1 * sum(morseLL, 1)' ...
                + dt * (alpha - beta * vel1) .* velocity1 - dt * go_to_r_L / tau_r * (velocity1 - v_r) - dt * go_to_b_L / tau_b * (velocity1 - v_b);
    position1 = position1 + prefac1 * velocity1;
    preference1 = preference1 + prefac3 * sum(interactionUL .* prefdiffUL, 2)' - prefac1 * sum(interactionLL .* prefdiffLL, 2)' ...
                  - prefac2 * sum(interactionLF .* prefdiffLF, 2)' - muL * (preference1 - 1);

    % Update for followers (F)
    velocity2 = velocity2 - prefac3 * sum(morseUF, 1)' - prefac1 * sum(morseLF, 1)' + prefac2 * sum(morseFF, 1)' ...
                + dt * (alpha - beta * vel2) .* velocity2 - dt * go_to_r_F / tau_r * (velocity2 - v_r) - dt * go_to_b_F / tau_b * (velocity2 - v_b);
    position2 = position2 + prefac2 * velocity2;
    preference2 = preference2 + prefac1 * sum(interactionLF .* prefdiffLF, 2)' - prefac2 * sum(interactionFF .* prefdiffFF, 2)' ...
                  + prefac3 * sum(interactionUF .* prefdiffUF, 2)' - muF * (preference2 + 1);

    % Mean and deviation calculations
    mean_vel(i, :) = (sum(velocity, 2) + sum(velocity2, 2) + sum(velocity1, 2)) / N;
    dev_vel(i) = sum((velocity(1, :) - mean_vel(i, 1)).^2 + (velocity(2, :) - mean_vel(i, 2)).^2) + ...
                   sum((velocity1(1, :) - mean_vel(i, 1)).^2 + (velocity1(2, :) - mean_vel(i, 2)).^2) + ...
                   sum((velocity2(1, :) - mean_vel(i, 1)).^2 + (velocity2(2, :) - mean_vel(i, 2)).^2);
    mean_pref(i) = (sum(preference) + sum(preference1) + sum(preference2)) / N;
    dev_pref(i) = sum((preference - mean_pref(i)).^2) / N3 + sum((preference1 - mean_pref(i)).^2) / N1 + ...
                  sum((preference2 - mean_pref(i)).^2) / N2;

    % Save data every `save_every` iterations
    if mod(i, save_every) == 0
        z(count, 1:2, :) = position;
        z(count, 3:4, :) = velocity;
        z(count, 5, :) = preference;
        
        z1(count, 1:2, :) = position1;
        z1(count, 3:4, :) = velocity1;
        z1(count, 5, :) = preference1;
        
        z2(count, 1:2, :) = position2;
        z2(count, 3:4, :) = velocity2;
        z2(count, 5, :) = preference2;
        count = count + 1;
    end


end

% Define the number indices
num1 = 1;
num2 = 5;
num3 = 10;
num4 = 50;

% Define color normalization (similar to mpl.colors.Normalize in Python)
cmap = [-1, 1];

% Create a 2x2 subplot figure with a large size
figure('Position', [100, 100, 1200, 1200]);

% Subplot 1 (top-left)
subplot(2, 2, 1);
quiver(z(num1, 1, :), z(num1, 2, :), z(num1, 3, :), z(num1, 4, :), 0.5, 'Color', 'r');
hold on;
quiver(z1(num1, 1, :), z1(num1, 2, :), z1(num1, 3, :), z1(num1, 4, :), 0.5, 'Color', 'b');
quiver(z2(num1, 1, :), z2(num1, 2, :), z2(num1, 3, :), z2(num1, 4, :), 0.5, 'Color', 'g');
hold off;

% Subplot 2 (top-right)
subplot(2, 2, 2);
quiver(z(num2, 1, :), z(num2, 2, :), z(num2, 3, :), z(num2, 4, :), 0.5, 'Color', 'r');
hold on;
quiver(z1(num2, 1, :), z1(num2, 2, :), z1(num2, 3, :), z1(num2, 4, :), 0.5, 'Color', 'b');
quiver(z2(num2, 1, :), z2(num2, 2, :), z2(num2, 3, :), z2(num2, 4, :), 0.5, 'Color', 'g');
hold off;

% Subplot 3 (bottom-left)
subplot(2, 2, 3);
quiver(z(num3, 1, :), z(num3, 2, :), z(num3, 3, :), z(num3, 4, :), 0.5, 'Color', 'r');
hold on;
quiver(z1(num3, 1, :), z1(num3, 2, :), z1(num3, 3, :), z1(num3, 4, :), 0.5, 'Color', 'b');
quiver(z2(num3, 1, :), z2(num3, 2, :), z2(num3, 3, :), z2(num3, 4, :), 0.5, 'Color', 'g');
hold off;

% Subplot 4 (bottom-right)
subplot(2, 2, 4);
quiver(z(num4, 1, :), z(num4, 2, :), z(num4, 3, :), z(num4, 4, :), 0.5, 'Color', 'r');
hold on;
quiver(z1(num4, 1, :), z1(num4, 2, :), z1(num4, 3, :), z1(num4, 4, :), 0.5, 'Color', 'b');
quiver(z2(num4, 1, :), z2(num4, 2, :), z2(num4, 3, :), z2(num4, 4, :), 0.5, 'Color', 'g');
hold off;

% Save the figure as PNG file
saveas(gcf, 'fishy_second_attempt.png');

% Show the figure
shg;


end

function [diffxixj, distance] = euclidean_distance(y1, y2)
    % Calculates the Euclidean distance between two sets of points y1 and y2
    % Input:
    %   y1 - First array of points, size [N, D]
    %   y2 - Second array of points, size [M, D]
    % Output:
    %   diffxixj - Difference between each pair of points in y1 and y2, size [N, M, D]
    %   distance - Euclidean distance between each pair of points, size [N, M]
    
    % Calculate the differences between each point in y1 and each point in y2
    diffxixj = bsxfun(@minus, y1, permute(y2, [3, 2, 1]));
    
    % Calculate the Euclidean distance
    distance = sqrt(sum(diffxixj .^ 2, 2));
end

function opinion_dist = opinion_distance(w1, w2)
    % Calculates the opinion distance (element-wise difference) between w1 and w2
    % Input:
    %   w1 - First array, size [N, 1]
    %   w2 - Second array, size [M, 1]
    % Output:
    %   opinion_dist - Difference between each element in w1 and w2, size [N, M]
    
    opinion_dist = bsxfun(@minus, w1, permute(w2, [2, 1]));
end