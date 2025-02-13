N = 100;                      % Choosing the number of agents N.
L = 50;                      % Choosing opinion space.
v = sort(L*rand(1,N));       % Generating x0, here uniformly at random.
M = v;                       % Opinions are stored in a matrix M.
t = 0;                       % t denotes the current time step.

count = 1;
while  1                     % The loop will run until terminated.

u = zeros(1,N);              % Resetting the vector that will become.

% the next v

 for i = 1:N
w = v;                            % A temporary vector w of potential neighbours.
w(abs(v-v(i))>(1+1e-13)) = [];    % Finding the neighbours of i by removing
                                  % opinions that differ by more than
                                  % some r>1, see discussion below.
u(1,i) = mean(w);                 % Agent i chooses the mean of neighbouring
                                  % opinions.
 end

v = u;                            % Updating the vector.
M(t+1,:) = u;                     % Updating M.
t = t+1;                          % Moving on to the next time step.
q = abs(diff(v));                 % These lines detects if the system is
q(q==0) = 2;                      % frosen and break the main loop if it is.

if min(q) > 1

    break
end

end
plot(M,'.-') 