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
NL = 60;                             % number of individuals from each group (change the number N to get different dynamics)
%Ca = 1; la = 1; Cr = 0.5; lr = 0.5; % attraction-repulsion parameters
Ca = 1; la = 1; Cr = 0.6; lr = 0.5;
T = 30;                             % final time
dt = 1.0e-2;                        % timestep
M = floor(T/dt);                    % number of time steps
alpha = 1; beta = 0.5;              % friction parameters
N = NL;                   % total number of individuals
rx = 0.05;                             % spatial radius for alignment in velocity
rw = 0.05;                           % preference radius for alignment il velocity
taur = 0.05;                         % strenght of preference towards 1
taub = 0;                        % strenght of preference towards -1

%% Changing taur and taub gives very rich dynamics

% Initial conditions
x0L = -1 + 2*rand(NL,2); v0L = -1 + 2*rand(NL,2);
w0L = -1 + 2*rand(NL,1);

x0 = x0L;
v0 = v0L;
w0 = w0L;

% Initial condition for Adams-Bashforth
x = zeros(2*N,2,M);
x(:,:,1) = [x0;v0]; % Initial state
w(:,1) = w0;
[F1x, F1w] = F(x(:,:,1), w(:,1), NL, alpha, beta, Cr, Ca, lr, la, rx, rw, taur, taub); % Initial F value

% Use Euler for the first step
x(:,:,2) = x(:,:,1) + dt * F1x; 
w(:,2) = w(:,1) + dt * F1w; 
% Compute F for the second step (needed for Adams-Bashforth)
[F2x, F2w] = F(x(:,:,2), w(:,2), NL, alpha, beta, Cr, Ca, lr, la, rx, rw, taur, taub);

% Adams-Bashforth 2-step method
for i = 3:M
    % Adams-Bashforth 2-step formula
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * F2x - F1x);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * F2w - F1w);
    % Update F values for the next step
    F1x = F2x;  % F from previous step
    F1w = F2w;
    [F2x, F2w] = F(x(:,:,i), w(:,i), NL, alpha, beta, Cr, Ca, lr, la, rx, rw, taur, taub);  % F for current step
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')
        error('NaN or Inf encountered at time step %d', i);
    end
end

% figure;
% hold on;
% for i = 1:N
%    plot(squeeze(x(i, 1, 5996:5999)),  squeeze(x(i, 2, 5996:5999)), 'r'); 
% end
% hold on
% 
% for i = 1:N
%    plot(squeeze(x(i, 1, 5999:end)),  squeeze(x(i, 2, 5999:end)), 'b');
% end

% Plot solution at last instant in time
%  figure();
% 
% % % plots uninformed (preference towards -1) in red and leaders and followers
% % % (preference towards 1) in blue
quiver(x(1:N,1,end),x(1:N,2,end),x(N+1:N+N,1,end),x(N+1:N+N,2,end),'r');

figure()
quiver(x(1:N,1,:),x(1:N,2,:),x(N+1:N+N,1,:),x(N+1:N+N,2,:),'r');
% quiver(x(N+1:N,1,end),x(N+1:N,2,end),x(N+N+1:end,1,end),x(N+N+1:end,2,end),'b');

figure ()
quiver(x(1:N,1,end),x(1:N,2,end),x(1:N,1,end),x(1:N,2,end));
xlabel('x');
ylabel('y');
title('Trajectories of Agents in Cucker-Smale System with Morse Potential');
grid on;
hold off;

figure();
plot(1:M,w);

% -------------------PROBLEM-RELATED FUNCTIONS------------------- %

function [Fx,Fw] = F(x, w, NL, alpha, beta, Cr, Ca, lr, la, rx, rw, taur, taub)
    N = NL;
    l = x(1:NL,:);
    v = x(N+1:(N+NL),:);
    wu = w;
    Fx = [v; ...
          1/N*sumU(l, NL, Cr, Ca, lr, la) + (alpha - beta*sum(v.^2,2)).*v ...
          + taur*Op_alignLeft(l,v,wu,NL,rw) + taub*Op_alignRight(l,v,wu,NL,rw)];
    Fw = sumOP(l, wu, NL, rx, rw) + taur*(-1 - wu) + taub*(1 - wu);
       
end

function sumOP = sumOP(x,w1,N1,rx,rw)
    sumOP = zeros(size(x, 1), 1); 
    for i = 1:size(sumOP, 1)
        for j = 1:N1
            if abs(w1(i) - w1(j)) < rw && norm(x(i,:)-x(j,:)) < rx 
            sumOP(i) = sumOP(i) +  (w1(j) - w1(i));
            end
        end
    end
end

function sumU = sumU(x, N1, Cr, Ca, lr, la)
    sumU = zeros(size(x, 1), size(x, 2)); 
    for i = 1:size(sumU, 1)
        for j = 1:N1
            if j ~= i
                sumU(i,:) = sumU(i,:) - morsepotential(sqrt(sum((x(i,:)-x(j,:)).^2,2)),Cr,Ca,lr,la) .* (x(i,:) - x(j,:)) ./ norm(x(i,:)-x(j,:),2);
            end
        end
    end
end

function morsepotential = morsepotential(r, Cr, Ca, lr, la)
    morsepotential = -(Cr/lr)*exp(-r./lr) + (Ca/la)*exp(-r./la);
end


function Op_alignLeft = Op_alignLeft(x,v,w1,N1,rw)
    Op_alignLeft = zeros(size(x, 1), 1); 
    for i = 1:size(Op_alignLeft, 1)
        for j = 1:N1
            if abs(-1 - w1(j)) < rw 
            Op_alignLeft(i) = Op_alignLeft(i) + (-1 - v(i));
            end
        end
    end
end

function Op_alignRight = Op_alignRight(x,v,w1,N1,rw)
    Op_alignRight = zeros(size(x, 1), 1); 
    for i = 1:size(Op_alignRight, 1)
        for j = 1:N1
            if abs(1 - w1(j)) < rw 
            Op_alignRight(i) = Op_alignRight(i) +  (1 - v(i));
            end
        end
    end
end





