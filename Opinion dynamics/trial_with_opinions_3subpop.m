clear all;
clc;
close all;
format long

%-------------------READ ME-----------------------%

% The "cube" x stores both spatial coordinates and velocities for each instant in time:
% x = [spatial coord. uninformed; spatial coord. leaders; spatial coord. followers;
%       velocity coord. uninformed; velocity coord. leaders; velocity
%       coord. followers];
% The indexing for each kind of velocity and spatial coord. can be found in
% the line where I plot the solution
% The vector w stores opinions from each subpopulation:
% w = [uninformed op; leader op.; follower op.];

% The only place where I multiply by the factor 1/N is in the sumphi term,
% which is only found in the equation regarding w

% All the parameters used in the model are stated and described right bellow

%----------------------------------------------%

% Problem data
NL = 50; NF = 50; NU = 50;          % number of individuals from each group
Ca = 1; la = 1; Cr = 0.6; lr = 0.5; % attraction-repulsion parameters
T = 60;                             % final time
dt = 1.0e-2;                        % timestep
M = floor(T/dt);                    % number of time steps
alpha = 1; beta = 0.5;              % friction parameters
N = NU + NL + NF;                   % total number of individuals
rx = 0;                             % spatial radius for alignment in velocity
rw = 0.1;                           % preference radius for alignment il velocity
tau1 = 0.5;                         % strenght of preference towards 1
taum1 = 0.5;                        % strenght of preference towards -1

% Initial conditions
x0L = -1 + 2*rand(NL,2); v0L = -1 + 2*rand(NL,2);
x0F = -1 + 2*rand(NF,2); v0F = -1 + 2*rand(NF,2);
x0U = -1 + 2*rand(NU,2); v0U = -1 + 2*rand(NU,2);
w0U = -1 + 2*rand(NU,1);
w0F = -1 + 2*rand(NF,1);
w0L = -1 + 2*rand(NL,1);

x0 = [x0U;x0L;x0F];
v0 = [v0U;v0L;v0F];
w0 = [w0U;w0L;w0F];

% Initial condition for Adams-Bashforth
x = zeros(2*N,2,M);
x(:,:,1) = [x0;v0]; % Initial state
w(:,1) = w0;
[F1x, F1w] = F(x(:,:,1), w(:,1), NL, NF, NU, alpha, beta, Cr, Ca, lr, la, rx, rw, tau1, taum1); % Initial F value

% Use Euler for the first step
x(:,:,2) = x(:,:,1) + dt * F1x; 
w(:,2) = w(:,1) + dt * F1w; 
% Compute F for the second step (needed for Adams-Bashforth)
[F2x, F2w] = F(x(:,:,2), w(:,2), NL, NF, NU, alpha, beta, Cr, Ca, lr, la, rx, rw, tau1, taum1);

% Adams-Bashforth 2-step method
for i = 3:M
    % Adams-Bashforth 2-step formula
    x(:,:,i) = x(:,:,i-1) + (dt / 2) * (3 * F2x - F1x);  
    w(:,i) = w(:,i-1) + (dt / 2) * (3 * F2w - F1w);
    % Update F values for the next step
    F1x = F2x;  % F from previous step
    F1w = F2w;
    [F2x, F2w] = F(x(:,:,i), w(:,i), NL, NF, NU, alpha, beta, Cr, Ca, lr, la, rx, rw, tau1, taum1);  % F for current step
    if any(isnan(x(:,:,i)), 'all') || any(isinf(x(:,:,i)), 'all')
        error('NaN or Inf encountered at time step %d', i);
    end
end

% Plot solution at last instant in time
figure();
hold on;
% plots uninformed (preference towards -1) in red and leaders and followers
% (preference towards 1) in blue
quiver(x(1:NU,1,end),x(1:NU,2,end),x(N+1:N+NU,1,end),x(N+1:N+NU,2,end),'r');
quiver(x(NU+1:N,1,end),x(NU+1:N,2,end),x(NU+N+1:end,1,end),x(N+NU+1:end,2,end),'b');
hold off;
figure();
plot(1:M,w);

% -------------------PROBLEM-RELATED FUNCTIONS------------------- %

function [Fx,Fw] = F(x, w, NL, NF, NU, alpha, beta, Cr, Ca, lr, la, rx, rw, tau1, taum1)
    N = NL + NF + NU;
    u = x(1:NU,:);
    l = x(NU+1:NU+NL,:);
    f = x((NU+NL)+1:N,:);
    v = x(N+1:(N+NU),:);
    p = x((N+NU)+1:(N+NU+NL),:);
    q = x((N+NU+NL)+1:end,:);
    wu = w(1:NU);
    wl = w(NU+1:NU+NL);
    wf = w((NU+NL)+1:end);
    Fx = [v; ...
          p; ...
          q; ...
          sumpsi(u, l, f, NU, NL, NF, Cr, Ca, lr, la) + (alpha - beta*sum(v.^2,2)).*v + ...
          sumh(u, l, f, wu, wl, wf, v, p, q, NU, NL, NF, rx, rw) + abs(wu).*(-[1 0] - v); ...
          sumpsi(l, u, f, NL, NU, NF, Cr, Ca, lr, la) + (alpha - beta*sum(p.^2,2)).*p + ...
          sumh(l, u, f, wl, wu, wf, p, v, q, NL, NU, NF, rx, rw) + wl.*([1 0] - p); ...
          sumpsi(f, u, l, NF, NU, NL, Cr, Ca, lr, la) + (alpha - beta*sum(q.^2,2)).*q + ...
          sumh(f, u, l, wf, wu, wl, q, v, p, NF, NU, NL, rx, rw) + wf.*([1 0] - q)];
    Fw = [sumphi(u, l, f, wu, wl, wf, NU, NL, NF, rx, rw) + taum1*(-1 - wu); ...
          sumphi(l, u, f, wl, wu, wf, NL, NU, NF, rx, rw) + tau1*(1 - wl); ...
          sumphi(f, u, l, wf, wu, wl, NF, NU, NL, rx, rw) + tau1*(1 - wf)];
         
end

function sumpsi = sumpsi(x, y, z, N1, N2, N3, Cr, Ca, lr, la)
    sumpsi = zeros(size(x, 1), size(x, 2)); 
    for i = 1:size(sumpsi, 1)
        for j = 1:N1
            if j ~= i
                sumpsi(i,:) = sumpsi(i,:) - morsepotential(sqrt(sum((x(i,:)-x(j,:)).^2,2)),Cr,Ca,lr,la) .* (x(i,:) - x(j,:)) ./ norm(x(i,:)-x(j,:),2);
            end
        end
    end
    
    for i = 1:size(sumpsi, 1)
        for j = 1:N2
            sumpsi(i,:) = sumpsi(i,:) - morsepotential(sqrt(sum((x(i,:)-y(j,:)).^2,2)),Cr,Ca,lr,la) .* (x(i,:) - y(j,:)) ./ norm(x(i,:)-y(j,:),2);
        end
    end
    
    for i = 1:size(sumpsi, 1)
        for j = 1:N3
            sumpsi(i,:) = sumpsi(i,:) - morsepotential(sqrt(sum((x(i,:)-z(j,:)).^2,2)),Cr,Ca,lr,la) .* (x(i,:) - z(j,:)) ./ norm(x(i,:)-z(j,:),2);
        end
    end
end

function sumh = sumh(x, y, z, w1, w2, w3, v1, v2, v3, N1, N2, N3, rx, rw)
    sumh = zeros(size(x, 1), 2); 
    for i = 1:size(sumh, 1)
        for j = 1:N1
            if abs(w1(i) - w1(j)) < rw && norm(x(i,:)-x(j,:)) < rx 
            sumh(i,:) = sumh(i,:) +  (v1(j,:) - v1(i,:));
            end
        end
    end

    for i = 1:size(sumh, 1)
        for j = 1:N2
            if abs(w1(i) - w2(j)) < rw && norm(x(i,:)-y(j,:)) < rx 
            sumh(i,:) = sumh(i,:) + (v2(j,:) - v1(i,:));
            end
        end
    end

    for i = 1:size(sumh, 1)
        for j = 1:N3
            if abs(w1(i) - w3(j)) < rw && norm(x(i,:)-z(j,:)) < rx 
            sumh(i,:) = sumh(i,:) + (v3(j,:) - v1(i,:));
            end
        end
    end
end

function morsepotential = morsepotential(r, Cr, Ca, lr, la)
    morsepotential = -(Cr/lr)*exp(-r./lr) + (Ca/la)*exp(-r./la);
end

function sumphi = sumphi(x,y,z,w1,w2,w3,N1,N2,N3,rx,rw)
    sumphi = zeros(size(x, 1), 1); 
    for i = 1:size(sumphi, 1)
        for j = 1:N1
            if abs(w1(i) - w1(j)) < rw && norm(x(i,:)-x(j,:)) < rx 
            sumphi(i) = sumphi(i) +  (1/N1)*(w1(j) - w1(i));
            end
        end
    end

    for i = 1:size(sumphi, 1)
        for j = 1:N2
            if abs(w1(i) - w2(j)) < rw && norm(x(i,:)-y(j,:)) < rx 
            sumphi(i) = sumphi(i) + (1/N2)*(w2(j) - w1(i));
            end
        end
    end

    for i = 1:size(sumphi, 1)
        for j = 1:N3
            if abs(w1(i) - w3(j)) < rw && norm(x(i,:)-z(j,:)) < rx 
            sumphi(i) = sumphi(i) + (1/N3)*(w3(j) - w1(i));
            end
        end
    end
end