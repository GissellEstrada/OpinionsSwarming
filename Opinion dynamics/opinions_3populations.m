clear all;
close all;

T = 8;              % final time
dt = 1.0e-1;          % timestep
M = floor(T/dt);      % number of time steps
L = 30;                % computational domain (opinion space)


Na = 50;              % individuals with preference for a (opinion is 1)
Nb = 100;              % individuals with preference for b (opinion is -1)
Nu = 50;               % individuals without any preference

a1 = 1; a2 = 10; a3 = 1; % strength of the interaction opinions
b1 = 0.1; b2 = 10; b3 = 0.5; % strength of the interaction opinions
u1 = 0.3; u2 = 0.3; u3 = 0.3; % strength of the interaction opinions

mu_a = 0;                   % strength of a towards opinion 1
mu_b = 0;                   % strength of b towards opinion -1


%% new parameters to test
sigma = 0.05;         % noise

%% store position, speed and gradient of 
xa = zeros(Na,M); 
xb = zeros(Nb,M); 
xu = zeros(Nu,M); 

%% initialise positions and speeds randomly
Icon = 1;            % starting point of trajectories from -1 to 1
xa(:,1) = sort(L*rand(1,Na));
xb(:,1) = sort(L*rand(1,Nb));
xu(:,1) = sort(L*rand(1,Nu));

% xa(:,1) = -Icon + (Icon+Icon)*rand(Na,1);
% xb(:,1) = -Icon + (Icon+Icon)*rand(Nb,1);
% xu(:,1) = -Icon + (Icon+Icon)*rand(Nu,1);


%% Evolution of opinions

%% time step

for k = 1:M-1
    Psi_aa = psi1(xa(:,k),Na);
    Psi_bb = psi1(xb(:,k),Nb);
    Psi_uu = psi1(xu(:,k),Nu);

    Psi_ab = psi2(xa(:,k),xb(:,k),Na,Nb);
    Psi_ba = psi2(xb(:,k),xa(:,k),Nb,Na);
    Psi_au = psi2(xa(:,k),xu(:,k),Na,Nu);
    Psi_ua = psi2(xu(:,k),xa(:,k),Nu,Na);
    Psi_bu = psi2(xb(:,k),xu(:,k),Nb,Nu);
    Psi_ub = psi2(xu(:,k),xb(:,k),Nu,Nb);
    
    xa(:,k+1) = xa(:,k) + a1*dt*Psi_aa(:)/Na + a2*dt*Psi_ab(:)/Nb + a3*dt*Psi_au(:)/Nu + mu_a*(xa(:,k)-ones(Na,1));% + sqrt(dt)*randn(N,1)*sigma; % 
    xb(:,k+1) = xb(:,k) + b1*dt*Psi_bb(:)/Nb + b2*dt*Psi_ba(:)/Na + b3*dt*Psi_bu(:)/Nu + mu_b*(xb(:,k)+zeros(Nb,1));
    xu(:,k+1) = xu(:,k) + u1*dt*Psi_uu(:)/Nu + u2*dt*Psi_ua(:)/Na + u3*dt*Psi_ub(:)/Nb ;

end

%% plot the results
figure
subplot(1,3,1)
plot(1:M,xa,'r')
subplot(1,3,2)
plot(1:M,xb,'b')
subplot(1,3,3)
plot(1:M,xu,'k')

%save ('Trajectories_repulsive.mat','x');

function Psi = psi1(X,N) % function to evaluate grad W (Xi-Xj)
Psi = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if i ~= j %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(X(i)-X(j))<1
                    
                    Psi(i) = Psi(i) +  (X(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end
                 %
         
            end
        end
    end    
end

function Psi = psi2(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
            if i ~= j %&& abs(X(i)-Y(j))>(1+1e-13)

                if abs(X(i)-Y(j))<1
                    
                    Psi(i) = Psi(i) +  (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end
         
            end
        end
    end    
end





