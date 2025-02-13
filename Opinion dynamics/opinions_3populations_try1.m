clear all;
close all;

T = 50;              % final time
dt = 1.0e-1;          % timestep
M = floor(T/dt);      % number of time steps
L = 10;                % computational domain (opinion space)


Na = 5;              % individuals with preference for a (opinion is 1), leaders
Nb = 200;              % individuals with preference for b (opinion is -1), followers
Nu = 100;               % individuals without any preference

a1 = 1; a2 = 1; a3 = 1; % strength of the interaction opinions
b1 = 1; b2 = 1; b3 = 1; % strength of the interaction opinions
u1 = 1; u2 = 1; u3 = 1; % strength of the interaction opinions

mu_a = 0.1;                   % strength of a towards opinion 1
mu_b = 0.05;                   % strength of b towards opinion -1


%% new parameters to test
sigma = 0.05;         % noise

%% store position, speed and gradient of 
xa = zeros(Na,M); 
xb = zeros(Nb,M); 
xu = zeros(Nu,M); 

%% initialise positions and speeds randomly

% xa(:,1) = sort(L*rand(1,Na));
% xb(:,1) = sort(L*rand(1,Nb));
% xu(:,1) = sort(L*rand(1,Nu));

xa(:,1) = -L + (L+L)*rand(Na,1);
xb(:,1) = -L + (L+L)*rand(Nb,1);
xu(:,1) = -L + (L+L)*rand(Nu,1);


%% Evolution of opinions

%% time step

for k = 1:M-1

    % Interactions between the same population result in opinion average
    Psi_aa = psi1(xa(:,k),Na);
    Psi_bb = psi1(xb(:,k),Nb);
    Psi_uu = psi1(xu(:,k),Nu);
    
    % Leaders don't change their opinion after interaction with followers
    Psi_ab = psi_lead(xa(:,k),xb(:,k),Na,Nb);

    % Neither followers nor leaders change the opinion after interaction
    % with the uninformed
    Psi_au = psi_lead(xa(:,k),xu(:,k),Na,Nu);

    % Followers might change their opinion after interaction with
    % uninformed
    Psi_bu = psi_folounin(xb(:,k),xu(:,k),Nb,Nu);

    % Follower-leaders interaction results in average of opinion for the
    % follower population
    Psi_ba = psi_leadfol(xb(:,k),xa(:,k),Nb,Na);

%     % The uninformed remain uninformed after interaction
%     Psi_ua = psi_lead(xu(:,k),xa(:,k),Nu,Na);
%     Psi_ub = psi_lead(xu(:,k),xb(:,k),Nu,Nb);

    % The uninformed might chnage their opinion
    Psi_ua = psi_unin(xu(:,k),xa(:,k),Nu,Na);
    Psi_ub = psi_unin(xu(:,k),xb(:,k),Nu,Nb);
    
    xa(:,k+1) = xa(:,k) + a1*dt*Psi_aa(:)/Na + a2*dt*Psi_ab(:)/Nb + a3*dt*Psi_au(:)/Nu - mu_a*(xa(:,k)-L*ones(Na,1));% + sqrt(dt)*randn(N,1)*sigma; % 
    xb(:,k+1) = xb(:,k) + b1*dt*Psi_bb(:)/Nb + b2*dt*Psi_ba(:)/Na + b3*dt*Psi_bu(:)/Nu - mu_b*(xb(:,k)+L*ones(Nb,1));
    xu(:,k+1) = xu(:,k) + u1*dt*Psi_uu(:)/Nu + u2*dt*Psi_ua(:)/Na + u3*dt*Psi_ub(:)/Nb ;

end

%% plot the results
figure
subplot(1,3,1)
plot(1:M,xa,'r'); hold on
subplot(1,3,2)
plot(1:M,xb,'b'); hold on
subplot(1,3,3)
plot(1:M,xu,'k')



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

function Psi = psi_lead(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
                if abs(Y(j)-X(i))<1
                    
                    Psi(i) = Psi(i) +  (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end
        end
    end    
end

function Psi = psi_leadfol(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
                if abs(Y(j)-X(i))<5
                    
                    Psi(i) = Psi(i) +  (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end   
         
         end
     end
end    

function Psi = psi_folounin(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
                if abs(Y(j)-X(i))<5
                    
                    Psi(i) = Psi(i) + (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end   
         
         end
     end
end

function Psi = psi_unin(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
                if abs(Y(j)-X(i))<0.5
                    
                    Psi(i) = Psi(i) + (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end   
         
        end
    end    
end



