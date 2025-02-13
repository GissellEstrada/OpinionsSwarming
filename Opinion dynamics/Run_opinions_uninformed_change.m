clear all;
close all;

T = 20;              % final time
dt = 1.0e-1;          % timestep
M = floor(T/dt);      % number of time steps
L = 1;                % computational domain (opinion space)


Na = 6;              % individuals with preference for a (opinion is 1), leaders
Nb = 6;              % individuals with preference for b (opinion is -1), followers
Nu = 10;               % individuals without any preference

a1 = 1; a2 = 1; a3 = 0; % strength of the interaction opinions
b1 = 1; b2 = 1; b3 = 0; % strength of the interaction opinions
u1 = 0; u2 = 0; u3 = 0; % strength of the interaction opinions

mu_a = 0.1;                   % strength of a towards opinion 1
mu_b = 0.1;                   % strength of b towards opinion -1


%% new parameters to test
sigma = 0;         % noise

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

aver_a = sum(xa(:,1))/Na;
aver_b = sum(xb(:,1))/Nb;
aver_u = sum(xu(:,1))/Nu;
total_aver_initial = (sum(xa(:,1))+sum(xb(:,1))+sum(xu(:,1)))/(Na+Nb+Nu);

%% Evolution of opinions

%% time step

energ = zeros(1,M-1);

for k = 1:M-1

    % Interactions between the same population result in opinion average
    Psi_aa = psi1(xa(:,k),Na);
    Psi_bb = psi1(xb(:,k),Nb);
    Psi_uu = psi1(xu(:,k),Nu);
    
%% Leaders
    % Change opinion after interaction with followers
    Psi_ab = psi_leadfoll(xa(:,k),xb(:,k),Na,Nb);

    % Don't change opinion after interaction with followers
    % Psi_ab = psi_lead(xa(:,k),xb(:,k),Na,Nb);

    % Change opinion after interaction with uninformed
    Psi_au = psi_follunin(xa(:,k),xu(:,k),Na,Nu);

    % Don't change opinion after interaction with uninformed
    % Psi_au = psi_lead(xa(:,k),xu(:,k),Na,Nu);

%% Followers
    % Change opinion after interaction with leader (always)
    Psi_ba = psi_leadfoll(xb(:,k),xa(:,k),Nb,Na);

    % Change opinion after interaction with uninformed
    Psi_bu = psi_follunin(xb(:,k),xu(:,k),Nb,Nu);

    % Don't change opinion after interaction with uninformed
%     Psi_bu = psi_lead(xb(:,k),xu(:,k),Nb,Nu);    

%% Uninformed
    % Always change opinion 
    Psi_ua = psi_unin(xu(:,k),xa(:,k),Nu,Na);
    Psi_ub = psi_unin(xu(:,k),xb(:,k),Nu,Nb);

   % Remained uninformed 
%    Psi_ua = psi_lead(xu(:,k),xa(:,k),Nu,Na);
%    Psi_ub = psi_lead(xu(:,k),xb(:,k),Nu,Nb);

 %%
    xa(:,k+1) = xa(:,k) + a1*dt*Psi_aa(:)/Na + a2*dt*Psi_ab(:)/Nb + a3*dt*Psi_au(:)/Nu - mu_a*(xa(:,k)-L*ones(Na,1)) + sqrt(dt)*randn(Na,1)*sigma; % 
    xb(:,k+1) = xb(:,k) + b1*dt*Psi_bb(:)/Nb + b2*dt*Psi_ba(:)/Na + b3*dt*Psi_bu(:)/Nu - mu_b*(xb(:,k)+L*ones(Nb,1)) +  sqrt(dt)*randn(Nb,1)*sigma;
    xu(:,k+1) = xu(:,k) + u1*dt*Psi_uu(:)/Nu + u2*dt*Psi_ua(:)/Na + u3*dt*Psi_ub(:)/Nb; %+  sqrt(dt)*randn(Nu,1)*sigma;

    mean_op = (sum(xa(:,k+1)) + sum(xb(:,k+1)) + sum(xu(:,k+1)))/(Na+Nb+Nu);
    
    energ(k+1) = sum((xa(:,k+1)-mean_op).^2) + sum((xb(:,k+1)-mean_op).^2) + sum((xu(:,k+1)-mean_op).^2);
    var_energ(k) = (energ(k+1)-energ(k))/dt;
end
figure
subplot(1,2,1)
plot(1:M,energ,'LineWidth',2)
title('Energy')
set(gca,'FontSize',15); hold on
subplot(1,2,2)
plot(1:M-1,var_energ,'LineWidth',2)
title('Energy variation')
set(gca,'FontSize',15);

finalAv_a = sum(xa(:,end))/Na;
finalAv_b = sum(xb(:,end))/Nb;
finalAv_c = sum(xb(:,end))/Nb;
totalAv_final = (sum(xa(:,end))+sum(xb(:,end))+sum(xu(:,end)))/(Na+Nb+Nu);

%% plot the results
figure
grayColor = [.7 .7 .7];
subplot(1,3,1)
plot(1:M,xa,'Color',[0 0.4470 0.7410],'LineWidth',2); hold on 
plot(1:M,aver_a*ones(1,M),'--','Color',[0 0.4470 0.7410],'LineWidth',2); hold on
plot(1:M, totalAv_final*ones(1,M),'Color', grayColor,'LineWidth',2)
title('Leaders')
ylim([-1 1])
set(gca,'FontSize',15); hold on

subplot(1,3,2)
plot(1:M,xb,'Color',[0.8500 0.3250 0.0980],'LineWidth',2); hold on
plot(1:M,aver_b*ones(1,M),'--','Color',[0.8500 0.3250 0.0980],'LineWidth',2); hold on
plot(1:M, totalAv_final*ones(1,M),'Color', grayColor, 'LineWidth',2)
title('Followers')
ylim([-1 1])
set(gca,'FontSize',15); hold on
subplot(1,3,3)

plot(1:M,xu,'k','LineWidth',2); hold on
plot(1:M,aver_u*ones(1,M),'--k','LineWidth',2); hold on
plot(1:M, totalAv_final*ones(1,M),'Color', grayColor,'LineWidth',2)
title('Uninformed')
ylim([-1 1])
set(gca,'FontSize',15)



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
                if abs(Y(j)-X(i))<0.1
                    
                    Psi(i) = Psi(i) +  (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end
        end
    end    
end

function Psi = psi_leadfoll(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
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

function Psi = psi_follunin(X,Y,Nx,Ny) % only change opinion when very close
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
                if abs(Y(j)-X(i))<1
                    
                    Psi(i) = Psi(i) + (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end   
         
         end
     end
end

function Psi = psi_unin(X,Y,Nx,Ny) % can be persuaded to change opinion a lot
Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
                if abs(Y(j)-X(i))<1
                    
                    Psi(i) = Psi(i) + (Y(j)-X(i));

                else
                    Psi(i) = Psi(i);
                end   
         
        end
    end    
end

% function Psi = psi_leadfoll(X,Y,Nx,Ny) % can be persuaded to change opinion a lot
% Psi = zeros(Nx,1);
%     for i = 1:Nx
%         for j = 1:Ny
%                 phi = exp(-2*abs(Y(j)-X(i)));
%                     Psi(i) = Psi(i) + phi*(Y(j)-X(i));  
%          
%         end
%     end    
% end


