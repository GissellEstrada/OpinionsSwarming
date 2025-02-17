clear all;
close all;

tf = 20;              % final time
dt = 1.0e-1;          % timestep
M = floor(tf/dt);     % number of time steps
L = 1;                % computational domain (opinion space)

Nl = 6;              % individuals with preference for a (opinion is 1), leaders
Nf = 6;              % individuals with preference for b (opinion is -1), followers
Nu = 10;             % individuals without any preference

% strength of the interaction opinions
p_ll = 1; 
p_lf = 1; p_lu = 0; 
p_fl = 1; p_ff = 1; p_fu = 0;
p_ul = 0; p_uf = 0; p_uu = 0;

tau_a = 0.1;                   % conviction of leaders towards opinion 1
tau_b = 0.1;                   % conviction of followers towards opinion -1


%% new parameters to test
sigma = 0;         % noise

%% store position, speed and gradient of opinions
wl = zeros(Nl,M); 
wf = zeros(Nf,M); 
wu = zeros(Nu,M); 

%% initialise positions and speeds randomly

rng(1234);

% xa(:,1) = sort(L*rand(1,Na));
% xb(:,1) = sort(L*rand(1,Nb));
% xu(:,1) = sort(L*rand(1,Nu));

wl(:,1) = -L + (L+L)*rand(Nl,1);
wf(:,1) = -L + (L+L)*rand(Nf,1);
wu(:,1) = -L + (L+L)*rand(Nu,1);

aver_l = sum(wl(:,1))/Nl;
aver_f = sum(wf(:,1))/Nf;
aver_u = sum(wu(:,1))/Nu;
total_aver_initial = (sum(wl(:,1))+sum(wf(:,1))+sum(wu(:,1)))/(Nl+Nf+Nu);

%% Evolution of opinions

%% time step

energ = zeros(1,M-1);

% let T0 = sum[psi(i,j)(wi-wj)]
% and T1 = sum (p_lm * T1 / Nm)
% then d(wi) = T1 + T2, see article for T2

% in the following loop we first

for k = 1:M-1

    % Interactions between the same population result in opinion average
    Psi_ll = psi1(wl(:,k),Nl);
    Psi_ff = psi1(wf(:,k),Nf);
    Psi_uu = psi1(wu(:,k),Nu);
    
    %% Leaders
    % Change opinion after interaction with followers
    Psi_lf = psi_leadfoll(wl(:,k),wf(:,k),Nl,Nf);

    % Don't change opinion after interaction with followers
    % Psi_lf = psi_lead(xa(:,k),xb(:,k),Na,Nb);

    % Change opinion after interaction with uninformed
    Psi_lu = psi_follunin(wl(:,k),wu(:,k),Nl,Nu);

    % Don't change opinion after interaction with uninformed
    % Psi_lu = psi_lead(xa(:,k),xu(:,k),Na,Nu);

    %% Followers
    % Change opinion after interaction with leader (always)
    Psi_fl = psi_leadfoll(wf(:,k),wl(:,k),Nf,Nl);

    % Change opinion after interaction with uninformed
    Psi_fu = psi_follunin(wf(:,k),wu(:,k),Nf,Nu);

    % Don't change opinion after interaction with uninformed
    % Psi_fu = psi_lead(xb(:,k),xu(:,k),Nb,Nu);    

    %% Uninformed
    % Always change opinion 
    Psi_ul = psi_unin(wu(:,k),wl(:,k),Nu,Nl);
    Psi_uf = psi_unin(wu(:,k),wf(:,k),Nu,Nf);

    % Remained uninformed 
    % Psi_ul = psi_lead(xu(:,k),xa(:,k),Nu,Na);
    % Psi_uf = psi_lead(xu(:,k),xb(:,k),Nu,Nb);

    %% Equation (no noise => sigma = 0)
    wl(:,k+1) = wl(:,k) + p_ll*dt*Psi_ll(:)/Nl + p_lf*dt*Psi_lf(:)/Nf + p_lu*dt*Psi_lu(:)/Nu - tau_a*(wl(:,k)-L*ones(Nl,1)) + sqrt(dt)*randn(Nl,1)*sigma; % 
    wf(:,k+1) = wf(:,k) + p_fl*dt*Psi_ff(:)/Nf + p_ff*dt*Psi_fl(:)/Nl + p_fu*dt*Psi_fu(:)/Nu - tau_b*(wf(:,k)+L*ones(Nf,1)) +  sqrt(dt)*randn(Nf,1)*sigma;
    wu(:,k+1) = wu(:,k) + p_ul*dt*Psi_uu(:)/Nu + p_uf*dt*Psi_ul(:)/Nl + p_uu*dt*Psi_uf(:)/Nf; %+  sqrt(dt)*randn(Nu,1)*sigma;

    mean_op = (sum(wl(:,k+1)) + sum(wf(:,k+1)) + sum(wu(:,k+1)))/(Nl+Nf+Nu);
    
    energ(k+1) = sum((wl(:,k+1)-mean_op).^2) + sum((wf(:,k+1)-mean_op).^2) + sum((wu(:,k+1)-mean_op).^2);
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

finalAv_a = sum(wl(:,end))/Nl;
finalAv_b = sum(wf(:,end))/Nf;
finalAv_c = sum(wf(:,end))/Nf;
totalAv_final = (sum(wl(:,end))+sum(wf(:,end))+sum(wu(:,end)))/(Nl+Nf+Nu);

%% plot the results
figure
grayColor = [.7 .7 .7];
subplot(1,3,1)
plot(1:M,wl,'Color',[0 0.4470 0.7410],'LineWidth',2); hold on 
plot(1:M,aver_l*ones(1,M),'--','Color',[0 0.4470 0.7410],'LineWidth',2); hold on
plot(1:M, totalAv_final*ones(1,M),'Color', grayColor,'LineWidth',2)
title('Leaders')
ylim([-1 1])
set(gca,'FontSize',15); hold on

subplot(1,3,2)
plot(1:M,wf,'Color',[0.8500 0.3250 0.0980],'LineWidth',2); hold on
plot(1:M,aver_f*ones(1,M),'--','Color',[0.8500 0.3250 0.0980],'LineWidth',2); hold on
plot(1:M, totalAv_final*ones(1,M),'Color', grayColor, 'LineWidth',2)
title('Followers')
ylim([-1 1])
set(gca,'FontSize',15); hold on
subplot(1,3,3)

plot(1:M,wu,'k','LineWidth',2); hold on
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

% function Psi = psi_lead(X,Y,Nx,Ny) % function to evaluate grad W (Xi-Xj)
% Psi = zeros(Nx,1);
%     for i = 1:Nx
%         for j = 1:Ny
%                 if abs(Y(j)-X(i))<0.1
                    
%                     Psi(i) = Psi(i) +  (Y(j)-X(i));

%                 else
%                     Psi(i) = Psi(i);
%                 end
%         end
%     end    
% end

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


