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
p_ff = 1;
p_uu = 0;
p_lf = 1; p_fl = p_lf;
p_lu = 0; p_ul = p_lu;
p_fu = 0; p_uf = p_fu; 

tau_B = 0.1;                   % conviction of leaders towards opinion 1
tau_R = 0.1;                   % conviction of followers towards opinion -1

w_B = 1
w_R = -1


%% new parameters to test
sigma = 0;         % noise


%% store position, speed and gradient of opinions
wl = zeros(Nl,M); 
wf = zeros(Nf,M); 
wu = zeros(Nu,M); 


%% initialise positions and speeds randomly

rng(1234);

% wa(:,1) = sort(L*rand(1,Nl));
% wb(:,1) = sort(L*rand(1,Nf));
% wu(:,1) = sort(L*rand(1,Nu));

wl(:,1) = -L + (L+L)*rand(Nl,1);
wf(:,1) = -L + (L+L)*rand(Nf,1);
wu(:,1) = -L + (L+L)*rand(Nu,1);

initial_avg_l = sum(wl(:,1))/Nl;
initial_avg_f = sum(wf(:,1))/Nf;
initial_avg_u = sum(wu(:,1))/Nu;
initial_avg = (sum(wl(:,1))+sum(wf(:,1))+sum(wu(:,1)))/(Nl+Nf+Nu);


%% Evolution of opinions: step

% energ = zeros(1,M-1);
r_w = 1;

% let T0 = sum[psi(i,j)(wi-wj)]
% and T1 = sum (p_lm * T1 / Nm)
% then d(wi) = T1 + T2, see article for T2
% in the following loop we first compute T1 and then solve the equation

for k = 1:M-1

    % Interactions between the same population result in opinion avgage
    Psi_ll = psi(wl(:,k),wl(:,k),Nl,Nl,r_w);
    Psi_ff = psi(wf(:,k),wf(:,k),Nf,Nf,r_w);
    Psi_uu = psi(wu(:,k),wu(:,k),Nu,Nu,r_w);
    
    %% Leaders
    % Change opinion after interaction with followers
    Psi_lf = psi(wl(:,k),wf(:,k),Nl,Nf,r_w);
    Psi_lu = psi(wl(:,k),wu(:,k),Nl,Nu,r_w);    % also with uninformed
    % Psi_lf =      % not with followers
    % Psi_lu =      % not with uninformed

    %% Followers
    % Change opinion after interaction with leader (always)
    Psi_fl = psi(wf(:,k),wl(:,k),Nf,Nl,r_w);
    Psi_fu = psi(wf(:,k),wu(:,k),Nf,Nu,r_w);    % also with uninformed
    % Psi_fu =      % not with uninformed

    %% Uninformed
    % Always change opinion 
    Psi_ul = psi(wu(:,k),wl(:,k),Nu,Nl,r_w);
    Psi_uf = psi(wu(:,k),wf(:,k),Nu,Nf,r_w);
    % Remained uninformed 
    % Psi_ul = 
    % Psi_uf = 

    %% Equation (no noise => sigma = 0)
    wl(:,k+1) = wl(:,k) + ...
        dt*p_ll*Psi_ll(:)/Nl + ...
        dt*p_lf*Psi_lf(:)/Nf + ...
        dt*p_lu*Psi_lu(:)/Nu + ...
        tau_B*(L*w_B*ones(Nl,1)-wl(:,k));
        % + sqrt(dt)*randn(Nl,1)*sigma;
    wf(:,k+1) = wf(:,k) + ...
        dt*p_fl*Psi_ff(:)/Nf + ...
        dt*p_ff*Psi_fl(:)/Nl + ...
        dt*p_fu*Psi_fu(:)/Nu + ...
        tau_R*(L*w_R*ones(Nf,1)-wf(:,k));
        % + sqrt(dt)*randn(Nf,1)*sigma;
    wu(:,k+1) = wu(:,k) + ...
        dt*p_ul*Psi_uu(:)/Nu + ...
        dt*p_uf*Psi_ul(:)/Nl + ...
        dt*p_uu*Psi_uf(:)/Nf; 
        %+ sqrt(dt)*randn(Nu,1)*sigma;

    % mean_op = (sum(wl(:,k+1)) + sum(wf(:,k+1)) + sum(wu(:,k+1)))/(Nl+Nf+Nu);
    % energ(k+1) = sum((wl(:,k+1)-mean_op).^2) + sum((wf(:,k+1)-mean_op).^2) + sum((wu(:,k+1)-mean_op).^2);
    % var_energ(k) = (energ(k+1)-energ(k))/dt;

end


%% plot the results

% final_avg_l = sum(wl(:,end))/Nl;
% final_avg_f = sum(wf(:,end))/Nf;
% final_avg_u = sum(wu(:,end))/Nu;
final_avg = (sum(wl(:,end))+sum(wf(:,end))+sum(wu(:,end)))/(Nl+Nf+Nu);

col_gray = [.7 .7 .7];
col_red = [0 0.4470 0.7410];
col_blue = [0.8500 0.3250 0.0980]

figure
subplot(1,3,1)
    plot(1:M,wl,'Color',col_red,'LineWidth',2); hold on 
    plot(1:M,initial_avg_l*ones(1,M),'--','Color',col_red,'LineWidth',2); hold on
    plot(1:M, final_avg*ones(1,M),'Color', col_gray,'LineWidth',2)
    ylim([-1 1])
    set(gca,'FontSize',15)
    title('Leaders'); hold on
subplot(1,3,2)
    plot(1:M,wf,'Color',col_blue,'LineWidth',2); hold on
    plot(1:M,initial_avg_f*ones(1,M),'--','Color',col_blue,'LineWidth',2); hold on
    plot(1:M, final_avg*ones(1,M),'Color', col_gray, 'LineWidth',2)
    ylim([-1 1])
    set(gca,'FontSize',15)
    title('Followers'); hold on
subplot(1,3,3)
    plot(1:M,wu,'k','LineWidth',2); hold on
    plot(1:M,initial_avg_u*ones(1,M),'--k','LineWidth',2); hold on
    plot(1:M, final_avg*ones(1,M),'Color', col_gray,'LineWidth',2)
    ylim([-1 1])
    set(gca,'FontSize',15)
    title('Uninformed')

% figure
% subplot(1,2,1)
%     plot(1:M,energ,'LineWidth',2)
%     title('Energy')
%     set(gca,'FontSize',15); hold on
% subplot(1,2,2)
%     plot(1:M-1,var_energ,'LineWidth',2)
%     title('Energy variation')
%     set(gca,'FontSize',15);



function Psi = psi(X,Y,Nx,Ny, r_w) % function to compute 1
    Psi = zeros(Nx,1);
    for i = 1:Nx
        for j = 1:Ny
            % if i ~= j                   %&& abs(X(i)-X(j))>(1+1e-13)
                if abs(Y(j)-X(i))<r_w
                    Psi(i) = Psi(i) +  (Y(j)-X(i));     % consultar gissell: j-i?
                end
            % end
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


