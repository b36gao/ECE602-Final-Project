%% simulate_SINDy_CSTR.m
% Simulation of a nonlinear dynamic system
% dX1dt = -X1 + X1X2 + X1^2 
% dX2dt = -2X2 + X1X2 
%
%generates a dataset containing dXdt, X, theta, eta
%where the system is represented as dXdt = Theta(X)*eta
%Theta is a matrix of monomials: [1, X1, X2, X1^2, X2^2, X1X2]
% Author: [Your Name]
% Date: [Today's Date]

clear; clc; close all;

time = 0:0.01:4;
X0 = [1;-1];
eta_true = [0, -1, 0, 1, 0, 1; 
    0, 0, -2, 0, 0, 1].';

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);
[t_sol, X_sol] = ode15s(@(t,X) dynsys_ode(t,X), time, X0, opts);

%reconstruct dXdt:
dXdt = zeros(size(X_sol));
Theta = zeros(size(X_sol, 1), 6);
for i = 1:size(X_sol, 1)
    X = X_sol(i,:).';
    Theta_i = [1, X(1), X(2), X(1)^2, X(2)^2, X(1)*X(2)];
    Theta(i,:) = Theta_i;
    dXdt(i,:) = Theta_i*eta_true;
end

noise = 0.01*[0.1, 0.05].* randn(size(dXdt));
dXdt = dXdt + noise;

%% Package the simulated dataset
dataset.t = t_sol;
dataset.X = X_sol;
dataset.dXdt = dXdt;
dataset.eta = eta_true;
dataset.Theta = Theta;
save('simulated_SINDy_dataset.mat', 'dataset');

%% Plot the results
figure;
subplot(2,1,1);
plot(time, X_sol.', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('X');
title('System component over time');

%% Plot the results
figure;
subplot(2,1,1);
plot(time, dXdt.', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('dX/dt');
title('System component rate of change over time');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dXdt = dynsys_ode(~, X)
    % ODE for the dynamic system:
    % X = [X1; X2]
eta = [0, -1, 0, -1, 0, 0; 
    0, 0, -2, 0, 0, 1].';
    Theta = [1, X(1), X(2), X(1)^2, X(2)^2, X(1)*X(2)];
    dXdt = (Theta*eta).';
end
