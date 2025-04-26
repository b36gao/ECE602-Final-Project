%% simulate_acetoacetylation_CSTR.m
% Simulation of a CSTR reactor for the acetoacetylation of pyrrole with diketene.
%
% Reaction mechanism:
%   Reaction 1: P + D -> PAA,    r1 = k_a * [P] * [D] * [K]
%   Reaction 2: D + D -> DHA,    r2 = k_b * [D]^2 * [K]
%   Reaction 3: D -> OL,         r3 = k_c * [D]
%   Reaction 4: PAA + D -> G,    r4 = k_d * [PAA] * [D] * [K]
%                (k_d is set to zero for Scenario I)
%
% Species order:
%   1: Diketene (D)
%   2: Pyrrole (P)
%   3: 2-Acetoacetyl pyrrole (PAA)
%   4: Dehydroacetic acid (DHA)
%   5: Oligomers (OL)
%   6: Byproduct (G)
%   7: Catalyst (K)
%
% Stoichiometric matrix N (4 reactions x 7 species):
%   R1: [-1, -1, +1,  0,  0,  0,  0]
%   R2: [ 0, -2,  0, +1,  0,  0,  0]
%   R3: [ 0, -1,  0,  0, +1,  0,  0]
%   R4: [ 0, -1, -1,  0,  0, +1,  0]
%
% The CSTR model (assuming constant volume) is:
%   dC/dt = (Q/V) * (C_in - C) + N' * r(C)
%
% The simulated dataset will contain:
%   - A matrix R (4 x m) with each column giving [r1; r2; r3; r4] at a sampling time.
%   - A vector alpha containing the true rate constants: [k_a; k_b; k_c; k_d].
%
% Author: [Your Name]
% Date: [Today's Date]

clear; clc; close all;

%% Reactor and simulation parameters
V = 0.5;                 % Reactor volume [L]
Q = 0.05/60;             % Flow rate [L/s] (5 mL/min converted to L/s)
Ts = 10;                 % Sampling time [s]
t_final = 3600;          % Final simulation time [s] (60 minutes)
time = 0:Ts:t_final;

%% Inlet conditions and initial conditions
% Species order: [D; P; PAA; DHA; OL; G; K]
C_in = [6; 0; 0; 0; 0; 0; 0];         % Inlet concentrations [mol/L]
C0   = [0.14; 0.3; 0.08; 0.01; 0.01; 0.01; 2];  % Initial reactor concentrations

%% True rate constants (alpha)
% Kinetic parameters for the four reactions:
% k_a, k_b, k_c, k_d, where for Scenario I we set k_d = 0.
ka = 0.0523;
kb = 0.1279;
kc = 0.0281;
kd = 0;    % Reaction 4 is inactive in Scenario I
alpha_true = [ka; kb; kc; kd];

%% Stoichiometric matrix N (4 reactions x 7 species)
    N = [ -1, -1, +1,  0,  0,  0, 0;
       -2, 0,  0, +1,  0,  0, 0;
       -1, 0,  0,  0, +1,  0, 0;
       -1, 0, -1,  0,  0, +1, 0];

%% Solve the CSTR ODEs
opts = odeset('RelTol',1e-6,'AbsTol',1e-9,'NonNegative',1:7);
[t_sol, C_sol] = ode45(@(t,C) cstr_ode(t, C, Q, V, C_in, alpha_true), time, C0, opts);

%% Compute the reaction rate matrix R at sampling times
% R is a 4 x m matrix. Each column gives the vector [r1; r2; r3; r4] computed at that time.
numSamples = length(t_sol);
R = zeros(4, numSamples);
for i = 1:numSamples
    C = C_sol(i,:).';
    R(:,i) = reaction_rates(C, alpha_true);
end

%% Package the simulated dataset
dataset.t = t_sol;
dataset.C = C_sol;
dataset.R = R;
dataset.alpha = alpha_true;
save('simulated_acetoacetylation_dataset.mat', 'dataset');

%% Plot the results
figure;
subplot(2,1,1);
plot(t_sol, C_sol, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Concentration (mol/L)');
legend({'Diketene','Pyrrole','PAA','DHA','Oligomers','Byproduct','Catalyst'}, 'Location', 'Best');
title('Species Concentrations in the CSTR');

subplot(2,1,2);
plot(t_sol, R', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Reaction Rate');
legend({'r_1','r_2','r_3','r_4'}, 'Location', 'Best');
title('Reaction Rates in the CSTR');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dCdt = cstr_ode(~, C, Q, V, C_in, alpha)
    % ODE for the CSTR reactor:
    % C = [D; P; PAA; DHA; OL; G; K]
    r = reaction_rates(C, alpha);    % Compute the 4 reaction rates (4x1 vector)
    r = max(r, 0);
    % Define stoichiometric matrix (same as above)
    N = [ -1, -1, +1,  0,  0,  0, 0;
       -2, 0,  0, +1,  0,  0, 0;
       -1, 0,  0,  0, +1,  0, 0;
       -1, 0, -1,  0,  0, +1, 0];
    reaction_term = N.' * r;  % N' is 6x4, r is 4x1, so reaction_term is 6x1.
    dCdt = (Q/V) * (C_in - C) + reaction_term;
end

function r = reaction_rates(C, alpha)
    % Compute the reaction rates based on current concentrations C and rate constants alpha.
    % C = [D; P; PAA; DHA; OL; G; K]
    % alpha = [k_a; k_b; k_c; k_d]
    k_a = alpha(1); k_b = alpha(2); k_c = alpha(3); k_d = alpha(4);
    % Reaction 1: r1 = k_a * [P] * [D] * [K]
    r1 = k_a * C(2) * C(1) * C(7);
    % Reaction 2: r2 = k_b * [D]^2 * [K]
    r2 = k_b * C(1)^2 * C(7);
    % Reaction 3: r3 = k_c * [D]
    r3 = k_c * C(1);
    % Reaction 4: r4 = k_d * [PAA] * [D] * [K]
    r4 = k_d * C(3) * C(1) * C(7);
    r = [r1; r2; r3; r4];
end