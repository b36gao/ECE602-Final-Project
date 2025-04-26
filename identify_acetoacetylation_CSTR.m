%% identify_acetoacetylation_model_loaded_with_discretization.m
% This script implements the identification algorithm for Section 5.1: simulation study 
% by loading the simulated dataset (from simulate_acetoacetylation_CSTR.m)
% and then (i) computing the discretized state-space matrices as in Eq. (12) (with T_s=1),
% and (ii) building the regression (data–equation) matrices as in Eq. (13).
%

clear; clc; close all;

%% PART 1: Load the simulation dataset
load('simulated_acetoacetylation_dataset.mat', 'dataset');
t_sol = dataset.t;      % time vector
C_sol = dataset.C(2:361, :);      % species concentrations;
C0_true = dataset.C(1, :);
alpha_true = dataset.alpha;  % true parameters


%% Reactor and simulation parameters
V = 0.5;                 % Reactor volume [L]
Q = 0.05/60;             % Flow rate [L/s] (5 mL/min converted to L/s)
T_s = 10;                 % Sampling time [s]
C_in = [6; 0; 0; 0; 0; 0; 0];   % Inlet concentrations [mol/L]

%% PART 2: Define stoichiometric matrices
%% Stoichiometric matrix N (4 reactions x 7 species)
    N = [ -1, -1, +1,  0,  0,  0, 0;
       -2, 0,  0, +1,  0,  0, 0;
       -1, 0,  0,  0, +1,  0, 0;
       -1, 0, -1,  0,  0, +1, 0];
% Augmented stoichiometric matrix for 8 candidate reactions 
N_aug = N;

%% PART 3: Compute candidate reaction rates at each sample time
numSamples   = length(t_sol)-1;
numReactions = 4;
s_tot        = 32;  % total candidate rate laws per reaction from Table 1
R_candidates = zeros(numSamples, numReactions, s_tot);

C = C0_true;
for i = 1:numSamples
    %% Rate laws
    R1k = [1, C(1), C(2), C(7), C(1)*C(2), C(2)*C(7), C(1)*C(7), C(1)*C(2)*C(7), C(1)*C(2)^2, C(1)^2*C(2)];
    R2k = [1, C(1), C(1)^2, C(1)*C(7), C(1)^2*C(7), C(7)];
    R3k = [1, C(1), C(1)^2, C(1)*C(7), C(1)^2*C(7), C(7)];
    R4k = [1, C(1), C(3), C(7), C(1)*C(3), C(3)*C(7), C(1)*C(7), C(1)*C(3)*C(7), C(1)*C(3)^2, C(1)^2*C(3)];

    R_k = zeros(numReactions, s_tot);
    R_k(1, 1:10) = R1k;
    R_k(2, 11:16) = R2k;
    R_k(3, 17:22) = R3k;
    R_k(4, 23:32) = R4k;
    
    % Store the candidate rate matrix for this sample time.
    R_candidates(i,:,:) = R_k;
    C = C_sol(i,:)';  % current state (7x1 vector for the 7 species)
end
beta_true = zeros(s_tot, 1);
beta_true(8) = alpha_true(1);
beta_true(15) = alpha_true(2);
beta_true(18) = alpha_true(3);

%% PART 4: Compute the discrete-time matrices (Equation (12))
n = size(C_sol,2);  % number of species (8)
A = -(Q/V) * eye(n);        % A = -0.1*I, since (Q/V)=0.1
B = (Q/V) * eye(n);         % B = 0.1*I
A_d = expm(A*T_s);        % Discrete state matrix, A_d = expm(-0.1*I)
% Using the form from the article, note that:
%    B_d = A^{-1}(A_d - I)*B, D_d = A^{-1}(A_d - I)
% Since A is negative, we write:
B_d = A \ (A_d - eye(n)) * B;  % B_d = I - A_d
D_d = A \ (A_d - eye(n));        % D_d = inv(A)*(A_d-I) > 0
% (For our parameters, A_d ≈ exp(-0.1)=0.9048, so B_d ≈ 0.09516*I, and D_d ≈ 10*(I-A_d) ≈ 0.9516*I.)

%% PART 5: Construct augmented regression matrix (as per Eq. (13))
NR = zeros(numSamples*n, s_tot);
H = zeros(numSamples*n, numSamples*n);
Theta = zeros(numSamples*n, n);
T = zeros(numSamples*n, numSamples*n);
Y = zeros(numSamples*n, 1);
U = zeros(numSamples*n, 1);
for k = 1:numSamples
    Y((k-1)*n+1:k*n,:) = C_sol(k, :)';
    U((k-1)*n+1:k*n,:) = C_in;
    
    NR((k-1)*n+1:k*n,:) = N_aug'*squeeze(R_candidates(k,:,:));
    H_row = zeros(n, numSamples*n);
    T_row = zeros(n, numSamples*n);
    for j = 1:k
        H_row(:,(j-1)*n+1:j*n) = A_d^(k-j)*D_d;
        T_row(:,(j-1)*n+1:j*n) = A_d^(k-j)*B_d;
    end
    H((k-1)*n+1:k*n,:) = H_row;
    T((k-1)*n+1:k*n,:) = T_row;
    Theta((k-1)*n+1:k*n,:) = A_d^k;
end
Y = Y - T*U;
Psi = H*NR; 
X = [Psi, Theta]; 

%setup check - Y - X*[beta_true; C_0] should be 0
error_true = norm(Y - X*[beta_true; C0_true.'], 2);

%% PART 6: Identification via nonnegative Lasso using the discretized data
w = [ones(s_tot, 1); zeros(n,1)];
lambda_max = max(Psi.' * Y / (sqrt(numSamples)*norm(Y, 2)));
numLambda = 100;
lambda_min = 1e-6; %lambda_max / numLambda;
lambda_grid = logspace(log10(lambda_min), log10(lambda_max), numLambda);

rho_opt = -1*ones(numLambda, 1);
beta_opt = zeros(numLambda, s_tot+n);
threshold = -1*ones(numLambda, 1);
for i=1:numLambda
    lambda = lambda_grid(i);
    %solve the lasso optimization problem
    %the original objective function is (-log(rho) + 1/(2*numSamples)*sum_square(rho*Y - X*phi) + lambda*w'*phi) 
    cvx_begin
        % Define variables
        variable phi(s_tot + n)  
        variable rho
        % Define the objective function
        minimize (-log(rho) + 1/(2*numSamples)*sum_square(rho*Y - X*phi) + lambda*w'*phi)  
        % Define constraints
        rho > 0;
        phi >= 0;
    cvx_end

    rho_min = rho;
    phi_min = phi;
    beta_min = phi/rho;
    support_index = zeros(size(beta_min));
    support_index_complement = zeros(size(beta_min));
    support_index(beta_min > 1e-6) = 1;
    support_index_complement(beta_min <= 1e-6) = 1;
    
    %find eigenvector
    %The original objective function is (1/numSamples * sum_square(X*v) /
    %(v'*v)), but CVX cannot solve quotient of two quadratic functions.
    %Instead we can find eigenvalues and then find the eigenvector that
    %satisfies the constraints by trial and error:
    [v_candidates, kappa_candidates] = eig(X.'*X);
    [kappa_candidates_sorted,ind] = sort(diag(kappa_candidates));
    v_candidates_sorted = v_candidates(:,ind);

    for j = 1:size(kappa_candidates_sorted,1)
        if (support_index_complement' - 3*support_index')*abs(v_candidates_sorted(:,j)) <= 0 && kappa_candidates_sorted(j) > 1e-9
            kappa = kappa_candidates_sorted(j);
            break;
        end
    end

    k_min = 3*lambda*sqrt(numReactions+n)/(sqrt(rho)*kappa);
    threshold(i,:) = k_min;

    %Update support index to eliminate params that don't meet threshold
    support_index = zeros(size(beta_min));
    support_index_complement = zeros(size(beta_min));
    support_index(beta_min > k_min) = 1;
    support_index_complement(beta_min <= k_min) = 1;

    %re-optimize parameters to only use parameters selected by support
    %index
    X_s = X(:, beta_min > k_min);
    if size(X_s, 2) == 0
        continue;
    end
    cvx_begin
        % Define variables
        variable phi(size(X_s, 2))  
        variable rho
        % Define the objective function
        minimize (-log(rho) + 1/(2*numSamples)*sum_square(rho*Y - X_s*phi) )  
        % Define constraints
        rho > 0;
        phi >= 0;
    cvx_end
    rho_opt(i,:) = rho;
    phi_aug = zeros(s_tot + n, 1);
    phi_aug(support_index == 1, :) = phi;
    beta_opt(i,:) = phi_aug/rho;
 
        beta_est = phi_aug/rho;
    K = w'*beta_est
    %feasibility check - terminate loop on first feasible solution
    if sum(support_index(1:10)) <=1 && sum(support_index(11:16)) <= 1 && sum(support_index(17:22)) <= 1 && sum(support_index(23:32)<= 1)
        break;
    end
    

 
end

%%PART 7: summarize results
figure;
semilogx(lambda_grid, beta_opt', 'LineWidth', 2);
xlabel('Lambda'); ylabel('Coefficient Value');
title('Coefficient Paths from Nonnegative Lasso (Discrete Model Data)');
legend({'k1','k2','k3','k4','k5','k1_h','k2_h','k4_h'}, 'Location', 'Best');

disp(' ');
disp('--- Discretization Matrices (Eq. (12)) ---');
disp('A_d (state matrix):'); disp(A_d);
disp('B_d (input matrix):'); disp(B_d);
disp('D_d (flux matrix):'); disp(D_d);

disp(' ');
disp('True Augmented Kinetic Parameters:');
disp(beta_true);
disp('Estimated Kinetic Parameters (Discrete Regression):');
disp(beta_est);
