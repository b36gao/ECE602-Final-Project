%% identify_SINDy.m
% This script implements the SINDy method for identifying nonlinear dynamic
% system models with non-negative parametsr. It enhances the standard SINDy process with the
% innovations from Al-Matouq & Vincent, 2020. Namely:
% 1) the fact that parameters are non-negative is exploited to convert the
% non-smooth absolute value regularization term into a linear term
% 2) objective function explicitely optimizes noise variance while
% remaining convex
% 3) an automatic and iterative process for tuning the lasso parameter (lambda) and
% sparsification threshold (k) is used. Threshold is determined based on
% noise variance.
%
% The SINDy method typically represent the system model as
% dX/dt = Theta(X)*E
% wehre dX/dt and X are measured. Theta represents a library of monomial functions
% of X, and Theta(X) is a matrix where each column is the result of
% applying one of the library functions to the X dataset. E is a matrix of
% coefficients where each column is the coeffients of the monomials that
% forms the governing equation of each component of dX/dt
% 
% The dX/dt term is
% analogous to the Y matrix and Theta(X) is analogous to the Psi matrix in the acetone example. 
% The overall regression equation can be
% rearranged into the same form as the acetone example, by flattening
% dX/dt and E into a stack of vectors representing each component. Theta
% would be flattened horizontally so that each row represents the basis
% functions whose linear combination represents the effect of past and
% present X on dX/dt
%


clear; clc; close all;

%% PART 1: Load the simulation dataset
load('simulated_SINDy_dataset.mat', 'dataset');
t_sol = dataset.t;      % time vector
dXdt = dataset.dXdt;      
X = dataset.X;
Theta = dataset.Theta;
eta_true = dataset.eta;  % matrix of true coefficients
numSamples   = length(t_sol);
numFunctions = size(eta_true, 1);
n = size(eta_true, 2); %number of state variables
s_tot = numFunctions;  %total number of candidate monomials

beta_true = eta_true(:); % flatten eta by stacking the coefficients for each state variable

%% PART 3: construct monomial library 

Y = dXdt;
Psi = [Theta, -Theta];


%% PART 4: Identification via nonnegative Lasso using the discretized data
PsiY = Psi.'*Y;
lambda_max = max(PsiY(:)) / (sqrt(numSamples)*norm(Y, 2));
numLambda = 30;
lambda_min = 1e-6;
lambda_grid = logspace(log10(lambda_min), log10(lambda_max), numLambda);

var_opt = -1*ones(numLambda, n);
beta_opt = zeros(numLambda, s_tot*n);
threshold = -1*ones(numLambda, 1);
w = ones(s_tot*2, 1);
for i=1:numLambda
    lambda = lambda_grid(i);
    beta_est = zeros(s_tot, n);
    for x = 1 : n
        %solve the lasso optimization problem
        %the original objective function is (-log(rho) + 1/(2*numSamples)*sum_square(rho*Y - X*phi) + lambda*w'*phi) 
        cvx_begin
            % Define variables
            variable phi(s_tot*2, 1)  
            variable rho
            % Define the objective function
            minimize (-log(rho) + 1/(2*numSamples)*sum_square(rho*Y(:, x) - Psi*phi) + lambda*w.'*phi)  
            % Define constraints
            subject to
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
        [v_candidates, kappa_candidates] = eig(Psi.'*Psi);
        [kappa_candidates_sorted,ind] = sort(diag(kappa_candidates));
        v_candidates_sorted = v_candidates(:,ind);
    
        for j = 1:size(kappa_candidates_sorted,1)
            if all((support_index_complement' - 3*support_index')*abs(v_candidates_sorted(:,j)) <= 0) && kappa_candidates_sorted(j) > 1e-9
                kappa = kappa_candidates_sorted(j);
                break;
            end
        end
    
        k_min = 3*lambda*sqrt(s_tot + n)/(sqrt(rho)*kappa);
        threshold(i,:) = k_min;
    
        %Update support index to eliminate params that don't meet threshold
        support_index = zeros(size(beta_min));
        support_index_complement = zeros(size(beta_min));
        support_index(beta_min > k_min) = 1;
        support_index_complement(beta_min <= k_min) = 1;
    
        %re-optimize parameters to only use parameters selected by support
        %index
        Psi_s = Psi(:, beta_min > k_min);
        if size(Psi_s, 2) == 0
            continue;
        end
        cvx_begin
            % Define variables
            variable phi(size(Psi_s, 2), 1)  
            variable rho
            % Define the objective function
            minimize (-log(rho) + 1/(2*numSamples)*sum_square(rho*Y(:, x) - Psi_s*phi) )  
            % Define constraints
            subject to
                rho > 0;
                phi >= 0;
        cvx_end
        var_opt(i,x) = 1/rho;
        phi_aug = zeros(2*s_tot, 1);
        phi_aug(support_index == 1, :) = phi;
        beta_est(:, x) = (phi_aug(1:s_tot,:) - phi_aug(s_tot+1:2*s_tot,:))/rho;
    end
 
    %no need for feasibility check
    beta_opt(i,:) = beta_est(:).';
end

%%PART 8: summarize results
figure;
semilogx(lambda_grid, beta_opt', 'LineWidth', 2);
xlabel('Lambda'); ylabel('Coefficient Value');
title('Coefficient Paths from Nonnegative Lasso (Discrete Model Data)');

disp(' ');
disp('True Augmented Kinetic Parameters:');
disp(beta_true);
disp('Estimated Kinetic Parameters (Discrete Regression):');
disp(beta_est);
