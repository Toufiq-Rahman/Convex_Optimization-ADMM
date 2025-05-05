% MATLAB Implementation of LASSO using ADMM
% ---------------------------------------------------
% Solve: (1/2)||Ax - b||^2 + lambda * ||x||_1 using ADMM

clear; clc;close all;

%% -----------------------------
% Parameters and Data Generation
% -----------------------------
Nx = 1000;                      % Number of predictors (features)
Nb = 1000;                      % Number of samples (observations)
max_iters = 3000;               % Maximum number of ADMM iterations
epsilon = 1e-9;                % Convergence threshold

% Generate synthetic dataset
rng(70);                        % Reproducibility
A = randn(Nb, Nx);              % Measurement matrix (Gaussian)
true_coeffs = randn(Nx, 1);     % Ground-truth sparse coefficients
b = A * true_coeffs + 0.5 * randn(Nb, 1);  % Response with noise

% Initialization of variables
x = zeros(Nx, 1);               % Primal variable
z = zeros(Nx, 1);               % Auxiliary variable
y = zeros(Nx, 1);               % Dual variable (Lagrange multiplier)

% Precompute matrices for speed
AtA = A' * A;
Atb = A' * b;

%% --------------------------------------
% ADMM LASSO Solver Function
%% --------------------------------------
function [x, z, y, primal_res, dual_res] = admm_lasso(A, AtA, Atb, b, x, z, y, rho, lambda, max_iters, epsilon)
    Nx = size(A,2);
    L = chol(AtA + rho * eye(Nx), 'lower'); % Cholesky decomposition
    U = L';
    primal_res = zeros(max_iters, 1);
    dual_res = zeros(max_iters, 1);

    for k = 1:max_iters
        % x-update: solve quadratic subproblem
        q = Atb + rho * (z - y);
        x = U \ (L \ q);

        % z-update: soft thresholding
        x_hat = x + y;
        z_old = z;
        z = sign(x_hat) .* max(abs(x_hat) - lambda / rho, 0);

        % y-update: dual variable
        y = y + x - z;

        % Compute residuals
        primal_res(k) = norm(x - z);
        dual_res(k) = rho * norm(z - z_old);

        % Convergence check
        if primal_res(k) < epsilon && dual_res(k) < epsilon
            break;
        end
    end
end



%% --------------------------------
% Case 1: Varying rho, fixed lambda
%% --------------------------------
lambda = 0.1;
figure;
for i = 0:5
    rho = round(0.1 + 0.1*i, 1);
    [x, z, y, primal_res, dual_res] = admm_lasso(A, AtA, Atb, b, x, z, y, rho, lambda, max_iters, epsilon);
    subplot(2,3,i+1);
    semilogy(primal_res, 'b'); hold on;
    semilogy(dual_res, 'r--');
    title(sprintf('\\rho = %.1f', rho));
    xlabel('Iteration'); ylabel('Residual');
    legend('Primal', 'Dual'); grid on;
end
sgtitle('Case 1: ADMM Convergence for Varying \rho (\lambda=0.1)');

%% --------------------------------
% Case 2: Varying lambda, fixed rho
%% --------------------------------
rho = 1.0;
figure;
for i = 0:5
    lambda = 0.0001 * 10^i;
    [x, z, y, primal_res, dual_res] = admm_lasso(A, AtA, Atb, b, x, z, y, rho, lambda, max_iters, epsilon);
    subplot(2,3,i+1);
    semilogy(primal_res, 'b'); hold on;
    semilogy(dual_res, 'r--');
    title(sprintf('\\lambda = %.4f', lambda));
    xlabel('Iteration'); ylabel('Residual');
    legend('Primal', 'Dual'); grid on;
end
sgtitle('Case 2: ADMM Convergence for Varying \lambda (\rho=1.0)');


%% --------------------------
% Case 3: Fixed lambda, rho
%% --------------------------
lambda = 0.1; rho = 1.0;
[x, z, y, primal_res, dual_res] = admm_lasso(A, AtA, Atb, b, x, z, y, rho, lambda, max_iters, epsilon);
figure;
semilogy(primal_res, 'b'); hold on;
semilogy(dual_res, 'r--');
title('Case 3: ADMM Convergence (\lambda=0.1, \rho=1.0)');
legend('Primal residual', 'Dual residual');
xlabel('Iteration'); ylabel('Residual (log scale)'); grid on;

%% ---------------------------------------------------------
% Case 4: Histogram Visualization of b, estimated x, and true_coeffs
%% ---------------------------------------------------------
lambda = 0.001; rho = 1.0;
[x, z, y, primal_res, dual_res] = admm_lasso(A, AtA, Atb, b, x, z, y, rho, lambda, max_iters, epsilon);
figure;
subplot(1,3,1);
histogram(b, 50, 'FaceColor', 'blue');
title('Value Distribution of Vector b');
xlabel('Value'); ylabel('Frequency'); grid on;
subplot(1,3,2);
histogram(x, 50, 'FaceColor', 'green');
title('Value Distribution of Estimated x');
xlabel('Value'); ylabel('Frequency'); grid on;
subplot(1,3,3);
histogram(true_coeffs, 50, 'FaceColor', 'red');
title('Value Distribution of True Coeffs');
xlabel('Value'); ylabel('Frequency'); grid on;

%% ----------------------
% Inspect Input Matrices
%% ----------------------
figure;
subplot(1,3,1);
histogram(A(:), 50); title('Matrix A'); xlabel('Value'); ylabel('Freq'); grid on;
subplot(1,3,2);
histogram(b, 50); title('Vector b'); xlabel('Value'); ylabel('Freq'); grid on;
subplot(1,3,3);
histogram(true_coeffs, 50); title('True Coeffs'); xlabel('Value'); ylabel('Freq'); grid on;

%% --------------------------------------
% Final Plot: Estimated vs True Coeffs
%% --------------------------------------
figure;
plot(true_coeffs, 'b'); hold on;
plot(x, 'r--');
legend('True Coefficients', 'Estimated');
title('LASSO via ADMM: True vs Estimated'); xlabel('Index'); ylabel('Value'); grid on;

%% --------------------------------------
% Case 5: Error Distribution Visualization
% Error Distribution Visualization
%% --------------------------------------
abs_error = abs(x - true_coeffs);

figure;

% Histogram of absolute error
subplot(1,2,1);
histogram(abs_error, 50, 'FaceColor', [1 0.4 0.7]); % pinkish
xlabel('Absolute Error');
ylabel('Frequency');
title('Histogram of Absolute Error (|x - true\_coeffs|)');
grid on;

% Boxplot of absolute error
subplot(1,2,2);
boxplot(abs_error);
ylabel('|x - true\_coeffs|');
title('Boxplot of Absolute Error');
grid on;


