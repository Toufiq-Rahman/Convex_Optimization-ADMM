% MATLAB Implementation of LASSO using ADMM for Noisy Image Reconstruction
% ---------------------------------------------------
% Goal: Reconstruct a noisy image using sparse approximation
% Solve: (1/2)||A*x - b||^2 + lambda * ||x||_1 using ADMM

clear; clc; close all;

%% -----------------------------
% Load and Add Noise to Image
img = imread('cameraman_.jpg');         % Load sample grayscale image
img = rgb2gray(img)
img = im2double(img);
[N1, N2] = size(img);                  % Get image size
x_true = img(:);                       % Vectorized clean image

noise_std = .3;
noisy_img = img + noise_std * randn(size(img));
noisy_img = min(max(noisy_img, 0), 1); % Clip to [0, 1]
b = noisy_img(:);                     % Noisy image vector (observation)
Nx = numel(x_true);

%% ADMM LASSO Solver Function (Denoising with Identity A)
function [x, z, y, primal_res, dual_res] = admm_lasso_identity(b, x, z, y, rho, lambda, max_iters, epsilon)
    Nx = numel(b);
    primal_res = zeros(max_iters, 1);
    dual_res = zeros(max_iters, 1);

    for k = 1:max_iters
        % x-update: (I + rho*I)^(-1) * (b + rho*(z - y))
        q = b + rho * (z - y);
        x = q / (1 + rho);

        % z-update: soft thresholding
        x_hat = x + y;
        z_old = z;
        z = sign(x_hat) .* max(abs(x_hat) - lambda / rho, 0);

        % y-update
        y = y + x - z;

        % residuals
        primal_res(k) = norm(x - z);
        dual_res(k) = rho * norm(z - z_old);
        if primal_res(k) < epsilon && dual_res(k) < epsilon
            break;
        end
    end
end

%% Initialization
x = zeros(Nx, 1);
z = zeros(Nx, 1);
y = zeros(Nx, 1);

%% Reconstruction for Varying Lambda
rho = 1;
lambdas = [.001,.01,.1];
figure;
for i = 1:length(lambdas)
    lambda = lambdas(i);
    [x_hat, ~, ~, primal_res, dual_res] = admm_lasso_identity(b, x, z, y, rho, lambda, 1000, 1e-4);
    recon_img = reshape(x_hat, N1, N2);
    % Plot reconstruction
    subplot(3, length(lambdas), i);
    imagesc(reshape(x_hat, N1, N2)); colormap gray; axis image off;
    title(sprintf('\\lambda = %.3f', lambda));

    % Plot residuals
    subplot(3, length(lambdas), i + length(lambdas));
    semilogy(primal_res, 'b'); hold on; semilogy(dual_res, 'r--');
    title('Residuals'); legend('Primal','Dual'); grid on;
    % Compute PSNR, SSIM, and L2 norm of absolute error
    psnr_val = psnr(recon_img, img);
    ssim_val = ssim(recon_img, img);
    abs_error_norm = norm(recon_img - img, 'fro');
    fprintf('lambda = %.4f → PSNR = %.4f dB, SSIM = %.4f, ||Error||_2 = %.4f\n', lambda, psnr_val, ssim_val, abs_error_norm);

    % Plot absolute error map
    subplot(3, length(lambdas), i + 2*length(lambdas));
    imagesc(abs(reshape(x_hat, N1, N2) - img)); colormap hot; axis image off;
    title('Abs Error');
end
sgtitle(sprintf('Denoising via LASSO-ADMM for Different \\lambda (\\rho = %.2f)', rho), 'Interpreter', 'tex');


%% Reconstruction for Varying Rho
lambda = 0.1;
rhos = [0.1, 0.5, 1.0, 5.0,10,100];
figure;
for i = 1:length(rhos)
    rho = rhos(i);
    [x_hat, ~, ~, primal_res, dual_res] = admm_lasso_identity(b, x, z, y, rho, lambda, 10, 1e-4);
    recon_img = reshape(x_hat, N1, N2);
    % Plot reconstruction
    subplot(3, length(rhos), i);
    imagesc(reshape(x_hat, N1, N2)); colormap gray; axis image off;
    title(sprintf('\\rho = %.4f', rho));

    % Plot residuals
    subplot(3, length(rhos), i + length(rhos));
    semilogy(primal_res, 'b'); hold on; semilogy(dual_res, 'r--');
    title('Residuals'); legend('Primal','Dual'); grid on;

    % Compute PSNR, SSIM, and L2 norm of absolute error
    psnr_val = psnr(recon_img, img);
    ssim_val = ssim(recon_img, img);
    abs_error_norm = norm(recon_img - img, 'fro');
    fprintf('rho = %.2f → PSNR = %.2f dB, SSIM = %.4f, ||Error||_2 = %.4f\n', rho, psnr_val, ssim_val, abs_error_norm);

    % Plot absolute error map
    subplot(3, length(rhos), i + 2*length(rhos));
    imagesc(abs(recon_img - img)); colormap hot; axis image off;
    title('Abs Error');
end
sgtitle(sprintf('Denoising via LASSO-ADMM for Different \\rho (\\lambda = %.2f)', lambda), 'Interpreter', 'tex');

%% Compare Clean, Noisy, and Reconstructed

rho=.5; 
lambda=.1;
[x_hat, ~, ~, primal_res, dual_res] = admm_lasso_identity(b, x, z, y, rho, lambda, 10, 1e-4);
figure;
% subplot(1, 3, 1);
% imagesc(img); colormap gray; axis image off; title('Original Image');
subplot(1, 2, 1);
imagesc(noisy_img); colormap gray; axis image off; title('Noisy Image');
subplot(1, 2, 2);
imagesc(reshape(x_hat, N1, N2)); colormap gray; axis image off; title('Best Reconstruction');
