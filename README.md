# Convex_Optimization-ADMM

# LASSO via ADMM (MATLAB)

This repository provides a MATLAB implementation of the LASSO optimization problem using the Alternating Direction Method of Multipliers (ADMM). It supports:

- **Synthetic regression** (`lasso.m`) using Gaussian random data.
- **Image reconstruction** (`lasso_img_main.m`) on grayscale images.
- **Formulation of Hybrid Precoding** (`convex_optimization_project.pdf`)
- **Metric outputs**: PSNR, SSIM, $\ell_2$ error.
- **Visual analysis**: convergence curves, histograms, error maps.

## Run Instructions
```matlab
run('lasso.m')            % For sparse signal recovery
run('lasso_img_main.m')   % For image denoising
```

## Applications
Sparse learning, reconstruction, feature selection, Hybrid Precoding in MIMO.
