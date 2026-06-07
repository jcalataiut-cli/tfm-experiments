#!/usr/bin/env python3
"""
Experimento 2: Comparación de Kernels (NNGP, NTK, ancho finito)
================================================================
Compara los kernels NNGP y NTK analíticos (límite de ancho infinito)
con kernels empíricos de ancho finito para una red de una capa oculta
con activación ReLU.

Genera figuras en experiments/figures/exp02/
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- configuración ----------
PROJECT = "/home/radxa/Research/TFM-Advanced Mathematics"
FIG_DIR = os.path.join(PROJECT, "experiments", "figures", "exp02")
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# ---------- parámetros ----------
d_input = 20           # dimensión de entrada (más baja que exp1 para matrices grandes)
n_points = 30          # puntos de datos para el conjunto de test
widths = [1, 5, 10, 50, 100, 500, 1000, 2000]
n_ensembles = 1000     # máximo de ensambles para promedio del kernel empírico

# ---------- funciones auxiliares ----------
def relu(x):
    return np.maximum(x, 0.0)

def nngp_kernel_matrix_relu(X, sigma_w=1.0, sigma_b=1.0):
    """
    Matriz NNGP analítica para una capa ReLU.
    K_ij = sigma_w**2 * E_{z}[ReLU(z_i) ReLU(z_j)],
    con z_i ~ N(0, sigma_w**2 * ||x_i||**2 / d + sigma_b**2)
    y Cov(z_i, z_j) = sigma_w**2 * (x_i·x_j) / d + sigma_b**2

    Para ReLU: E[ReLU(z_i) ReLU(z_j)] = 
        (sigma_i * sigma_j / (2*pi)) * (sin(theta) + (pi - theta) * cos(theta))
    donde theta = arccos(Cov(z_i,z_j) / (sigma_i * sigma_j))
    """
    N = X.shape[0]
    K = np.zeros((N, N))
    norms2 = np.linalg.norm(X, axis=1)**2
    dots = X @ X.T
    d = X.shape[1]

    var_w_per_dim = sigma_w**2 / d

    for i in range(N):
        for j in range(N):
            var_i = var_w_per_dim * norms2[i] + sigma_b**2
            var_j = var_w_per_dim * norms2[j] + sigma_b**2
            cov_ij = var_w_per_dim * dots[i, j] + sigma_b**2

            sigma_i = np.sqrt(var_i)
            sigma_j = np.sqrt(var_j)

            cos_theta = cov_ij / (sigma_i * sigma_j + 1e-12)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            # Cho & Saul formula for ReLU
            expectation = (sigma_i * sigma_j / (2 * np.pi)) * (
                np.sin(theta) + (np.pi - theta) * cos_theta)
            K[i, j] = sigma_w**2 * expectation

    return K

def ntk_matrix_relu(X, sigma_w=1.0, sigma_b=1.0):
    """
    Matriz NTK analítica para una capa ReLU.
    Θ(x, x') = K^{NNGP}(x, x') + σ_w² * (x·x'/d) * E[σ'(z) σ'(z')]
    
    Para ReLU: E[σ'(z_i) σ'(z_j)] = P(z_i > 0, z_j > 0) = (1/2π) * (π - θ)
    donde cos θ = Cov(z_i,z_j) / (σ_i * σ_j)
    y z ~ N(0, σ_w²||x||²/d + σ_b²)
    """
    N = X.shape[0]
    K_nngp = nngp_kernel_matrix_relu(X, sigma_w, sigma_b)
    norms2 = np.linalg.norm(X, axis=1)**2
    dots = X @ X.T
    d = X.shape[1]
    var_w_per_dim = sigma_w**2 / d

    K_ntk = K_nngp.copy()
    for i in range(N):
        for j in range(N):
            var_i = var_w_per_dim * norms2[i] + sigma_b**2
            var_j = var_w_per_dim * norms2[j] + sigma_b**2
            cov_ij = var_w_per_dim * dots[i, j] + sigma_b**2

            sigma_i = np.sqrt(var_i)
            sigma_j = np.sqrt(var_j)

            cos_theta = cov_ij / (sigma_i * sigma_j + 1e-12)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            # Derivada de ReLU: P(z > 0, z' > 0)
            exp_deriv = (1.0 / (2 * np.pi)) * (np.pi - theta)
            # Término extra del NTK
            extra = sigma_w**2 * dots[i, j] / d * exp_deriv
            K_ntk[i, j] += extra

    return K_ntk

def empirical_kernel_single_layer(X, width, sigma_w=1.0, sigma_b=1.0, n_ensembles=100,
                                    compute_ntk=False):
    """
    Kernel empírico para red de 1 capa.
    compute_ntk=True también computa el NTK empírico.
    """
    N = X.shape[0]
    K_emp = np.zeros((N, N))
    NTK_emp = np.zeros((N, N)) if compute_ntk else None
    
    total_budget = 200000
    n_ens = max(200, min(n_ensembles, total_budget // width))
    n_ens = min(n_ens, n_ensembles)
    d = X.shape[1]
    
    for m in range(n_ens):
        V = np.random.randn(width, d) * sigma_w / np.sqrt(d)
        b = np.random.randn(width) * sigma_b
        w_out = np.random.randn(width) * sigma_w
        
        pre = X @ V.T + b[np.newaxis, :]        # (N, width)
        h = relu(pre)                             # (N, width)
        f = (h * w_out[np.newaxis, :]).sum(axis=1) / np.sqrt(width)  # (N,)
        
        K_emp += np.outer(f, f)
        
        if compute_ntk:
            h_prime = (pre > 0).astype(float)       # (N, width)
            
            ntk_w = (h @ h.T) / width               # (N, N) contribución w_out
            
            w_out_sq = w_out ** 2                    # (width,)
            weighted_hprime = h_prime * w_out_sq[np.newaxis, :]  # (N, width)
            ntk_bv_base = weighted_hprime @ h_prime.T            # (N, N)
            
            dots_scaled = (X @ X.T) / d              # (N, N)
            ntk_bv = ntk_bv_base * (1.0 + dots_scaled) / width  # bias + V
            
            NTK_emp += ntk_w + ntk_bv
    
    K_emp /= n_ens
    if compute_ntk:
        NTK_emp /= n_ens
    return (K_emp, NTK_emp) if compute_ntk else K_emp

# ---------- generar dataset ----------
X = np.random.randn(n_points, d_input)
X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d_input)

print("=" * 65)
print("Experimento 2: Comparación de Kernels (NNGP vs NTK vs ancho finito)")
print("=" * 65)

# ---------- kernels analíticos ----------
print("\nComputando kernels analíticos (límite de ancho infinito)...")
t0 = time.time()
K_nngp = nngp_kernel_matrix_relu(X)
K_ntk = ntk_matrix_relu(X)
t1 = time.time()
print(f"  NNGP kernel: {K_nngp.shape}, rango = {np.linalg.matrix_rank(K_nngp, tol=1e-10)}")
print(f"  NTK:         {K_ntk.shape}, rango = {np.linalg.matrix_rank(K_ntk, tol=1e-10)}")
print(f"  Tiempo: {t1-t0:.2f}s")

# ---------- kernels empíricos para distintos anchos ----------
print("\nComputando kernels empíricos de ancho finito...")
empirical_kernels = {}
empirical_ntks = {}
errors_nngp = {}
errors_ntk = {}
errors_ntk_emp = {}

for n in widths:
    t0 = time.time()
    K_emp, NTK_emp = empirical_kernel_single_layer(X, n, n_ensembles=n_ensembles,
                                                     compute_ntk=True)
    t1 = time.time()
    empirical_kernels[n] = K_emp
    empirical_ntks[n] = NTK_emp
    
    # Error vs NNGP (output covariance)
    err_nngp = np.linalg.norm(K_emp - K_nngp, 'fro') / np.linalg.norm(K_nngp, 'fro')
    errors_nngp[n] = err_nngp
    
    # Error vs NTK (gradient kernel)
    err_ntk = np.linalg.norm(K_emp - K_ntk, 'fro') / np.linalg.norm(K_ntk, 'fro')
    errors_ntk[n] = err_ntk
    
    # Error del NTK empírico vs NTK analítico
    err_ntk_emp = np.linalg.norm(NTK_emp - K_ntk, 'fro') / np.linalg.norm(K_ntk, 'fro')
    errors_ntk_emp[n] = err_ntk_emp
    
    print(f"  n = {n:5d}  |  ε_F(NNGP) = {err_nngp:.5f}  "
          f"ε_F(NTK-emp) = {err_ntk_emp:.5f}  [{t1-t0:.1f}s]")

# ---------- Figuras ----------
print("\nGenerando figuras...")

# 1. Mapas de calor
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
plot_kernels = [('NNGP (∞)', K_nngp), ('NTK (∞)', K_ntk)]
plot_kernels += [(f'n = {n}', empirical_kernels[n]) for n in [10, 100, 500, 1000, 2000]]

vmin = 0
vmax = max(K_ntk.max(), max(empirical_kernels[n].max() for n in widths))

for idx, (label, K) in enumerate(plot_kernels):
    ax = axes[idx // 4, idx % 4]
    im = ax.imshow(K, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(label, fontsize=11)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, shrink=0.75)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'kernel_matrices.png'), dpi=150)
plt.close()
print("  ✓ kernel_matrices.png")

# 2. Convergencia de errores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n_vals = np.array(widths)
err_nngp_arr = np.array([errors_nngp[n] for n in widths])
err_ntk_emp_arr = np.array([errors_ntk_emp[n] for n in widths])

# Error de kernel empírico vs NNGP
ax = axes[0]
ax.loglog(n_vals, err_nngp_arr, 'o-', color='#3498db', markersize=8, 
          label='NNGP empírico vs NNGP teórico')
coeff_nngp = np.polyfit(np.log(n_vals), np.log(err_nngp_arr), 1)
ax.loglog(n_vals, np.exp(coeff_nngp[1]) * n_vals**coeff_nngp[0], '--',
          color='#3498db', alpha=0.5, label=f'∼ n^{coeff_nngp[0]:.2f}')
ax.set_xlabel('Ancho n', fontsize=12)
ax.set_ylabel('Error relativo de Frobenius', fontsize=12)
ax.set_title('Kernel de salida → NNGP', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Error del NTK empírico vs NTK analítico
ax = axes[1]
ax.loglog(n_vals, err_ntk_emp_arr, 's-', color='#e74c3c', markersize=8,
          label='NTK empírico vs NTK teórico')
coeff_ntk = np.polyfit(np.log(n_vals), np.log(err_ntk_emp_arr), 1)
ax.loglog(n_vals, np.exp(coeff_ntk[1]) * n_vals**coeff_ntk[0], '--',
          color='#e74c3c', alpha=0.5, label=f'∼ n^{coeff_ntk[0]:.2f}')
ax.set_xlabel('Ancho n', fontsize=12)
ax.set_ylabel('Error relativo de Frobenius', fontsize=12)
ax.set_title('NTK empírico → NTK teórico', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'convergence_error.png'), dpi=150)
plt.close()
print("  ✓ convergence_error.png")

# 3. Espectro de autovalores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (name, K_inf, color_inf, emp_dict) in enumerate([
    ('NNGP', K_nngp, '#3498db', empirical_kernels),
    ('NTK', K_ntk, '#e74c3c', empirical_ntks)
]):
    ax = axes[ax_idx]
    eigs_inf = np.linalg.eigvalsh(K_inf)[::-1]
    ax.semilogy(range(1, len(eigs_inf)+1), eigs_inf, 'o-', color=color_inf,
                markersize=6, linewidth=2, label='Ancho ∞')
    
    for n, style, color_n in [(10, 's--', '#e67e22'), (100, '^--', '#2ecc71'), (1000, 'd--', '#9b59b6')]:
        K_n = emp_dict[n]
        eigs_n = np.linalg.eigvalsh(K_n)[::-1]
        ax.semilogy(range(1, len(eigs_n)+1), eigs_n, style, color=color_n, alpha=0.6, label=f'n={n}')
    
    ax.set_xlabel('Índice de autovalor', fontsize=11)
    ax.set_ylabel('Autovalor', fontsize=11)
    ax.set_title(f'Espectro {name}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eigenvalue_spectra.png'), dpi=150)
plt.close()
print("  ✓ eigenvalue_spectra.png")

# 4. Diferencia NTK - NNGP
fig, ax = plt.subplots(figsize=(6, 5))
diff = K_ntk - K_nngp
im = ax.imshow(diff, cmap='RdBu_r')
ax.set_title('NTK - NNGP (teóricos)', fontsize=13)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ntk_minus_nngp.png'), dpi=150)
plt.close()
print("  ✓ ntk_minus_nngp.png")

# ---------- Guardar resultados ----------
import json
results = {
    'widths': widths,
    'errors_nngp': {str(n): float(errors_nngp[n]) for n in widths},
    'errors_ntk_emp': {str(n): float(errors_ntk_emp[n]) for n in widths},
    'nngp_rank': int(np.linalg.matrix_rank(K_nngp, tol=1e-10)),
    'ntk_rank': int(np.linalg.matrix_rank(K_ntk, tol=1e-10)),
}

with open(os.path.join(FIG_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Resultados guardados en {FIG_DIR}/")
print("Experimento 2 completado.\n")
