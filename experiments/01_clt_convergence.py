#!/usr/bin/env python3
"""
Experimento 1: Convergencia CLT (Central Limit Theorem)
=========================================================
Demuestra que la distribución de salida de una red neuronal
(un solo paso de random features) converge a una Gaussiana
a medida que el ancho n → ∞ (límite NNGP).

Genera figuras en experiments/figures/exp01/
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, entropy

# ---------- configuración ----------
PROJECT = "/home/radxa/Research/TFM-Advanced Mathematics"
FIG_DIR = os.path.join(PROJECT, "experiments", "figures", "exp01")
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# ---------- parámetros ----------
d_input = 50          # dimensión de entrada
n_samples = 20000     # muestras Monte Carlo por ancho
widths   = [1, 5, 10, 50, 100, 500, 1000, 5000]
# usar menos muestras para anchos grandes
n_samples_list = [50000 if w <= 10 else 20000 if w <= 100 else 8000 for w in widths]

# ---------- función de activación ----------
def activation(x):
    """ReLU"""
    return np.maximum(x, 0.0)

# ---------- kernel NNGP analítico para ReLU (un solo paso) ----------
def nngp_kernel_relu(x1, x2, sigma_w=1.0, sigma_b=1.0):
    """
    NNGP kernel analítico para una capa ReLU.
    K(x, x') = (σ_b^2 + σ_w^2 * E[σ(z1)σ(z2)]) / d_input
    donde z1, z2 son features gaussianas.
    
    Para ReLU: E[σ(z1)σ(z2)] = (σ_1 σ_2 / 2π) * (sin θ + (π-θ) cos θ)
    donde cos θ = (z1·z2) / (σ_1 σ_2)
    """
    # Pre-normalización: asumimos x1, x2 normalizados
    dot = np.dot(x1, x2)
    norm2_x1 = np.dot(x1, x1)
    norm2_x2 = np.dot(x2, x2)
    
    # Si las entradas son vectores normalizados (norma ~ d_input)
    sigma_1 = np.sqrt(sigma_w**2 * norm2_x1 / d_input + sigma_b**2)
    sigma_2 = np.sqrt(sigma_w**2 * norm2_x2 / d_input + sigma_b**2)
    
    if sigma_1 < 1e-10 or sigma_2 < 1e-10:
        return 0.0
    
    cos_theta = (sigma_w**2 * dot / d_input + sigma_b**2) / (sigma_1 * sigma_2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Fórmula de Cho & Saul (ReLU)
    expectation = sigma_1 * sigma_2 / (2 * np.pi) * (np.sin(theta) + (np.pi - theta) * cos_theta)
    return expectation

# ---------- forward pass de random features ----------
def random_features_forward(x, width, sigma_w=1.0, sigma_b=1.0):
    """
    Red de 1 capa oculta: f(x) = (1/√width) Σ w_j σ(v_j·x/√d + b_j)
    con w_j ~ N(0,σ_w²), v_jd ~ N(0,σ_w²/d_input), b_j ~ N(0,σ_b²)
    """
    d = x.shape[0]
    # Pesos y sesgos de entrada
    V = np.random.randn(width, d) * sigma_w / np.sqrt(d)
    b = np.random.randn(width) * sigma_b
    # Pesos de salida
    w = np.random.randn(width) * sigma_w
    
    # Forward
    pre = V @ x + b
    h = activation(pre)
    return np.sum(w * h) / np.sqrt(width)

# ---------- análisis de convergencia ----------
print("=" * 65)
print("Experimento 1: Convergencia CLT al límite NNGP")
print("=" * 65)

# Fijamos un solo input para este experimento
# Normalizamos para que ||x||² ≈ d_input
x_test = np.random.randn(d_input)
x_test = x_test / np.sqrt(np.dot(x_test, x_test) / d_input)

# Valor teórico: varianza NNGP
k_xx = nngp_kernel_relu(x_test, x_test)
print(f"\nInput dim d = {d_input}")
print(f"NNGP kernel teórico K(x,x) = {k_xx:.6f}")
print(f"σ teórica = {np.sqrt(k_xx):.6f}")

results = {}

for idx, n in enumerate(widths):
    t0 = time.time()
    ns = n_samples_list[idx]
    
    # Recolectamos ns muestras de la salida
    outputs = np.zeros(ns)
    for i in range(ns):
        outputs[i] = random_features_forward(x_test, n)
    
    t1 = time.time()
    
    # Estadísticas
    mu_emp = np.mean(outputs)
    var_emp = np.var(outputs)
    std_emp = np.std(outputs)
    
    # KL divergence (aproximada mediante histograma vs Gaussiana)
    # Histograma
    bins = 80
    counts, edges = np.histogram(outputs, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    # Gaussiana teórica
    gauss_pdf = norm.pdf(bin_centers, loc=0.0, scale=np.sqrt(k_xx))
    
    # KL(P || Q)  (evitando log(0))
    eps = 1e-12
    p = counts + eps
    q = gauss_pdf + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    kl_div = np.sum(p * np.log(p / q))
    
    # Kolmogorov-Smirnov test contra N(0, k_xx)
    ks_stat, ks_pval = kstest(outputs, 'norm', args=(0.0, np.sqrt(k_xx)))
    
    # También máxima desviación de cumulante
    skew = np.mean(((outputs - mu_emp) / std_emp)**3) if std_emp > 1e-10 else 0.0
    kurt = np.mean(((outputs - mu_emp) / std_emp)**4) - 3.0 if std_emp > 1e-10 else 0.0
    
    results[n] = {
        'mu': mu_emp, 'var': var_emp, 'std': std_emp,
        'kl': kl_div, 'ks_stat': ks_stat, 'ks_pval': ks_pval,
        'skew': skew, 'kurt': kurt, 'time': t1 - t0, 'n_samples': ns
    }
    
    print(f"\n  n = {n:5d}  |  μ = {mu_emp:+.5f}  σ² = {var_emp:.5f}  "
          f"skew = {skew:+.4f}  kurt = {kurt:+.4f}")
    print(f"           KL = {kl_div:.6f}  KS = {ks_stat:.4f}  "
          f"(p = {ks_pval:.4f})  [{ns} muestras en {t1-t0:.1f}s]")

# ---------- Figuras ----------
print("\nGenerando figuras...")

# 1. Histogramas comparativos para anchos seleccionados
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
selected_widths = [1, 10, 100, 5000]
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']

for idx, (n, ax) in enumerate(zip(selected_widths, axes.flatten())):
    ns = n_samples_list[widths.index(n)]
    outputs = np.zeros(ns)
    for i in range(ns):
        outputs[i] = random_features_forward(x_test, n)
    
    mu_emp = np.mean(outputs)
    std_emp = np.std(outputs)
    
    # Histograma
    ax.hist(outputs, bins=60, density=True, alpha=0.6,
            color=colors[idx], label=f'n = {n}')
    
    # Gaussiana NNGP teórica
    x_range = np.linspace(-4*np.sqrt(k_xx), 4*np.sqrt(k_xx), 500)
    ax.plot(x_range, norm.pdf(x_range, 0.0, np.sqrt(k_xx)),
            'k--', linewidth=2, label='NNGP (ancho ∞)')
    
    ax.set_xlabel('Salida f(x)', fontsize=11)
    ax.set_ylabel('Densidad', fontsize=11)
    ax.set_title(f'n = {n}', fontsize=13)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'histograms.png'), dpi=150)
plt.close()
print("  ✓ histograms.png")

# 2. Convergencia de métricas
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Extraer datos
n_vals = np.array(widths)
kl_vals = np.array([results[n]['kl'] for n in widths])
ks_vals = np.array([results[n]['ks_stat'] for n in widths])
var_vals = np.array([results[n]['var'] for n in widths])
mu_vals = np.array([results[n]['mu'] for n in widths])
skew_vals = np.array([results[n]['skew'] for n in widths])
kurt_vals = np.array([results[n]['kurt'] for n in widths])

# KL divergence
ax = axes[0, 0]
ax.semilogy(n_vals, kl_vals, 'o-', color='#e74c3c', markersize=8)
ax.axhline(0.001, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Ancho n', fontsize=11)
ax.set_ylabel('KL(P || N(0,K))', fontsize=11)
ax.set_title('Divergencia KL', fontsize=13)
ax.grid(True, alpha=0.3)

# Varianza
ax = axes[0, 1]
ax.semilogx(n_vals, var_vals, 's-', color='#3498db', markersize=8, label='Empírica')
ax.axhline(k_xx, color='k', linestyle='--', label=f'NNGP = {k_xx:.4f}')
ax.set_xlabel('Ancho n', fontsize=11)
ax.set_ylabel('Varianza', fontsize=11)
ax.set_title('Convergencia de varianza', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Media
ax = axes[0, 2]
ax.semilogx(n_vals, np.abs(mu_vals), '^--', color='#2ecc71', markersize=8)
ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Ancho n', fontsize=11)
ax.set_ylabel('|Media empírica|', fontsize=11)
ax.set_title('Convergencia de media a 0', fontsize=13)
ax.grid(True, alpha=0.3)

# KS statistic
ax = axes[1, 0]
ax.loglog(n_vals, ks_vals, 'o-', color='#e67e22', markersize=8)
ax.set_xlabel('Ancho n', fontsize=11)
ax.set_ylabel('Estadístico KS', fontsize=11)
ax.set_title('Kolmogorov-Smirnov', fontsize=13)
ax.grid(True, alpha=0.3)

# Skewness
ax = axes[1, 1]
ax.semilogx(n_vals, skew_vals, 'd-', color='#9b59b6', markersize=8)
ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Ancho n', fontsize=11)
ax.set_ylabel('Skewness', fontsize=11)
ax.set_title('Asimetría (skewness)', fontsize=13)
ax.grid(True, alpha=0.3)

# Kurtosis
ax = axes[1, 2]
ax.semilogx(n_vals, kurt_vals, 'p-', color='#1abc9c', markersize=8)
ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Ancho n', fontsize=11)
ax.set_ylabel('Exceso de curtosis', fontsize=11)
ax.set_title('Curtosis (kurtosis - 3)', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'convergence_metrics.png'), dpi=150)
plt.close()
print("  ✓ convergence_metrics.png")

# 3. Q-Q plot para el ancho más grande
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
qq_widths = [10, 100, 5000]
for idx, n in enumerate(qq_widths):
    ns = n_samples_list[widths.index(n)]
    outputs = np.zeros(ns)
    for i in range(ns):
        outputs[i] = random_features_forward(x_test, n)
    
    outputs_sorted = np.sort(outputs)
    theoretical_q = norm.ppf(np.linspace(0.001, 0.999, ns), 0.0, np.sqrt(k_xx))
    
    ax = axes[idx]
    ax.scatter(theoretical_q, outputs_sorted, s=1, alpha=0.3, c='#3498db')
    lim = max(np.abs(outputs_sorted).max(), np.abs(theoretical_q).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlabel('Cuantiles teóricos N(0,K)', fontsize=10)
    ax.set_ylabel('Cuantiles empíricos', fontsize=10)
    ax.set_title(f'Q-Q plot: n = {n}', fontsize=12)
    ax.axis('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'qq_plots.png'), dpi=150)
plt.close()
print("  ✓ qq_plots.png")

# ---------- Guardar resultados para el informe ----------
import json
with open(os.path.join(FIG_DIR, 'results.json'), 'w') as f:
    # Convertir numpy types a float
    clean = {}
    for n, r in results.items():
        clean[str(n)] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in r.items()}
    clean['nngp_kernel_xx'] = float(k_xx)
    json.dump(clean, f, indent=2)

print(f"\n✓ Resultados guardados en {FIG_DIR}/")
print("Experimento 1 completado.\n")
