# tfm-experiments

Experimentación numérica para TFM sobre generalización del teorema NNGP/NTK.

## 📓 Notebooks Jupyter

| Notebook | Descripción |
|----------|-------------|
| `01_clt_convergence.ipynb` | Verificación del **Teorema Central del Límite** en espacio de parámetros. Mide convergencia a la normal con Wasserstein, KS, QQ-plot. |
| `02_kernel_comparison.ipynb` | Comparación del kernel **NNGP analítico vs. empírico** para FC y CNN con peso compartido. Análisis espectral y NKA. |

## 📜 Scripts auxiliares

| Script | Descripción |
|--------|-------------|
| `compile_report.py` | Genera informe LaTeX unificado a partir de resultados experimentales |

## 📊 Figuras generadas

- `fig01_clt_convergence.png` — Convergencia CLT (Wasserstein, QQ-plot, histogramas)
- `fig02_joint_convergence.png` — Covarianza empírica → kernel NNGP
- `fig03_kernel_convergence.png` — Comparación de kernels FC vs CNN

## 🔬 Resultados clave

| Arquitectura | Convergencia GP | Correlación NNGP |
|-------------|----------------|-----------------|
| **FC (Fully Connected)** | ✅ Sí | ~0.958 |
| **CNN (peso compartido)** | ⚠️ Parcial | ~0.416 |

El hallazgo principal: **CNN con peso compartido rompe la convergencia GP ingenua**, validando la necesidad del marco de Tensor Programs para la generalización del teorema NNGP.

## 📚 Referencias

- Lee et al. (2018) — Deep Neural Networks as Gaussian Processes
- Jacot et al. (2018) — Neural Tangent Kernel
- Yang (2020) — Tensor Programs I: Wide Feedforward or Recurrent Neural Networks
