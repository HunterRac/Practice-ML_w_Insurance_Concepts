# Part 11: Explainability with SHAP (Rate Filings and Model Governance)

Script: [`11_explainability_shap.py`](/mnt/project-files/ml-insurance/11_explainability_shap.py) (needs `shap`, `lightgbm`)

## 1. Definition

**SHAP (SHapley Additive exPlanations)** splits each prediction into per-feature contributions using Shapley values from cooperative game theory:

$$f(x) = \phi_0 + \sum_{j=1}^{p}\phi_j(x)$$

$\phi_0$ is the average model output and $\phi_j(x)$ is feature $j$'s fair share of moving *this* prediction away from the average: its marginal contribution averaged over all orders of adding features. **TreeSHAP** computes this exactly and quickly for tree ensembles.

**The insurance-specific trick.** A Poisson, Gamma or Tweedie GBM works on the **log scale**, so SHAP values add on the log scale and **multiply on the premium scale**:

$$\hat\lambda(x) = e^{\phi_0}\times\prod_j e^{\phi_j(x)}$$

Each $e^{\phi_j(x)}$ is a **per-policy rating factor**. A GBM can then be presented like a rate manual: base rate × factor for age × factor for credit × …

**SHAP interaction values** $\phi_{jk}$ further split each $\phi_j$ into a main effect and pairwise interactions. This is how you find out *which* interactions a GBM is using.

Related tools: **PDP** (average prediction as one feature varies), **ICE** (the same per policy), **ALE** (like PDP but robust to correlated features), and **permutation importance** (Part 5).

## 2. Worked example: explaining the Part 4 frequency GBM

The same 200,000-policy auto book (truth: non-linear credit curve, a young × urban interaction, and *no* frequency effect from vehicle value).

**Additivity check:** base + Σ SHAP = log(prediction) to within 1e-14. Exact, not approximate.

**Global importance** (mean |SHAP|, and the typical factor swing):
| Feature | Mean \|SHAP\| | ≈ factor |
|---|---|---|
| region | 0.184 | ×1.20 |
| credit_score | 0.171 | ×1.19 |
| driver_age | 0.145 | ×1.16 |
| prior_claims | 0.114 | ×1.12 |
| vehicle_age | 0.105 | ×1.11 |
| vehicle_value | 0.020 | ×1.02 |

Compare Part 4, where *gain* importance gave vehicle_value 8%. SHAP correctly shows it doing almost nothing.

**SHAP relativities vs truth.** Average $e^{\phi_j}$ by band gives GLM-style relativity tables straight from the GBM:
| Credit band | SHAP relativity | Truth |
|---|---|---|
| 450–550 | 2.60 | 2.86 |
| 550–600 | 1.95 | 2.12 |
| 600–650 | 1.53 | 1.57 |
| 650–700 | 1.11 | 1.16 |
| **700–750 (base)** | **1.00** | **1.00** |
| 750–800 | 1.00 | 1.00 |
| 800–850 | 0.94 | 1.00 |

| Region | SHAP | Truth |
|---|---|---|
| Rural | 0.794 | 0.80 |
| Urban | 1.359 | 1.35 |

Vehicle value comes out flat (0.997 to 1.041), as it should.

**Interaction detected.** The strongest pairwise SHAP interaction is **driver_age × region** (0.031, nearly double the next one). The interaction factor is 1.151 for young & urban vs 0.888 for young & non-urban. Their ratio, 1.151 / 0.888 = **1.30**, **exactly matches the true 1.3× extra loading.** This is the evidence to bring to a pricing committee for adding `young × urban` to the filed GLM.

**Local explanation (one policy)**, as a factor build-up an underwriter or regulator can read:
```
Policy: 22-yr-old, Urban, 2-yr-old $32k car, 1 prior claim, credit 610
  Base frequency                      0.1097
  x driver_age     = 22      x1.702   -> 0.1868
  x credit_score   = 610     x1.451   -> 0.2710
  x region         = Urban   x1.283   -> 0.3476
  x prior_claims   = 1       x1.183   -> 0.4113
  x vehicle_age    = 2       x1.154   -> 0.4745
  x vehicle_value  = 32000   x0.965   -> 0.4580
  Model prediction:                      0.4580   (matches exactly)
```
The same build-up gives **adverse-action reason codes** ("your premium is higher mainly because of driver age and credit-based insurance score"), required in many states when credit affects the rate.

## 3. Python code (core)

```python
import shap, numpy as np, pandas as pd

explainer = shap.TreeExplainer(gbm)                 # works for LightGBM, XGBoost, sklearn trees
sv = explainer.shap_values(X)                       # log-scale contributions for Poisson/Gamma/Tweedie
base = explainer.expected_value

# Global importance
pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values()

# SHAP relativity table for a feature
S = pd.DataFrame(sv, columns=X.columns, index=X.index)
rel = np.exp(S["credit_score"].groupby(pd.cut(X.credit_score, bands)).mean())
rel / rel.loc[base_band]

# Interactions (slower: run on a sample; categoricals as numeric codes)
siv = explainer.shap_interaction_values(X_sample)

# Plots (in a notebook)
shap.summary_plot(sv, X)                            # beeswarm
shap.dependence_plot("credit_score", sv, X, interaction_index="driver_age")
shap.plots.waterfall(explainer(X.iloc[[0]])[0])     # one-policy build-up
```

## 4. When to use it, and caveats

**Use for:** rate filings and regulator Q&A on GBMs; finding interactions to add back to the GLM; adverse-action and reason codes; model validation (does each effect make actuarial sense and have the right direction?); monitoring drift in *why* the model predicts, not just *what*.

**Caveats:**
- **SHAP explains the model, not the world.** It isn't causal. If the model leans on a proxy (e.g., a zip code standing in for a protected class), SHAP will faithfully report it. That makes it useful for fairness testing, but it is not a fix.
- **Correlated features** split credit in ways that can look odd (TreeSHAP's default "path-dependent" mode follows the tree's conditional splits). Group correlated features, or compare with the "interventional" mode using a background dataset.
- **Log-scale vs premium scale.** Add on log, multiply on premium. Don't sum `exp(shap)`.
- **Averaged SHAP relativities ≠ a GLM.** They're a summary of the GBM. If you need a filed GLM, refit the GLM using the GBM-discovered structure, rather than filing averaged SHAP numbers as-is.
