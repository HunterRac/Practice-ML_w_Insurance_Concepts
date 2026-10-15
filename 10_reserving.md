# Part 10: Loss Reserving: Chain Ladder, GLMs, Bootstrap, and ML

Script: [`10_reserving.py`](/mnt/project-files/ml-insurance/10_reserving.py)

## 1. Definitions

**Reserving** estimates what will still be paid on claims that have already happened (reported but not settled, plus IBNR, incurred but not reported). Data is arranged in a **development triangle**: rows are accident years (AY), columns are development age, and the lower-right is the unknown future.

**Chain ladder (CL).** Estimate age-to-age factors from the history and roll each AY's latest diagonal forward:
$$f_j = \frac{\sum_i C_{i,j+1}}{\sum_i C_{i,j}}, \qquad \hat C_{i,\text{ult}} = C_{i,\text{latest}}\prod_{k\ge \text{latest}} f_k$$

**Over-dispersed Poisson (ODP) GLM.** Model incremental paid as quasi-Poisson with AY and development factors:
$$\log \mathbb{E}[X_{ij}] = \alpha_i + \beta_j, \qquad \text{Var}(X_{ij}) = \phi\,\mu_{ij}$$
It gives **exactly the chain ladder reserve**, but now you're in a regression framework: you can get standard errors, bootstrap, add calendar-year terms, use exposure offsets, and drop parameters.

**Bootstrap ODP** (England & Verrall). Resample Pearson residuals to make pseudo-triangles, re-run CL on each, and add process noise. The result is a **reserve distribution**, needed for reserve ranges, risk margins and capital (e.g., the 99.5th percentile for Solvency II).

**ML reserving at the claim level.** Instead of aggregating, predict each open claim's ultimate from its features (injury, attorney, payments and case reserve so far) with a GBM. This brings in information the triangle throws away.

## 2. Worked example

### a) Triangle with a hidden calendar-year inflation spike
10 × 10 paid triangle (\$000s), exposure growing 3%/yr, ODP noise (φ = 40). **Claims inflation runs 3%/yr, then jumps to 10%/yr in the last three calendar years** (think social inflation or used-car parts costs).

| Method | Total reserve | vs truth \$59.2M |
|---|---|---|
| Chain ladder | \$51.9M | **−12%** |
| ODP GLM (AY + dev) | \$51.9M | identical to CL ✓ |
| GLM: exposure offset + dev + piecewise CY trend (fitted) | \$67.4M | +14% |
| Same GLM, future trend **selected** at 10% | **\$61.4M** | +4% |

- **CL under-reserves by 12%.** Its factors average history from the 3%-inflation era and assume the past calendar-year pattern repeats. A calendar-year shift breaks that.
- The **ODP GLM reproduces CL to the dollar.** That equivalence is the bridge from actuarial methods to statistical modeling.
- The **calendar-trend GLM detects the shift** (pre-trend fitted +2.5% vs a true 3%; post-trend +13.3% vs a true 10%). But the post-trend rests on only 3 diagonals, so projecting it forward over-reserves. Selecting a 10% future trend with judgment gets within 4%. This is the real workflow: **models surface the signal; the actuary selects assumptions.**
- The error is biggest on recent AYs (AY9: CL \$18.5M vs true \$21.2M), which have the most future payments exposed to the new inflation.

### b) Bootstrap ODP distribution of the CL reserve
| Mean | SD | CV | 75th | 95th | 99.5th |
|---|---|---|---|---|---|
| \$51.9M | \$3.4M | 6.6% | \$54.2M | \$57.9M | \$60.4M |

The true reserve (\$59.2M) falls at the **98th percentile** of the CL distribution. The bootstrap measures *noise around the model*, not *model error*. If the model's assumptions are broken (here, inflation), the distribution is confidently wrong. Always run diagnostics (calendar-year residual plots) before trusting the range.

### c) Individual-claim GBM
15,000 BI claims at 12 months, with the ultimate depending on injury type, attorney involvement and claimant age. The hold-out is 4,500 "open" claims.
| Method | Total vs actual \$26.54M | Claim-level MAE |
|---|---|---|
| One aggregate incurred LDF (0.959) | \$26.09M (−1.7%) | \$1,869 |
| **LightGBM, gamma loss, target = ultimate / paid** | \$26.07M (−1.8%) | **\$1,230 (−34%)** |

In total, both are similar. Good case reserves already carry most of the aggregate signal, and segment A/Es were within ±10% for both. **The GBM's edge is claim-level accuracy**, which matters for large-loss early warning, case-reserve adequacy reviews, and triaging which open claims will develop adversely. It complements the triangle; it doesn't replace it.

## 3. Python code (core)

```python
# Chain ladder on a cumulative triangle (NaN in the future cells)
def chain_ladder(cum):
    n = cum.shape[0]
    ldf = np.array([np.nansum(cum[:n-j-1, j+1]) / np.nansum(cum[:n-j-1, j]) for j in range(n-1)])
    latest = np.array([cum[i, n-1-i] for i in range(n)])
    cdf = np.append(np.cumprod(ldf[::-1])[::-1], 1.0)
    ult = latest * cdf[::-1]
    return ldf, latest, ult, ult - latest

# ODP GLM on incrementals in long format (one row per cell): identical reserve to CL
odp = smf.glm("inc ~ C(ay) + C(dev)", data=observed_cells,
              family=sm.families.Poisson()).fit(scale="X2")      # quasi-Poisson
reserve = odp.predict(future_cells).sum()

# Calendar-year trend with a break, exposure offset instead of AY factors
cyglm = smf.glm("inc ~ C(dev) + cy + np.maximum(cy - 6, 0)", data=observed_cells,
                family=sm.families.Poisson(), offset=np.log(observed_cells.exposure)).fit(scale="X2")

# Individual-claim development ratio
gbm = lgb.LGBMRegressor(objective="gamma", n_estimators=800, learning_rate=0.03,
                        num_leaves=15, min_child_samples=50)
gbm.fit(closed[feats], closed.ultimate / closed.paid_12m)
ultimate_open = gbm.predict(open_[feats]) * open_.paid_12m
```

For production, the **`chainladder`** Python package (`pip install chainladder`) implements CL, Mack, bootstrap ODP, Bornhuetter-Ferguson, Cape Cod and more with a scikit-learn-like API.

## 4. When to use which

| Method | Use when | Watch out for |
|---|---|---|
| Chain ladder | Stable, mature lines; the starting point everywhere | Calendar-year shifts, changes in claims handling speed or case reserving, thin recent AYs |
| Bornhuetter-Ferguson / Cape Cod | Immature AYs where the latest diagonal is unreliable | Needs an a priori loss ratio |
| ODP GLM | Need distributions, diagnostics, trends, or fewer parameters | Same structural assumptions as CL unless you add terms |
| Bootstrap ODP / Mack | Reserve ranges, risk margins, capital | Measures noise, not model misspecification |
| Claim-level ML | Rich claim data; early large-loss detection; case-reserve review | Leakage (closed-claim bias: claims that closed early aren't like open ones); explainability for auditors; reconcile to triangle totals |
