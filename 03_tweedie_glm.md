# Part 3: Tweedie GLM for Pure Premium

Script: [`03_tweedie_glm.py`](/mnt/project-files/ml-insurance/03_tweedie_glm.py)

## 1. Definition

The **Tweedie** family is the exponential-dispersion family with variance $\text{Var}(Y) = \phi\,\mu^p$. The power $p$ selects the distribution:

| p | Distribution | Insurance use |
|---|---|---|
| 1 | Poisson | Frequency |
| **1 < p < 2** | **Compound Poisson-Gamma** | **Pure premium / loss cost** |
| 2 | Gamma | Severity |
| 3 | Inverse Gaussian | Heavy-tailed severity |

For $1<p<2$, $Y$ is literally a Poisson number of Gamma-sized claims summed together. That gives an **exact point mass at zero** (no claims) plus a continuous positive part, which is the shape of policy-level loss data (93.9% zeros here).

Model setup:
- Response = **loss cost** = total loss / exposure
- `var_weights` = exposure
- Log link → one set of multiplicative relativities for the pure premium

$$\log \mathbb{E}[\text{loss cost}] = \beta_0 + \sum_j \beta_j x_j$$

**Tweedie vs. frequency × severity**

| | Freq × Sev | Tweedie |
|---|---|---|
| Models | 2 | 1 |
| Insight | Separates "how often" from "how much" | Blended |
| Assumes | Nothing linking the two | Same covariates move freq and sev in a tied way |
| Data needs | Claim-level detail | Just policy-level loss totals |
| Typical use | Personal lines rating plans | GBMs, commercial lines, quick models, when claim counts are unreliable |

## 2. Worked example: auto loss cost

The same 100,000-policy book. Mean loss cost is \$444.83 per policy-year.

**Choosing p.** Test results for p = 1.2, 1.5 and 1.8 were nearly identical (A/E 0.984 to 0.987, normalized Gini 0.328 for all three). The power mainly changes how much weight large-loss policies get, not the ranking. In practice, pick p by profile likelihood (the `tweedie` R package, or a grid over p with a proper Tweedie log-likelihood), or by out-of-sample deviance at a fixed p. Values around 1.5 to 1.7 are common for auto.

**Relativities (p = 1.5) vs. freq × sev vs. truth:**

| Factor | Truth (freq × sev) | Freq GLM × Sev GLM | Tweedie |
|---|---|---|---|
| Urban | 1.35 × 1.15 = 1.553 | 1.624 | 1.564 |
| Per year of vehicle age | 0.980 × 0.970 = 0.951 | 0.952 | 0.951 |

The Tweedie also estimated age 18–25 at 2.39. The pure frequency truth is 1.9, but it is inflated by the Urban × young-driver interaction the GLM doesn't model (Part 4 shows a GBM catching it).

**Head-to-head on the test set:**

| Model | A/E | Normalized Gini | Correlation with true premium |
|---|---|---|---|
| Tweedie p = 1.5 | 0.986 | 0.328 | 0.970 |
| Freq × Sev | 0.987 | 0.329 | 0.976 |
| True premium (oracle) | 0.973 | 0.341 | 1.000 |

They are essentially tied. The oracle's Gini of 0.341 is the ceiling given randomness in claims, and both GLMs get 96% of the way there.

**Normalized Gini** is the standard pricing-lift metric: sort policies by predicted loss cost, plot cumulative exposure vs cumulative actual loss (the Lorenz curve), and measure the area away from the diagonal, scaled by the best possible ordering.

## 3. Python code (core)

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf

policies["loss_cost"] = policies.total_loss / policies.exposure
tw = smf.glm("loss_cost ~ C(age_band, Treatment('41-60')) + C(region, Treatment('Suburban'))"
             " + vehicle_age + prior_claims + credit_score + np.log(vehicle_value / 25000)",
             data=train,
             family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
             var_weights=train.exposure).fit()
np.exp(tw.params)                         # pure-premium relativities
test["pure_premium"] = tw.predict(test)

# scikit-learn equivalent (same API as the GBMs in Part 4)
from sklearn.linear_model import TweedieRegressor
TweedieRegressor(power=1.5, link="log", alpha=0).fit(X, y_loss_cost, sample_weight=exposure)
```

## 4. When to use it, and when not to

**Use when:** you only have policy-level loss totals; you want one model; as the objective for a GBM (`objective="tweedie"` in LightGBM/XGBoost is one of the most-used pricing setups); commercial lines with sparse claims.

**Avoid / be careful when:**
- Variables drive frequency and severity in **opposite directions** (e.g., deductible: higher deductible → fewer claims but larger ones). Tweedie's tied structure blurs this; use freq × sev.
- You need to explain *why* a segment is expensive (frequency or severity?) to underwriters.
- Large losses: cap them, the same as for severity.
- The dispersion φ is assumed constant. If it varies by segment, look at "double GLMs" that model φ too.
