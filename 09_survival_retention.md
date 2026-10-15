# Part 9: Survival Models for Policy Retention

Script: [`09_survival_retention.py`](/mnt/project-files/ml-insurance/09_survival_retention.py) (statsmodels only)

## 1. Definition

**Survival analysis** models *time until an event* (lapse, cancellation, claim closure) when many subjects **haven't had the event yet**. Those are *right-censored*: we know they lasted at least this long.

- **Survival function** $S(t) = P(T > t)$, e.g., the retention curve.
- **Hazard** $h(t)$: the instantaneous event rate at $t$ given survival to $t$.

**Kaplan-Meier (KM)**, non-parametric:
$$\hat S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)$$
Here $d_i$ = lapses at time $t_i$ and $n_i$ = customers still at risk. Censored customers count in $n_i$ until they leave observation.

**Cox proportional hazards**, semi-parametric regression:
$$h(t\mid x) = h_0(t)\, e^{\beta^\top x}$$
The baseline hazard $h_0(t)$ is left unspecified. $e^{\beta_j}$ is a **hazard ratio (HR)**: HR = 0.55 means 45% lower lapse intensity *at every point in time*. The key assumption is that HRs are constant over time (proportional hazards).

**Why not just logistic regression on "lapsed within 12 months"?** Customers who joined 5 months ago haven't had 12 months to lapse. Count them as retained and you underestimate lapse; drop them and you overestimate it. Survival models use every customer's partial information correctly.

## 2. Worked example: personal lines retention

**Data:** 20,000 customers who joined at staggered times over the last 48 months. **81% are still active (censored).** True lapse process: Weibull with decreasing hazard (new customers churn most), and drivers of rate change at last renewal, bundling (home + auto), autopay, a claim last year, and age.

**Kaplan-Meier matches the truth almost exactly** despite 81% censoring:
| Retention | 12 m | 24 m | 36 m |
|---|---|---|---|
| Bundled (KM / truth) | 0.913 / 0.913 | 0.854 / 0.853 | 0.802 / 0.805 |
| Monoline (KM / truth) | 0.847 / 0.848 | 0.751 / 0.754 | 0.680 / 0.681 |

**The censoring trap, quantified** (12-month lapse rate):
| Method | Estimate | Truth 0.122 |
|---|---|---|
| Naive: censored counted as retained | 0.109 | −11% |
| Naive: drop actives with < 12 m tenure | 0.142 | +16% |
| **Kaplan-Meier** | **0.123** | ✓ |

**Cox hazard ratios:**
| Driver | HR | Truth | Reading |
|---|---|---|---|
| +10 pts rate change | 1.46 | 1.50 | A 10% increase raises lapse intensity ~46% |
| Bundled | 0.557 | 0.55 | Bundling nearly halves lapse |
| Autopay | 0.752 | 0.75 | |
| Claim last year | 1.41 | 1.30 | (slightly high; CI 1.29–1.54) |
| Age (per 10 yrs) | 0.89 | 0.90 | |

C-index = 0.63 (pairs ranked correctly; 0.5 = random). That is typical for retention, where much of churn is idiosyncratic.

**Customer-level predictions** (Breslow baseline × HR):
| Customer | 12 m | 24 m | 36 m | Expected months of next 48 |
|---|---|---|---|---|
| Monoline, +15% rate, no autopay, claim, age 25 | 0.63 | 0.44 | 0.32 | **23.5** |
| Bundled, 0% rate, autopay, no claim, age 55 | 0.95 | 0.91 | 0.87 | **43.7** |

Expected tenure (restricted mean survival time = area under the curve) is the input to **customer lifetime value**, and to **price optimization**: the rate change HR is a direct estimate of price elasticity of retention.

**PH assumption check:** bundled HR = 0.551 in months 0–12 vs 0.564 after month 12 (fitted with left truncation at 12). It is stable, so the assumption holds. In real data, check with Schoenfeld residuals (`lifelines`: `cph.check_assumptions()`).

## 3. Python code (core)

```python
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

# Kaplan-Meier
km = sm.SurvfuncRight(df.duration, df.lapsed)        # duration = months observed or to lapse
km.surv_times, km.surv_prob                           # the retention curve

# Cox PH
cols = ["rate_change_pct", "bundled", "autopay", "claim_last_year", "age"]
cox = PHReg(df.duration, df[cols], status=df.lapsed, ties="efron").fit()
np.exp(cox.params)                                    # hazard ratios

# Individual survival curve: S(t|x) = exp(-H0(t) * exp(x'b))
t, H0 = cox.baseline_cumulative_hazard[0][:2]
S = np.exp(-H0 * np.exp(x_new @ cox.params))
```

With **lifelines** (more common; install trouble here, so not used):
```python
from lifelines import KaplanMeierFitter, CoxPHFitter
CoxPHFitter().fit(df[cols + ["duration", "lapsed"]], "duration", "lapsed").print_summary()
```

## 4. When to use it, and when not to

**Insurance uses:** retention and lapse (personal lines, life persistency), time to claim closure (claims ops staffing, reserving signals), time to first claim, time to litigation, and agent or producer attrition.

**Variations:**
- **Discrete renewals.** Personal auto lapses mostly at 6/12-month renewal points. A *discrete-time* model (logistic regression on person-period rows: one row per customer per renewal) is often more natural and handles time-varying covariates like each renewal's rate change.
- **Competing risks.** Lapse to a competitor vs cancellation for non-payment vs moving out of state. Use cause-specific hazards or Fine-Gray.
- **Non-linear / interactions:** random survival forests or gradient-boosted Cox (`scikit-survival`, XGBoost `survival:cox`).
- **Non-proportional hazards:** stratify (`strata=`) or add time interactions.

**Pitfalls:** don't use future information (a covariate measured after the start of follow-up); handle left truncation when customers enter your data after joining (`entry=`).
