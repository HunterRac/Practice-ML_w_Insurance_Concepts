# Part 2: Gamma GLM for Claim Severity (and Pure Premium)

Script: [`02_gamma_glm_severity.py`](/mnt/project-files/ml-insurance/02_gamma_glm_severity.py) · shared data: [`insurance_data.py`](/mnt/project-files/ml-insurance/insurance_data.py)

## 1. Definition

**Severity** = the dollar amount of a claim, *given* a claim happened. It is positive, right-skewed, and its spread grows with its mean (a \$40k claim varies by thousands; a \$2k claim by hundreds).

A **Gamma GLM with log link** matches exactly that:

$$Y_i \sim \text{Gamma}, \qquad \log \mathbb{E}[Y_i] = \beta_0 + \sum_j \beta_j x_{ij}, \qquad \text{Var}(Y_i) = \phi\,\mu_i^2$$

- Variance ∝ mean², so the **coefficient of variation is constant** ($\sqrt{\phi}$). A 10% relativity means the same thing for cheap and expensive claims.
- Log link again gives **multiplicative relativities**, so severity factors multiply directly with the frequency factors from Part 1:

$$\text{Pure premium} = \underbrace{\mathbb{E}[\text{claims per year}]}_{\text{Poisson GLM}} \times \underbrace{\mathbb{E}[\text{cost per claim}]}_{\text{Gamma GLM}}$$

- Fit on **claim-level** data (only policies with claims). If you have policy-level aggregates, use average severity as the response with `var_weights = claim_count`.

**Why not OLS on log(amount)?** It models the *median*, not the mean. `exp(prediction)` is biased low by roughly $e^{\sigma^2/2}$ and needs a smearing correction. The Gamma GLM predicts the mean directly.

## 2. Worked example: auto physical damage severity

The shared book has 100,000 policies and 6,414 claims (mean \$3,818, median \$3,034, max \$29,289). The true severity drivers are region, vehicle age (newer cars cost more to repair), and vehicle value (elasticity 0.5, so a car worth 4× costs 2× to fix).

| Factor | Truth | Gamma GLM |
|---|---|---|
| Base severity (Suburban, new car, \$25k value) | \$4,500 | \$4,514 |
| Urban | 1.15 | 1.149 |
| Rural | 0.90 | 0.904 |
| Per year of vehicle age | 0.970 | 0.970 |
| log(value / 25k) elasticity | 0.50 | 0.511 |
| Dispersion φ (= 1/shape) | 0.50 | 0.482 |

**Retransformation bias, demonstrated.** Total test-set claim dollars:

| Method | Predicted | vs actual \$6.07M |
|---|---|---|
| Gamma GLM | \$6.22M | +2.4% |
| log-OLS, naive `exp()` | \$4.81M | **−21%** |
| log-OLS + Duan smearing | \$6.23M | +2.6% |

A 21% understatement of severity would underprice the whole book by 21%.

**Combining with frequency.** A Poisson GLM on the same book × this Gamma GLM gives pure premium. On the test set, expected loss was \$6.15M vs actual \$6.07M (**A/E = 0.987**). Loss cost per policy-year ranges from about \$112 in the lowest decile to \$1,051 in the highest, a 9× spread.

**One policy:** 22-year-old, Urban, 3-year-old \$35k car, 1 prior claim, credit 650:
frequency 0.418 × severity \$5,629 = **pure premium \$2,356/yr**.

## 3. Python code (core)

```python
import numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf

# claim-level data: one row per claim
sev = smf.glm("claim_amount ~ C(region, Treatment('Suburban'))"
              " + vehicle_age + np.log(vehicle_value / 25000)",
              data=claims_train,
              family=sm.families.Gamma(link=sm.families.links.Log())).fit()
np.exp(sev.params)        # severity relativities
sev.scale                 # dispersion phi; CV = sqrt(phi)

# pure premium on policies = freq (per policy-year) x severity
policies["pure_premium"] = (freq_glm.predict(policies, offset=np.zeros(len(policies)))
                            * sev.predict(policies))
```

## 4. When to use it, and when not to

**Use when:** building a classic frequency/severity rating plan; you want to know *whether* a variable drives how often or how much (e.g., vehicle value drives severity but barely frequency); filing support.

**Watch out for:**
- **Large losses.** A few \$500k BI claims dominate a Gamma fit. Cap losses (e.g., at \$100k), model the capped severity, and load the excess separately.
- **Heavier tails than Gamma.** Check deviance residuals; consider Inverse Gaussian (Var ∝ μ³) or lognormal with smearing.
- **Thin data.** Severity has far fewer rows than frequency (6k vs 100k here), so use fewer, coarser variables.
- **Splitting.** Split train/test by *policy*, not by claim, to avoid leakage.
