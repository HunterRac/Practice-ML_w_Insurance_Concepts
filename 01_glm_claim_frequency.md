# ML for P&C Insurance — Part 1: Poisson GLM for Claim Frequency

## Roadmap

| # | Algorithm | P&C use case |
|---|-----------|--------------|
| **1** | **Poisson GLM (this file)** | Claim frequency, rating relativities |
| 2 | Gamma GLM | Claim severity; frequency × severity = pure premium |
| 3 | Tweedie GLM | Pure premium modeled directly (zero-inflated, continuous) |
| 4 | Gradient boosting (LightGBM/XGBoost) | Frequency/severity with interactions; challenger to the GLM |
| 5 | Random forest | Variable discovery, claim triage, retention |
| 6 | Logistic regression vs. GBM classifier | Fraud flag, lapse/renewal, subrogation potential |
| 7 | K-means / clustering | Territory rating, customer segmentation |
| 8 | Isolation forest / anomaly detection | Claims fraud screening |
| 9 | Survival models (Cox, GBM-survival) | Policy retention, claim time-to-close |
| 10 | Chain ladder + ML reserving | Loss reserving, IBNR |
| 11 | Explainability (SHAP, PDPs) | Rate filing support, model governance |

---

## 1. Definition

A **Generalized Linear Model** links a linear combination of predictors to the mean of a response from the exponential family:

$$g(\mathbb{E}[Y]) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$$

For **claim frequency**:

- **Response** $Y$ = number of claims on a policy, assumed $Y \sim \text{Poisson}(\lambda \cdot e)$
- **Link** = log, so effects are **multiplicative**
- **Exposure** $e$ (policy-years in force) enters as an **offset**: $\log(e)$ added with a fixed coefficient of 1

$$\log \mathbb{E}[Y] = \log(e) + \beta_0 + \sum_j \beta_j x_j \quad\Longrightarrow\quad \mathbb{E}[Y] = e \cdot e^{\beta_0} \cdot \prod_j \left(e^{\beta_j}\right)^{x_j}$$

So each $e^{\beta_j}$ is a **rating relativity**, which is why GLMs map directly onto a rating plan / rate manual, and why they're still the regulatory workhorse for filed rates.

Key assumptions to check:
- **Variance = mean** (Poisson). If Pearson χ²/df ≫ 1, you have overdispersion → use quasi-Poisson or Negative Binomial.
- Log-linearity of continuous predictors (bin or spline them if not).
- Independence across policies.

**Why the offset and not a weight?** A policy in force for 6 months should be expected to have half the claims of an annual policy at the same risk. The offset encodes exactly that. (Equivalent alternative: model `claims / exposure` with `sample_weight = exposure`, which is what scikit-learn's `PoissonRegressor` expects.)

---

## 2. Worked example: personal auto

**Book:** 50,000 auto policies, varying exposure (0.1 to 1.0 years). Rating variables:

| Variable | Type | Levels |
|----------|------|--------|
| `age_band` | categorical | 18-25, 26-40, **41-60 (base)**, 61+ |
| `region` | categorical | Urban, **Suburban (base)**, Rural |
| `vehicle_age` | continuous | 0 to 19 years |
| `prior_claims` | count | 0 to 3 |

Base levels are chosen as the largest, most "typical" cell, so the intercept is the base rate.

**Data is simulated with a known truth** so you can see whether the GLM recovers it:

| Factor | True relativity | GLM estimate (95% CI) |
|--------|-----------------|-----------------------|
| Base rate (41-60, Suburban, new car, 0 priors) | 0.080 | 0.078 (0.068 to 0.089) |
| Age 18-25 | 1.90 | 2.07 (1.81 to 2.37) |
| Age 26-40 | 1.10 | 1.15 (1.01 to 1.32) |
| Age 61+ | 1.25 | 1.24 (1.10 to 1.39) |
| Urban | 1.35 | 1.30 (1.18 to 1.44) |
| Rural | 0.80 | 0.81 (0.71 to 0.93) |
| Per year of vehicle age | 0.980 | 0.980 |
| Per prior claim | 1.284 | 1.254 (1.17 to 1.35) |

Every true value is inside its confidence interval.

**Pricing a policy by hand.** A 22-year-old, Urban, 3-year-old car, 1 prior claim, annual policy:

$$\lambda = 0.078 \times 2.068 \times 1.302 \times 0.980^3 \times 1.254 \approx 0.247 \text{ claims/yr}$$

With an average severity of \$4,500 (which would come from the Part 2 Gamma severity GLM), pure premium ≈ 0.247 × 4,500 ≈ **\$1,114**. Before loading for expenses and profit, this is the rate manual: base rate × relativities.

**Diagnostics from the run:**
- Pearson χ²/df = **1.006**, so no overdispersion (as expected, since data is truly Poisson; real auto data is often 1.1 to 1.5).
- Test deviance explained = **1.8%**. That looks tiny but is normal for frequency: claims are rare and mostly random, so even a perfect model explains little *individual* variance. Judge frequency models with **lift / actual-vs-expected** tables and Gini, not R².
- Lift table: predicted frequency ranges from 0.054 (decile 1) to 0.175 (decile 10), a 3.2× spread, and the top decile's actual frequency (0.168) tracks expected well. Middle deciles are noisy at ~50 claims each.

---

## 3. Python code

Full script: [`01_glm_claim_frequency.py`](/mnt/project-files/ml-insurance/01_glm_claim_frequency.py) (needs `numpy pandas statsmodels scikit-learn`). Core pieces:

```python
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- simulate a book (see full script) ---
# df has: age_band, region, vehicle_age, prior_claims, exposure, claim_count

# --- fit Poisson GLM with log(exposure) offset ---
formula = ("claim_count ~ C(age_band, Treatment('41-60'))"
           " + C(region, Treatment('Suburban'))"
           " + vehicle_age + prior_claims")
glm = smf.glm(formula, data=train,
              family=sm.families.Poisson(),
              offset=np.log(train["exposure"])).fit()

# --- relativities ---
rel = pd.DataFrame({"relativity": np.exp(glm.params),
                    "ci_low":  np.exp(glm.conf_int()[0]),
                    "ci_high": np.exp(glm.conf_int()[1])})

# --- overdispersion check ---
print(glm.pearson_chi2 / glm.df_resid)   # ~1 is fine; >>1 -> NegBin / quasi-Poisson

# --- predict expected claim counts on new data ---
test["pred_count"] = glm.predict(test, offset=np.log(test["exposure"]))

# --- actual vs expected by decile (lift table) ---
test["pred_freq"] = test.pred_count / test.exposure
test["decile"] = pd.qcut(test.pred_freq, 10, labels=False) + 1
lift = test.groupby("decile").agg(actual=("claim_count", "sum"),
                                  expected=("pred_count", "sum"),
                                  exposure=("exposure", "sum"))
lift["A/E"] = lift.actual / lift.expected
```

**scikit-learn equivalent** (useful when you later swap in a GBM with the same API):

```python
from sklearn.linear_model import PoissonRegressor
# target = frequency, weight = exposure  (equivalent to count + log-offset)
model = PoissonRegressor(alpha=0).fit(X_train, y_train / exp_train,
                                      sample_weight=exp_train)
```

---

## 4. When to use it, and when not to

**Use a Poisson GLM when** you need a filed, explainable rating plan; regulators and pricing committees expect relativity tables; the data is moderate in size; or you need a benchmark for fancier models.

**Reach for something else when:**
- Overdispersion is large → **Negative Binomial GLM** (`sm.families.NegativeBinomial()`)
- Strong interactions/non-linearities you can't hand-engineer → **GBM with Poisson loss** (Part 4), often used to *find* interactions that you then add back into the GLM
- Too many zeros beyond Poisson → zero-inflated Poisson
- Many sparse levels (e.g. 3,000 territories) → GLM with ridge/lasso penalty, or credibility/mixed models

**Practitioner tips**
- Always choose the base level as the highest-exposure level, or CIs on the others get wide.
- Bin continuous variables like driver age; the true curve is U-shaped and a single linear term would miss it.
- Check A/E by each rating variable one-way, not just overall, before signing off.
