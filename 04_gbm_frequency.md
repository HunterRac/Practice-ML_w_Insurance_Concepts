# Part 4: Gradient Boosting (LightGBM) for Claim Frequency

Script: [`04_gbm_frequency.py`](/mnt/project-files/ml-insurance/04_gbm_frequency.py) (needs `lightgbm`)

## 1. Definition

**Gradient boosted trees** build an additive model of many shallow trees, each fit to the gradient of the loss with respect to the current prediction:

$$F_M(x) = F_0 + \eta \sum_{m=1}^{M} h_m(x), \qquad h_m \approx -\frac{\partial L(y, F)}{\partial F}\Big|_{F_{m-1}}$$

For pricing, the key choice is the **loss**. With `objective="poisson"`, LightGBM works on the log scale ($F = \log\lambda$), exactly like a Poisson GLM, but each $h_m$ is a tree rather than a linear term. So it:
- finds **non-linear** effects (credit curves, U-shaped age) without manual binning
- finds **interactions** automatically (a depth-$d$ tree can carry $d$-way interactions)
- handles missing values and doesn't need scaling

Other insurance objectives: `gamma` (severity), `tweedie` with `tweedie_variance_power` (pure premium), `binary` (fraud, lapse).

**Exposure.** Use target = claims / exposure with `weight = exposure`, or target = count with `init_score = log(exposure)`. Both are equivalent to the GLM offset.

**Hyperparameters that matter for insurance data:**
| Param | Why |
|---|---|
| `min_child_samples` (or `min_sum_hessian`) large, e.g. 200+ | Claims are rare; small leaves fit noise |
| `learning_rate` small + early stopping | Lets validation deviance pick the number of trees |
| `num_leaves` 7 to 31 | Shallow trees generalize better on noisy targets |
| `monotone_constraints` | Enforce e.g. "more prior claims never lowers the rate" (regulators like this) |

## 2. Worked example: can the GBM find what the GLM misses?

The book now has 200,000 policies, with two things deliberately hidden in the true frequency that a main-effects GLM cannot represent:
1. A **young driver × urban** interaction (an extra 1.3×)
2. A **non-linear credit curve**: flat above 700, rising steeply below

**Overall results (test set):**
| Model | Poisson deviance | Normalized Gini | Correlation with true freq |
|---|---|---|---|
| Poisson GLM | 0.60278 | 0.255 | 0.958 |
| LightGBM (385 trees, early stopped) | 0.60247 | 0.255 | 0.961 |
| True frequency (oracle) | 0.60119 | 0.263 | 1.000 |

That is a small overall gain, which is very typical. Frequency is mostly noise, so global metrics move a little. **The value shows up in segments:**

**Interaction recovered.** Actual frequency by segment:
| Young? | Urban? | Actual | GLM | GBM |
|---|---|---|---|---|
| Yes | No | 0.143 | 0.172 (overpriced 21%) | 0.153 |
| Yes | Yes | 0.292 | 0.262 (underpriced 10%) | 0.283 |

The GLM spreads the interaction over the main effects, so it overcharges young rural drivers and undercharges young urban drivers. That is exactly the adverse selection a competitor with a better model will exploit.

**Credit curve recovered** (partial dependence, relative to a 700 score):
| Score | Truth | GBM | GLM (linear) |
|---|---|---|---|
| 550 | 2.46 | 2.46 | 1.70 |
| 650 | 1.35 | 1.34 | 1.19 |
| 800 | 1.00 | 0.91 | **0.70** |
| 850 | 1.00 | 0.98 | **0.59** |

The linear GLM term gives large discounts to high scores that don't exist in the data. The GBM tracks the true hockey stick.

**A warning in the feature importance:** `vehicle_value` got 8% of the gain, even though it has **no** effect on true frequency. The GBM fit noise there. Importance ≠ causation, and it is one reason to validate with lift and partial dependence, not importance alone.

**The common practical workflow:** use the GBM to *discover* the interaction and the credit curve, then add `young:urban` and a credit spline or banding to the GLM. You keep the filing-friendly GLM and gain most of the lift.

## 3. Python code (core)

```python
import lightgbm as lgb

FEATURES = ["driver_age", "vehicle_age", "vehicle_value", "region", "prior_claims", "credit_score"]
df["region"] = df["region"].astype("category")       # native categorical support

params = dict(objective="poisson", learning_rate=0.03, num_leaves=15,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
# Optional: monotone_constraints=[0, 0, 0, 0, 1, -1]  (prior_claims up, credit down)

dtrain = lgb.Dataset(train[FEATURES], train.claim_count / train.exposure, weight=train.exposure)
dvalid = lgb.Dataset(valid[FEATURES], valid.claim_count / valid.exposure, weight=valid.exposure)
gbm = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(100)])

test["pred_freq"] = gbm.predict(test[FEATURES], num_iteration=gbm.best_iteration)

# Partial dependence by hand: vary one feature, average predictions
pdp = [gbm.predict(sample.assign(credit_score=c)).mean() for c in range(500, 851, 50)]
```

## 4. When to use it, and when not to

**Use when:** you're looking for lift beyond the GLM; finding interactions to add back into the GLM; internal models that aren't filed (underwriting tiers, marketing, claims triage, retention); large datasets.

**Be careful when:**
- **Rate filings.** Many US states require explainable factors. A GBM may need to be translated into GLM relativities or rating tables, or justified with SHAP (Part 11).
- **Small data.** Under ~10k policies, a regularized GLM often wins.
- **Extrapolation.** Trees are flat beyond the training range (e.g., a new high-value vehicle segment).
- **Stability.** Retraining can shift individual premiums a lot. Use monotone constraints and smooth, capped rate changes.
