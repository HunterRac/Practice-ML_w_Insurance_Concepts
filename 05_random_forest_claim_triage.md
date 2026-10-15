# Part 5: Random Forest for Claim Triage (Litigation Risk at FNOL)

Script: [`05_random_forest_claim_triage.py`](/mnt/project-files/ml-insurance/05_random_forest_claim_triage.py)

## 1. Definition

A **random forest** averages many deep decision trees, each made different by two sources of randomness:
1. **Bagging:** each tree trains on a bootstrap sample of rows.
2. **Feature subsampling:** each split only considers a random subset of features (`max_features`, often √p).

$$\hat p(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

A single deep tree has low bias but huge variance. Averaging B decorrelated trees keeps the low bias and cuts the variance. That gives:
- **Out-of-bag (OOB) validation for free:** each row is left out of ~37% of trees, so you can score it on those.
- **Few hyperparameters** and hard to badly overfit by adding trees.
- Non-linearities and interactions, like GBMs.

**RF vs GBM:** GBMs usually win on accuracy when tuned. RFs are more robust with little tuning, parallelize trivially, and are a strong fast baseline. Note that RFs can't use a Poisson/Tweedie loss in scikit-learn the way GBMs can (squared error or Poisson criterion only for regression), so for *pricing* GBMs are preferred. For **classification-style claims problems**, RFs are excellent.

## 2. Worked example: which BI claims will get an attorney?

**Business problem.** Attorney-represented bodily-injury claims cost several times more. If you can flag them at **first notice of loss (FNOL)**, you can route them to senior adjusters, make early contact and settle sooner.

**Data:** 30,000 synthetic auto BI claims (17.1% get an attorney). FNOL features: injury type, report lag, tort vs no-fault state, liability clarity, number of claimants, claimant age, hours to first adjuster contact, plus two **pure-noise** columns (a zip-code-like integer and a coin flip). The truth includes thresholds (report lag > 14 days, contact > 48 hours) and a three-way interaction (soft tissue × tort state × disputed liability).

**Single tree vs forest:**
| Model | Train AUC | Test AUC | Test PR-AUC |
|---|---|---|---|
| Single tree, fully grown | 1.000 | 0.574 | 0.206 |
| Single tree, depth 5 | 0.786 | 0.774 | 0.383 |
| **Random forest (500 trees)** | 0.857 | **0.784** | **0.410** |

The fully grown tree memorizes the training data and is barely better than a coin flip on new claims. The forest averages away that variance.

**Variable discovery: two importance measures disagree, and that's the lesson.**
| Feature | Impurity (MDI) | Permutation (test AUC drop) |
|---|---|---|
| injury_type = none_reported | 0.214 | 0.116 |
| tort_state | 0.121 | 0.048 |
| liability_clear | 0.138 | 0.035 |
| **noise_zip_code** | **0.088** (ranked 3rd!) | **−0.001** (useless) |
| noise_coin_flip | 0.013 | −0.001 |

Impurity importance is **biased toward high-cardinality features**: a column with 90,000 unique values offers endless split points to fit noise. Permutation importance on held-out data shows correctly that it's worthless. **Always use permutation importance (or SHAP, Part 11) on test data for variable selection.**

**Triage result:** routing the top 20% of scores to senior adjusters:
- Attorney rate in the flagged group: **42.7%** (vs 17.1% overall, 10.7% in the rest)
- Catches **50%** of all eventual attorney claims while reviewing 20% of claims

**Threshold found:** partial dependence on report lag jumps between 13 days (0.182) and 15 days (0.207), recovering the true 14-day cutoff without being told.

## 3. Python code (core)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                            min_samples_leaf=20,      # smooths probabilities
                            oob_score=True, n_jobs=-1, random_state=0)
rf.fit(X_train, y_train)
print(rf.oob_score_)                                  # free validation estimate

# Variable discovery: use permutation importance on TEST data, not rf.feature_importances_
pi = permutation_importance(rf, X_test, y_test, scoring="roc_auc", n_repeats=5)

# Triage: flag top 20%
p = rf.predict_proba(X_test)[:, 1]
flag = p >= np.quantile(p, 0.8)
```

## 4. When to use it, and when not to

**Insurance uses:** claim triage (litigation, large-loss, subrogation potential, total-loss prediction), fast-track vs complex routing, a quick variable-screening pass before building a GLM, and retention models.

**Be careful when:**
- **Probabilities matter** (pricing, reserving): RF probabilities are often poorly calibrated and squeezed toward the middle. Calibrate with `CalibratedClassifierCV` (see Part 6).
- **Leakage at FNOL.** Only use features known *at the moment of scoring*. "Paid to date" or "attorney letter received" would make the model look great and be useless.
- **Very large data** or the need for squeezing out max lift: GBMs are typically better and faster.
- **Extrapolation:** like all tree models, predictions are flat outside the training range.
