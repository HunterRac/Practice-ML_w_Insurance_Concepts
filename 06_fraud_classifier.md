# Part 6: Logistic Regression vs Gradient-Boosted Classifier for Claims Fraud

Script: [`06_fraud_classifier.py`](/mnt/project-files/ml-insurance/06_fraud_classifier.py)

## 1. Definition

**Logistic regression** is the binomial GLM with a logit link:

$$\log\frac{p}{1-p} = \beta_0 + \sum_j \beta_j x_j, \qquad p = \frac{1}{1+e^{-\eta}}$$

$e^{\beta_j}$ is an **odds ratio**. It is well calibrated by construction (on training data, mean prediction = observed rate), easy to explain to an SIU manager or a regulator, and only captures interactions or non-linearity you engineer.

A **GBM classifier** (LightGBM `objective="binary"`) is the same boosting machinery as Part 4, with log-loss. It finds interactions and thresholds automatically, and usually ranks better on rich data.

**Imbalanced labels (fraud is ~1–5% of claims)** change how you evaluate and decide:
- **PR-AUC** (average precision) is more informative than ROC-AUC when positives are rare. The baseline PR-AUC is the fraud rate itself.
- **Don't decide at 0.5.** Set the referral rule from the *economics*: refer when $p \times \text{savings} > \text{investigation cost}$.
- **Class reweighting/oversampling** (`class_weight="balanced"`, SMOTE) rarely improves ranking and **destroys calibration**. It's usually better to train on true rates and move the threshold.

## 2. Worked example: SIU referral model

**Data:** 60,000 synthetic auto claims, **2.06% confirmed fraud**. Signals: days from policy inception to loss, claim amount, report lag, prior claims, police report, witness, loss type, night-time, and how many claims share the same repair shop/clinic. The truth includes an interaction (theft × night × no police report) and a threshold (loss within 30 days of inception).

**Logistic regression odds ratios** (per 1 SD for numerics):
| Signal | Odds ratio | Reading |
|---|---|---|
| Theft (vs collision) | 3.34 | 3.3× the odds of fraud |
| Vandalism | 1.80 | |
| Same provider count | 1.24 | Provider rings |
| Police report filed | 0.68 | Protective |
| Days policy-to-loss | 0.71 | Newer policies are riskier |

**Model comparison (test set):**
| Model | ROC-AUC | PR-AUC | Brier | Mean prediction (actual 0.021) |
|---|---|---|---|---|
| Logistic regression | 0.750 | 0.095 | 0.0195 | 0.021 |
| LightGBM | **0.760** | **0.106** | **0.0193** | 0.020 |
| LightGBM, `class_weight="balanced"` | 0.755 | 0.103 | 0.1436 | **0.323** |

- The GBM edges out logistic regression, mainly from the interaction and the 30-day threshold.
- Rebalancing gave **no ranking gain** and inflated the average predicted fraud probability from 2% to 32%. Anyone using those scores as probabilities (e.g., for expected fraud savings or reserve adjustments) would be badly misled.
- PR-AUC around 0.10 is ~5× the 2% baseline. Fraud models are never "accurate" in an absolute sense; they *concentrate* fraud.

**Calibration (top decile):** actual fraud rate 8.5%. LightGBM predicted 8.8%, logistic regression 5.5% (it misses the interaction, so it under-predicts the riskiest claims), and balanced LightGBM 72%.

**Choosing the referral rule.** Investigation costs \$400; a caught fraud saves 60% of the claim amount.
| Rule | Referred | Fraud caught | Precision | Net benefit |
|---|---|---|---|---|
| p ≥ 0.05 | 1,495 | 37.5% | 9.3% | −\$68k |
| p ≥ 0.10 | 438 | 17.3% | 14.6% | \$77k |
| p ≥ 0.15 | 170 | 10.0% | 21.8% | \$87k (best flat threshold) |
| p ≥ 0.50 | 3 | 0.3% | 33.3% | −\$0.5k |
| **Refer if p × savings > \$400** | 512 | 16.2% | | **\$174k** |

The expected-value rule **doubles** the net benefit over the best flat threshold, because it also uses claim size: a 6% fraud probability on a \$40k claim is worth investigating, while a 20% probability on a \$600 glass claim isn't. This only works because the probabilities are calibrated.

## 3. Python code (core)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb

logreg = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), num_cols),
                                ("cat", OneHotEncoder(drop="first"), ["loss_type"])])),
    ("clf", LogisticRegression(max_iter=2000)),
]).fit(X_train, y_train)

gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                         min_child_samples=100, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, verbose=-1).fit(X_train_cat, y_train)

p = gbm.predict_proba(X_test_cat)[:, 1]
refer = p * 0.6 * X_test.claim_amount > 400        # expected-value referral rule

# If a model's probabilities are off, fix them rather than trusting them:
from sklearn.calibration import CalibratedClassifierCV
cal_model = CalibratedClassifierCV(gbm, method="isotonic", cv=5).fit(X_train_cat, y_train)
```

## 4. When to use which

| Situation | Choose |
|---|---|
| Need explainable reasons for referral (adjuster or regulator facing) | Logistic regression, or GBM + SHAP reason codes |
| Rich data, interactions, max lift | GBM |
| Small labeled set (a few hundred frauds) | Regularized logistic regression |
| Labels are scarce or biased (only investigated claims get labels) | Add unsupervised anomaly scores (Part 8) as features or a second screen |

**Fraud-specific pitfalls:**
- **Label bias.** "Confirmed fraud" only exists where SIU looked, so the model learns past SIU habits. Keep a small random audit sample.
- **Adversarial drift.** Fraud rings adapt, so monitor and retrain often.
- **Fairness.** Avoid proxies for protected classes (e.g., zip code as a race proxy). Many states regulate the use of models in claims handling.
