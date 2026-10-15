# Part 8: Anomaly Detection (Isolation Forest, LOF) for Fraud Screening

Script: [`08_anomaly_detection_fraud.py`](/mnt/project-files/ml-insurance/08_anomaly_detection_fraud.py)

## 1. Definition

Part 6 needed **labeled** fraud. Most carriers have few reliable labels, and new schemes have none. **Unsupervised anomaly detection** scores how unusual each claim is relative to the bulk of claims, with no labels.

**Isolation Forest.** Build many random trees, each splitting on a random feature at a random value. Anomalies are "few and different", so they get **isolated in fewer splits**:

$$s(x) = 2^{-\,\mathbb{E}[h(x)]\,/\,c(n)}$$

Here $h(x)$ is the path length to isolate $x$ and $c(n)$ normalizes for sample size. Scores near 1 mean anomalous. It is fast, scales to millions of rows, needs no distance metric, and handles mixed scales.

**Local Outlier Factor (LOF).** Compares a point's local density to its neighbors' densities. It is good at points that are outliers *relative to their local cluster*, but it depends on distance (sensitive to scaling and dimensionality) and is slower.

**Robust z-score baseline.** Scale each feature by median/IQR and take the max |z| across features. This is the "rules engine" in model form, and a baseline you must beat.

**The key insight for insurance:** raw features hide fraud (a \$9k claim is normal; a \$9k claim on a 2/15 damage score isn't). **Engineer "inconsistency" features**: residuals against what's expected, such as amount vs damage, rental days vs damage, or line items vs damage.

## 2. Worked example: screening 40,000 physical-damage claims

**Data:** normal claims where amount, rental days and line items all scale with the appraiser's damage score. Four hidden fraud types were injected (2% total) and **never shown to the models**:
| Type | What it looks like |
|---|---|
| Padded estimate | Amount 2.5–5× too high for the damage |
| Rental abuse | 20–45 rental days on minor damage |
| Staged / new policy | Big claim days after inception, reported late, many line items |
| Soft inflation | Everything ~1.5× high, nothing extreme on its own |

**Results (flag the top 2% for review):**
| Method | ROC-AUC | PR-AUC | Precision@2% |
|---|---|---|---|
| Max robust z-score (on engineered ratios) | 0.981 | 0.677 | 60.5% |
| Isolation Forest, **raw** features | 0.941 | 0.317 | 36.4% |
| Isolation Forest, engineered ratios | 0.977 | 0.594 | 51.6% |
| LOF, engineered ratios | 0.604 | 0.085 | 14.0% |
| **Ensemble** (average rank of z + IF) | **0.983** | 0.675 | 59.8% |

**Detection rate by type:**
| Type | z-score | IF raw | IF ratios | LOF | Ensemble |
|---|---|---|---|---|---|
| Padded estimate | 51% | 10% | 36% | 17% | 48% |
| Rental abuse | 100% | 37% | 45% | 4% | 87% |
| **Soft inflation** | 10% | 6% | **28%** | 25% | 15% |
| Staged / new policy | 94% | 100% | 100% | 6% | 100% |

**Lessons, honestly stated:**
1. **Feature engineering beat algorithm choice.** The same Isolation Forest went from 36% to 52% precision by switching from raw features to ratios.
2. **A simple z-score won overall**, because three of the four schemes are extreme on a single engineered feature. Don't assume the fancy method wins; always benchmark.
3. **Isolation Forest earns its place on multivariate anomalies.** For soft inflation, where no one feature is extreme, it caught 28% vs 10% for z-scores.
4. **An ensemble is robust**: nearly the best overall with no blind spot on any type.
5. LOF struggled here: one dense population with outliers isn't its sweet spot, and 7 dimensions blur distances.

**Reason codes.** For each flagged claim, the script lists the most extreme features (e.g., "items_per_damage +5.3, log_days_since_inception −4.7"). SIU staff need this to act on a flag.

## 3. Python code (core)

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

# Engineer inconsistency features (residual vs expected given damage)
X = pd.DataFrame({
    "amount_per_damage": np.log(df.claim_amount) - 0.25 * df.damage_score,
    "rental_per_damage": df.rental_days - 1.5 * df.damage_score,
    "items_per_damage": df.num_line_items - 2 * df.damage_score,
    "log_report_lag": np.log1p(df.report_lag_days),
    "log_days_since_inception": np.log1p(df.days_since_inception),
    # ...
})
Xs = RobustScaler().fit_transform(X)

iso = IsolationForest(n_estimators=300, max_samples=512, random_state=0).fit(Xs)
score_iso = -iso.score_samples(Xs)                  # higher = more anomalous
score_z = np.abs(Xs).max(axis=1)                    # robust z baseline
score = (pd.Series(score_iso).rank(pct=True) + pd.Series(score_z).rank(pct=True)) / 2

review_queue = df.iloc[np.argsort(-score)[: int(0.02 * len(df))]]
```

In practice, the "expected" values in the ratios (0.25, 1.5, 2) come from a regression of amount, rental days or line items on damage and claim characteristics. Your anomaly features are then that model's residuals.

## 4. When to use it, and when not to

**Use when:** labels are scarce or biased; you're looking for new schemes; as a feature or second screen next to a supervised model (Part 6); for provider/billing review, premium audit (misreported payroll or mileage), and data-quality checks.

**Be careful:**
- **Unusual ≠ fraudulent.** Catastrophe claims, exotic cars and legitimate total losses are all anomalous. Segment first (by line, loss type, CAT vs non-CAT) and score within segments.
- **The `contamination` parameter** only sets a threshold; it doesn't change the scores. Choose the review volume from SIU capacity.
- **Feedback loop:** once SIU reviews flagged claims, you get labels, so move toward a supervised model (Part 6) and keep the anomaly score as a feature.
- **Network fraud** (rings sharing phones, addresses, providers) is better found with graph methods (connected components, link analysis).
