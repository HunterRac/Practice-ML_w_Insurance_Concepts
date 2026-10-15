"""
Part 8: Unsupervised anomaly detection (Isolation Forest, LOF) for claims fraud screening.
No fraud labels are used for training; hidden labels are only used to evaluate.
Requires: numpy, pandas, scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(5)
n = 40_000

# ---------------------------------------------------------------
# 1. Normal auto physical-damage claims (correlated features)
# ---------------------------------------------------------------
damage_score = rng.gamma(2.0, 1.5, n)                       # appraiser damage estimate, 0-15ish
df = pd.DataFrame({
    "damage_score": damage_score,
    "claim_amount": np.exp(7.0 + 0.25 * damage_score + rng.normal(0, 0.25, n)),
    "rental_days": np.round(damage_score * 1.5 + rng.normal(0, 1.5, n)).clip(0),
    "report_lag_days": rng.exponential(3, n),
    "days_since_inception": rng.uniform(0, 365 * 3, n),
    "num_line_items": np.round(4 + damage_score * 2 + rng.normal(0, 2, n)).clip(1),
})
df["anomaly_type"] = "normal"

# ---------------------------------------------------------------
# 2. Inject ~2% fraud of four kinds (labels hidden from the model)
# ---------------------------------------------------------------
def pick(k):
    return rng.choice(df.index[df.anomaly_type == "normal"], k, replace=False)

# (a) Padded estimate: amount far too high for the damage
i = pick(250); df.loc[i, "claim_amount"] *= rng.uniform(2.5, 5, len(i)); df.loc[i, "anomaly_type"] = "padded_estimate"
# (b) Rental abuse: long rental for minor damage
i = pick(150); df.loc[i, "rental_days"] = rng.integers(20, 45, len(i)); df.loc[i, "anomaly_type"] = "rental_abuse"
# (c) Jump-in / staged: big claim days after inception, reported late, many line items
i = pick(200)
df.loc[i, "days_since_inception"] = rng.uniform(1, 20, len(i))
df.loc[i, "damage_score"] = rng.uniform(6, 12, len(i))
df.loc[i, "claim_amount"] = np.exp(7.0 + 0.25 * df.loc[i, "damage_score"]) * rng.uniform(1.2, 2.0, len(i))
df.loc[i, "report_lag_days"] = rng.uniform(10, 30, len(i))
df.loc[i, "num_line_items"] = np.round(4 + df.loc[i, "damage_score"] * 2 + rng.uniform(8, 15, len(i)))
df.loc[i, "anomaly_type"] = "staged_new_policy"

# (d) Soft fraud spread thin: everything a bit inflated, nothing extreme on its own
i = pick(200)
df.loc[i, "claim_amount"] *= rng.uniform(1.4, 1.7, len(i))
df.loc[i, "rental_days"] += rng.integers(4, 7, len(i))
df.loc[i, "num_line_items"] += rng.integers(4, 7, len(i))
df.loc[i, "report_lag_days"] = rng.uniform(6, 10, len(i))
df.loc[i, "anomaly_type"] = "soft_inflation"

df["is_fraud"] = (df.anomaly_type != "normal").astype(int)
print(f"Claims: {n:,}; hidden fraud: {df.is_fraud.sum()} ({df.is_fraud.mean():.2%})")

# ---------------------------------------------------------------
# 3. Feature engineering: ratios expose "inconsistency" fraud
# ---------------------------------------------------------------
X = pd.DataFrame({
    "log_amount": np.log(df.claim_amount),
    "amount_per_damage": np.log(df.claim_amount) - 0.25 * df.damage_score,  # residual vs expected
    "rental_per_damage": df.rental_days - 1.5 * df.damage_score,
    "items_per_damage": df.num_line_items - 2 * df.damage_score,
    "log_report_lag": np.log1p(df.report_lag_days),
    "log_days_since_inception": np.log1p(df.days_since_inception),
    "damage_score": df.damage_score,
})
Xs = RobustScaler().fit_transform(X)        # median/IQR scaling: outliers don't distort it

# ---------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------
iso = IsolationForest(n_estimators=300, max_samples=512, contamination="auto", random_state=0).fit(Xs)
score_iso = -iso.score_samples(Xs)          # higher = more anomalous

lof = LocalOutlierFactor(n_neighbors=50).fit(Xs)
score_lof = -lof.negative_outlier_factor_

# Baseline: max absolute robust z-score across features (a typical rules-style check)
score_z = np.abs(Xs).max(axis=1)

# Same Isolation Forest on raw features only (no ratio engineering)
raw = RobustScaler().fit_transform(df[["claim_amount", "rental_days", "report_lag_days",
                                       "days_since_inception", "num_line_items", "damage_score"]])
score_iso_raw = -IsolationForest(n_estimators=300, max_samples=512, random_state=0).fit(raw).score_samples(raw)

# Ensemble: average the percentile ranks of z-score and Isolation Forest
score_ens = (pd.Series(score_z).rank(pct=True) + pd.Series(score_iso).rank(pct=True)).values / 2

def top_k_stats(score, k_frac=0.02):
    k = int(len(score) * k_frac)
    top = np.argsort(-score)[:k]
    return df.is_fraud.values[top].mean(), df.is_fraud.values[top].sum() / df.is_fraud.sum()

print("\n                             ROC-AUC  PR-AUC  precision@2%  recall@2%")
for name, s in [("Max robust z-score", score_z), ("Isolation Forest (raw feats)", score_iso_raw),
                ("Isolation Forest (ratios)", score_iso), ("LOF (ratios)", score_lof),
                ("Ensemble: z + IF ratios", score_ens)]:
    p, r = top_k_stats(s)
    print(f"  {name:28s} {roc_auc_score(df.is_fraud, s):.3f}   {average_precision_score(df.is_fraud, s):.3f}"
          f"     {p:6.1%}      {r:6.1%}")

# ---------------------------------------------------------------
# 5. Which fraud types does each method catch? (top 2% flagged)
# ---------------------------------------------------------------
k = int(n * 0.02)
rows = {}
for name, s in [("z-score", score_z), ("IF raw", score_iso_raw), ("IF ratios", score_iso), ("LOF", score_lof), ("ensemble", score_ens)]:
    flagged = np.zeros(n, bool); flagged[np.argsort(-s)[:k]] = True
    rows[name] = df[df.is_fraud == 1].assign(f=flagged[df.is_fraud == 1]).groupby("anomaly_type").f.mean()
print("\nDetection rate by fraud type (top 2% flagged)")
print(pd.DataFrame(rows).map(lambda v: f"{v:.0%}"))

# ---------------------------------------------------------------
# 6. Reason codes for the top flagged claims: which features are most extreme
# ---------------------------------------------------------------
top5 = np.argsort(-score_iso)[:5]
print("\nTop 5 flagged claims, with the two most extreme features (robust z)")
for i in top5:
    z = pd.Series(Xs[i], index=X.columns)
    reasons = ", ".join(f"{c} {z[c]:+.1f}" for c in z.abs().nlargest(2).index)
    print(f"  claim {i:5d}  score {score_iso[i]:.3f}  true type: {df.anomaly_type[i]:18s}  {reasons}")
