"""
Part 5: Random forest for claim triage at first notice of loss (FNOL):
predict which bodily-injury claims will get an attorney (litigation risk).
Requires: numpy, pandas, scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(7)
n = 30_000

# ---------------------------------------------------------------
# 1. Synthetic FNOL data for auto bodily-injury claims
# ---------------------------------------------------------------
df = pd.DataFrame({
    "report_lag_days": rng.exponential(6, n).round(),          # days from accident to report
    "claimant_age": rng.integers(16, 85, n),
    "injury_type": rng.choice(["none_reported", "soft_tissue", "fracture", "head"], n,
                              p=[0.35, 0.45, 0.15, 0.05]),
    "damage_severity": rng.integers(1, 6, n),                   # adjuster 1-5 scale at FNOL
    "num_claimants": rng.choice([1, 2, 3, 4], n, p=[0.6, 0.25, 0.1, 0.05]),
    "tort_state": rng.choice([0, 1], n, p=[0.35, 0.65]),        # 0 = no-fault state
    "liability_clear": rng.choice([0, 1], n, p=[0.3, 0.7]),
    "adjuster_contact_hrs": rng.exponential(20, n).round(),     # hours to first contact
    "noise_zip_code": rng.integers(10_000, 99_999, n),          # high-cardinality noise
    "noise_coin_flip": rng.integers(0, 2, n),                   # pure noise
})

# True log-odds of attorney involvement (note the interactions and thresholds)
inj = df.injury_type.map({"none_reported": -1.5, "soft_tissue": 0.3, "fracture": 0.8, "head": 1.2})
logit = (-2.2 + inj
         + 0.9 * df.tort_state
         + 0.6 * (df.report_lag_days > 14)                      # late reporting
         + 0.5 * (df.adjuster_contact_hrs > 48)                 # slow contact -> lawyer up
         - 0.6 * df.liability_clear
         + 0.35 * (df.num_claimants - 1)
         + 0.8 * ((df.injury_type == "soft_tissue") & (df.tort_state == 1) & (df.liability_clear == 0))
         - 0.01 * np.abs(df.claimant_age - 40))
df["attorney"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Attorney involvement rate: {df.attorney.mean():.1%}")

X = pd.get_dummies(df.drop(columns="attorney"), columns=["injury_type"], dtype=int)
y = df.attorney
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# ---------------------------------------------------------------
# 2. Single tree vs random forest
# ---------------------------------------------------------------
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)     # fully grown, overfits
tree_pruned = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, random_state=0).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=20,
                            oob_score=True, n_jobs=-1, random_state=0).fit(X_train, y_train)

print("\n                       train AUC   test AUC   test PR-AUC")
for name, m in [("Single tree (full)", tree), ("Single tree (depth 5)", tree_pruned), ("Random forest", rf)]:
    p_tr, p_te = m.predict_proba(X_train)[:, 1], m.predict_proba(X_test)[:, 1]
    print(f"  {name:22s} {roc_auc_score(y_train, p_tr):.3f}      {roc_auc_score(y_test, p_te):.3f}"
          f"      {average_precision_score(y_test, p_te):.3f}")
print(f"  Random forest OOB score (accuracy, free validation): {rf.oob_score_:.3f}")

# ---------------------------------------------------------------
# 3. Variable discovery: impurity importance vs permutation importance
# ---------------------------------------------------------------
imp = pd.DataFrame({
    "impurity (MDI)": rf.feature_importances_,
    "permutation (test AUC drop)": permutation_importance(
        rf, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1
    ).importances_mean,
}, index=X.columns).sort_values("permutation (test AUC drop)", ascending=False)
print("\nFeature importance")
print(imp.round(4))

# ---------------------------------------------------------------
# 4. Triage: route the top 20% riskiest claims to senior adjusters
# ---------------------------------------------------------------
p = rf.predict_proba(X_test)[:, 1]
cut = np.quantile(p, 0.8)
flag = p >= cut
print(f"\nTriage: top 20% by score (prob >= {cut:.2f})")
print(f"  Attorney rate in flagged group:   {y_test[flag].mean():.1%}")
print(f"  Attorney rate in the rest:        {y_test[~flag].mean():.1%}")
print(f"  Share of all attorney claims caught: {y_test[flag].sum() / y_test.sum():.1%}")

# Partial dependence on report lag (does the forest find the 14-day threshold?)
base = X_test.sample(3000, random_state=0)
print("\nPredicted attorney prob by report lag (partial dependence)")
for lag in [0, 7, 13, 15, 21, 30]:
    print(f"  lag {lag:2d} days: {rf.predict_proba(base.assign(report_lag_days=lag))[:, 1].mean():.3f}")
