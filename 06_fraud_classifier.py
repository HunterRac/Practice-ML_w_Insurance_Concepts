"""
Part 6: Logistic regression vs gradient-boosted classifier for claims fraud (SIU referral).
Requires: numpy, pandas, scikit-learn, lightgbm.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

rng = np.random.default_rng(11)
n = 60_000

# ---------------------------------------------------------------
# 1. Synthetic auto claims with confirmed-fraud labels (~3%)
# ---------------------------------------------------------------
df = pd.DataFrame({
    "policy_tenure_months": rng.exponential(30, n).round(),
    "days_policy_to_loss": rng.exponential(200, n).round(),
    "claim_amount": rng.lognormal(8.2, 0.8, n).round(),
    "report_lag_days": rng.exponential(4, n).round(),
    "num_prior_claims_3y": rng.poisson(0.4, n),
    "police_report": rng.choice([0, 1], n, p=[0.35, 0.65]),
    "witness_present": rng.choice([0, 1], n, p=[0.7, 0.3]),
    "loss_type": rng.choice(["collision", "theft", "vandalism", "glass"], n, p=[0.6, 0.12, 0.08, 0.2]),
    "night_loss": rng.choice([0, 1], n, p=[0.75, 0.25]),
    "same_provider_count": rng.poisson(1.0, n),   # claims in book using same repair shop/clinic
})
logit = (-4.6
         + 1.4 * (df.days_policy_to_loss < 30)           # loss right after inception
         + 0.5 * np.log1p(df.claim_amount / 5000)
         + 0.04 * df.report_lag_days.clip(0, 30)
         + 0.35 * df.num_prior_claims_3y
         - 0.8 * df.police_report
         - 0.4 * df.witness_present
         + df.loss_type.map({"collision": 0, "theft": 1.1, "vandalism": 0.6, "glass": -0.5})
         + 0.9 * ((df.loss_type == "theft") & (df.night_loss == 1) & (df.police_report == 0))
         + 0.25 * (df.same_provider_count >= 3) * df.same_provider_count)
df["fraud"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Claims: {n:,}, fraud rate: {df.fraud.mean():.2%}")

X, y = df.drop(columns="fraud"), df.fraud
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
num_cols = [c for c in X.columns if c != "loss_type"]

# ---------------------------------------------------------------
# 2. Logistic regression (scaled numerics + one-hot loss type)
# ---------------------------------------------------------------
logreg = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), num_cols),
                                ("cat", OneHotEncoder(drop="first"), ["loss_type"])])),
    ("clf", LogisticRegression(C=1.0, max_iter=2000)),
]).fit(X_train, y_train)

# Odds ratios per 1 SD (numerics) or vs collision (loss type)
names = num_cols + list(logreg.named_steps["prep"].named_transformers_["cat"].get_feature_names_out())
odds = pd.Series(np.exp(logreg.named_steps["clf"].coef_[0]), index=names).sort_values()
print("\nLogistic regression odds ratios (numerics: per 1 standard deviation)")
print(odds.round(2))

# ---------------------------------------------------------------
# 3. LightGBM classifier
# ---------------------------------------------------------------
Xg_train, Xg_test = X_train.copy(), X_test.copy()
for d in (Xg_train, Xg_test):
    d["loss_type"] = d["loss_type"].astype("category")
gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                         min_child_samples=100, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, verbose=-1, random_state=0).fit(Xg_train, y_train)

# Same GBM with class reweighting, a common (often misguided) reflex for imbalance
gbm_bal = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=100, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, class_weight="balanced",
                             verbose=-1, random_state=0).fit(Xg_train, y_train)

preds = {"Logistic regression": logreg.predict_proba(X_test)[:, 1],
         "LightGBM": gbm.predict_proba(Xg_test)[:, 1],
         "LightGBM balanced": gbm_bal.predict_proba(Xg_test)[:, 1]}

print("\n                       ROC-AUC  PR-AUC  Brier    mean pred (actual {:.3f})".format(y_test.mean()))
for k, p in preds.items():
    print(f"  {k:20s}  {roc_auc_score(y_test, p):.3f}   {average_precision_score(y_test, p):.3f}"
          f"   {brier_score_loss(y_test, p):.4f}   {p.mean():.3f}")

# ---------------------------------------------------------------
# 4. Choose the SIU referral threshold by expected value, not 0.5
#    Investigation costs $400; catching a fraud saves avg 60% of claim amount
# ---------------------------------------------------------------
COST = 400
p = preds["LightGBM"]
saving = 0.6 * X_test.claim_amount.values
best = None
print("\nSIU referral policy (LightGBM)")
print("  threshold  referred  fraud caught  precision   net benefit")
for t in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    r = p >= t
    net = (saving[r] * y_test.values[r]).sum() - COST * r.sum()
    print(f"  {t:8.2f}  {r.sum():8,d}  {y_test.values[r].sum() / y_test.sum():11.1%}"
          f"  {y_test.values[r].mean() if r.any() else 0:9.1%}   ${net:>11,.0f}")

# Optimal rule per claim: refer if p * saving > cost (uses claim size too)
r = p * saving > COST
net = (saving[r] * y_test.values[r]).sum() - COST * r.sum()
print(f"  Refer if p x saving > $400: {r.sum():,} referred, "
      f"{y_test.values[r].sum() / y_test.sum():.1%} caught, net ${net:,.0f}")

# ---------------------------------------------------------------
# 5. Calibration table (decile of predicted prob)
# ---------------------------------------------------------------
cal = pd.DataFrame({"y": y_test.values, "logreg": preds["Logistic regression"],
                    "gbm": preds["LightGBM"], "gbm_bal": preds["LightGBM balanced"]})
cal["decile"] = pd.qcut(cal.gbm, 10, labels=False) + 1
print("\nCalibration by LightGBM decile: mean predicted vs actual fraud rate")
print(cal.groupby("decile")[["y", "logreg", "gbm", "gbm_bal"]].mean().round(3))
