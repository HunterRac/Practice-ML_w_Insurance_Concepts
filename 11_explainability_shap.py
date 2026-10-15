"""
Part 11: Explaining a GBM pricing model with SHAP (global, local, interactions),
and turning SHAP values into GLM-style rating relativities.
Requires: numpy, pandas, scikit-learn, lightgbm, shap. Uses insurance_data.py.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from insurance_data import make_auto_book, FEATURES

policies, _ = make_auto_book(n=200_000)
policies["region"] = policies["region"].astype("category")
train, test = train_test_split(policies, test_size=0.2, random_state=0)

gbm = lgb.train(dict(objective="poisson", learning_rate=0.03, num_leaves=15, min_child_samples=200,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, seed=0),
                lgb.Dataset(train[FEATURES], train.claim_count / train.exposure, weight=train.exposure),
                num_boost_round=400)

# ---------------------------------------------------------------
# 1. SHAP values. For a Poisson GBM they are on the LOG scale:
#    log(freq) = base + sum(shap)  ->  freq = exp(base) x prod(exp(shap))
#    so exp(shap_j) is a per-policy multiplicative "relativity" for feature j.
# ---------------------------------------------------------------
X = test[FEATURES].sample(5000, random_state=0)
explainer = shap.TreeExplainer(gbm)
sv = explainer.shap_values(X)
base = explainer.expected_value
pred_log = np.log(gbm.predict(X))
print(f"Additivity check: max |base + sum(shap) - log(pred)| = {np.abs(base + sv.sum(1) - pred_log).max():.2e}")
print(f"Base frequency exp(base) = {np.exp(base):.4f}")

# ---------------------------------------------------------------
# 2. Global importance: mean |SHAP|
# ---------------------------------------------------------------
imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
print("\nGlobal importance, mean |SHAP| (log scale) and typical multiplicative swing")
print(pd.DataFrame({"mean_abs_shap": imp, "~x factor": np.exp(imp)}).round(3))

# ---------------------------------------------------------------
# 3. Dependence: SHAP-derived relativity curves (vs the truth used to simulate)
# ---------------------------------------------------------------
S = pd.DataFrame(sv, columns=FEATURES, index=X.index)

def shap_curve(feature, bins):
    g = pd.cut(X[feature], bins)
    out = np.exp(S[feature].groupby(g, observed=True).mean())
    return out

credit = shap_curve("credit_score", [449, 550, 600, 650, 700, 750, 800, 850])
credit_ref = credit.iloc[4]  # (700, 750] band as base
true_credit = [np.exp(0.006 * max(700 - m, 0)) for m in [525, 575, 625, 675, 725, 775, 825]]
print("\nCredit score relativity from SHAP (normalized to 700-750 band) vs truth")
print(pd.DataFrame({"SHAP_rel": (credit / credit_ref).round(3).values, "truth": np.round(true_credit, 3)},
                   index=credit.index.astype(str)))

region = np.exp(S["region"].groupby(X["region"], observed=True).mean())
print("\nRegion relativity from SHAP (normalized to Suburban) vs truth (Urban 1.35, Rural 0.80)")
print((region / region["Suburban"]).round(3))

veh_val = shap_curve("vehicle_value", [0, 15_000, 25_000, 40_000, 1e7])
print("\nVehicle value SHAP relativity (truth: NO frequency effect, should be ~flat)")
print((veh_val / veh_val.iloc[1]).round(3))

# ---------------------------------------------------------------
# 4. Interaction: SHAP interaction values for driver_age x region
# ---------------------------------------------------------------
Xi = X.iloc[:1500]
# shap's interaction path needs numeric input: pass LightGBM's internal category codes
siv = explainer.shap_interaction_values(Xi.assign(region=Xi.region.cat.codes))
ia, ir = FEATURES.index("driver_age"), FEATURES.index("region")
inter = pd.Series(2 * siv[:, ia, ir], index=Xi.index)   # off-diagonal is split in half
young = Xi.driver_age <= 25
urban = Xi.region == "Urban"
print("\nAge x region interaction effect, exp(mean SHAP interaction)  (truth: young & urban extra 1.30x)")
tab = pd.DataFrame({"young": young, "urban": urban, "inter": inter}).groupby(["young", "urban"]).inter.mean()
print(np.exp(tab).round(3))
pair_strength = pd.Series({f"{FEATURES[a]} x {FEATURES[b]}": np.abs(siv[:, a, b]).mean() * 2
                           for a in range(len(FEATURES)) for b in range(a + 1, len(FEATURES))})
print("\nStrongest pairwise interactions (mean |SHAP interaction|)")
print(pair_strength.sort_values(ascending=False).head(4).round(4))

# ---------------------------------------------------------------
# 5. Local explanation: why is this policy priced where it is?
#    Rendered as a rate-manual style factor build-up.
# ---------------------------------------------------------------
pol = pd.DataFrame({"driver_age": [22], "vehicle_age": [2], "vehicle_value": [32_000],
                    "region": pd.Categorical(["Urban"], categories=policies.region.cat.categories),
                    "prior_claims": [1], "credit_score": [610]})
s1 = explainer.shap_values(pol)[0]
print("\nPolicy: 22-yr-old, Urban, 2-yr-old $32k car, 1 prior claim, credit 610")
print(f"  {'Base frequency':28s}            {np.exp(base):.4f}")
running = np.exp(base)
for f, v in sorted(zip(FEATURES, s1), key=lambda t: -abs(t[1])):
    running *= np.exp(v)
    print(f"  x {f:15s} = {str(pol[f].iloc[0]):>8s}   x{np.exp(v):.3f}   -> {running:.4f}")
print(f"  Model prediction:                       {gbm.predict(pol)[0]:.4f}")
