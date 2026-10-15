"""
Part 4: Gradient boosting (LightGBM) for claim frequency, vs. the Poisson GLM.
Requires: numpy, pandas, statsmodels, scikit-learn, lightgbm. Uses insurance_data.py.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance
from insurance_data import make_auto_book, FEATURES

policies, _ = make_auto_book(n=200_000)
policies["region"] = policies["region"].astype("category")
policies["freq"] = policies.claim_count / policies.exposure

train, test = train_test_split(policies, test_size=0.2, random_state=0)
train, valid = train_test_split(train, test_size=0.2, random_state=1)
test = test.copy()

# ---------------------------------------------------------------
# 1. Baseline: Poisson GLM (main effects only, linear credit score)
# ---------------------------------------------------------------
glm = smf.glm("claim_count ~ C(age_band) + C(region) + vehicle_age + prior_claims + credit_score",
              data=train, family=sm.families.Poisson(), offset=np.log(train.exposure)).fit()
test["glm_freq"] = glm.predict(test, offset=np.zeros(len(test)))

# ---------------------------------------------------------------
# 2. LightGBM with Poisson objective
#    Exposure handling: target = frequency, weight = exposure.
#    (Alternative: target = count with init_score = log(exposure).)
# ---------------------------------------------------------------
params = dict(objective="poisson", learning_rate=0.03, num_leaves=15,
              min_child_samples=200,       # bigger leaves: claims are rare and noisy
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1, seed=0)
dtrain = lgb.Dataset(train[FEATURES], train.freq, weight=train.exposure)
dvalid = lgb.Dataset(valid[FEATURES], valid.freq, weight=valid.exposure, reference=dtrain)
gbm = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"Early stopping chose {gbm.best_iteration} trees")
test["gbm_freq"] = gbm.predict(test[FEATURES], num_iteration=gbm.best_iteration)


# ---------------------------------------------------------------
# 3. Compare
# ---------------------------------------------------------------
def gini(actual, pred, weight):
    o = np.argsort(pred)
    a, w = np.asarray(actual)[o], np.asarray(weight)[o]
    g = 1 - 2 * np.trapezoid(np.cumsum(a) / a.sum(), np.cumsum(w) / w.sum())
    ob = np.argsort(np.asarray(actual) / np.asarray(weight))
    ab, wb = np.asarray(actual)[ob], np.asarray(weight)[ob]
    return g / (1 - 2 * np.trapezoid(np.cumsum(ab) / ab.sum(), np.cumsum(wb) / wb.sum()))


print("\nTest set                Poisson dev   Gini    corr w/ truth   A/E")
for name, col in [("GLM", "glm_freq"), ("LightGBM", "gbm_freq"), ("Truth", "true_freq")]:
    dev = mean_poisson_deviance(test.freq, test[col], sample_weight=test.exposure)
    print(f"  {name:20s}  {dev:.5f}   {gini(test.claim_count, test[col], test.exposure):.3f}"
          f"     {np.corrcoef(test[col], test.true_freq)[0, 1]:.3f}        "
          f"{test.claim_count.sum() / (test[col] * test.exposure).sum():.3f}")

# ---------------------------------------------------------------
# 4. Did the GBM find the hidden structure?
# ---------------------------------------------------------------
# (a) Young x Urban interaction (truth: extra 1.3x)
seg = test.assign(young=test.age_band.eq("18-25"), urban=test.region.eq("Urban"))
tab = seg.groupby(["young", "urban"], observed=True).apply(
    lambda d: pd.Series({"actual": d.claim_count.sum() / d.exposure.sum(),
                         "glm": np.average(d.glm_freq, weights=d.exposure),
                         "gbm": np.average(d.gbm_freq, weights=d.exposure)}),
    include_groups=False)
print("\nFrequency by segment (young driver x urban)")
print(tab.round(4))

# (b) Non-linear credit curve: partial dependence
grid = np.arange(500, 851, 50)
base = test[FEATURES].sample(5000, random_state=0)
pdp = [gbm.predict(base.assign(credit_score=c), num_iteration=gbm.best_iteration).mean()
       for c in grid]
glm_pdp = [glm.predict(test.sample(5000, random_state=0).assign(credit_score=c),
                       offset=np.zeros(5000)).mean() for c in grid]
true_curve = np.exp(0.006 * np.clip(700 - grid, 0, None))
print("\nCredit score effect, relative to score 700")
print(pd.DataFrame({"credit": grid,
                    "truth": true_curve / 1.0,
                    "GBM": np.array(pdp) / pdp[list(grid).index(700)],
                    "GLM": np.array(glm_pdp) / glm_pdp[list(grid).index(700)]}).round(3)
      .to_string(index=False))

# ---------------------------------------------------------------
# 5. Feature importance (gain)
# ---------------------------------------------------------------
imp = pd.Series(gbm.feature_importance("gain"), index=FEATURES)
print("\nFeature importance (share of total gain)")
print((imp / imp.sum()).sort_values(ascending=False).round(3))
