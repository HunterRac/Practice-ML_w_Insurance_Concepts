"""
Part 3: Tweedie GLM for pure premium, modeled directly.
Requires: numpy, pandas, statsmodels, scikit-learn. Uses insurance_data.py.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from insurance_data import make_auto_book

policies, claims = make_auto_book()
policies["loss_cost"] = policies.total_loss / policies.exposure   # $ per policy-year
print(f"Policies with zero loss: {(policies.total_loss == 0).mean():.1%}")
print(f"Mean loss cost: ${np.average(policies.loss_cost, weights=policies.exposure):,.2f} per policy-year")

train, test = train_test_split(policies, test_size=0.25, random_state=0)
test = test.copy()

rhs = ("C(age_band, Treatment('41-60')) + C(region, Treatment('Suburban'))"
       " + vehicle_age + prior_claims + credit_score + np.log(vehicle_value / 25000)")


def gini(actual, pred, weight):
    """Normalized Gini from an ordered Lorenz curve (sorted by prediction)."""
    order = np.argsort(pred)
    a, w = np.asarray(actual)[order], np.asarray(weight)[order]
    cum_w = np.cumsum(w) / w.sum()
    cum_a = np.cumsum(a) / a.sum()
    g = 1 - 2 * np.trapezoid(cum_a, cum_w)
    order_best = np.argsort(np.asarray(actual) / np.asarray(weight))
    a_b, w_b = np.asarray(actual)[order_best], np.asarray(weight)[order_best]
    g_best = 1 - 2 * np.trapezoid(np.cumsum(a_b) / a_b.sum(), np.cumsum(w_b) / w_b.sum())
    return g / g_best


# ---------------------------------------------------------------
# 1. Tweedie GLM: response = loss cost, var_weights = exposure
#    1 < p < 2 -> compound Poisson-Gamma (point mass at 0 + continuous)
# ---------------------------------------------------------------
results = {}
for p in [1.2, 1.5, 1.8]:
    m = smf.glm(f"loss_cost ~ {rhs}", data=train,
                family=sm.families.Tweedie(var_power=p, link=sm.families.links.Log()),
                var_weights=train.exposure).fit()
    pred = m.predict(test)
    results[p] = m
    print(f"p={p}: test A/E = {test.total_loss.sum() / (pred * test.exposure).sum():.3f}, "
          f"normalized Gini = {gini(test.total_loss, pred, test.exposure):.3f}")

tw = results[1.5]
print("\nTweedie (p=1.5) relativities")
print(np.exp(tw.params).round(3))

# ---------------------------------------------------------------
# 2. Compare with frequency x severity (Part 1 + Part 2 approach)
# ---------------------------------------------------------------
freq = smf.glm("claim_count ~ C(age_band, Treatment('41-60')) + C(region, Treatment('Suburban'))"
               " + vehicle_age + prior_claims + credit_score",
               data=train, family=sm.families.Poisson(), offset=np.log(train.exposure)).fit()
c_train = claims[claims.policy_id.isin(train.policy_id)]
sev = smf.glm("claim_amount ~ C(region, Treatment('Suburban')) + vehicle_age"
              " + np.log(vehicle_value / 25000)", data=c_train,
              family=sm.families.Gamma(link=sm.families.links.Log())).fit()

test["pp_tweedie"] = tw.predict(test)
test["pp_freqsev"] = freq.predict(test, offset=np.zeros(len(test))) * sev.predict(test)
test["pp_true"] = test.true_freq * test.true_sev

print("\nTest set comparison")
for col in ["pp_tweedie", "pp_freqsev", "pp_true"]:
    print(f"  {col:11s} A/E = {test.total_loss.sum() / (test[col] * test.exposure).sum():.3f}  "
          f"Gini = {gini(test.total_loss, test[col], test.exposure):.3f}  "
          f"corr with truth = {np.corrcoef(test[col], test.pp_true)[0, 1]:.3f}")

# Side-by-side relativities: Tweedie blends freq and sev effects into one factor
fs_rel = pd.Series({
    "Urban": np.exp(freq.params["C(region, Treatment('Suburban'))[T.Urban]"]
                    + sev.params["C(region, Treatment('Suburban'))[T.Urban]"]),
    "vehicle_age": np.exp(freq.params["vehicle_age"] + sev.params["vehicle_age"]),
})
print("\nCombined relativity, freq x sev vs Tweedie")
urban = "C(region, Treatment('Suburban'))[T.Urban]"
print(f"  Urban:       {fs_rel['Urban']:.3f} vs {np.exp(tw.params[urban]):.3f}"
      "  (truth 1.35 x 1.15 = 1.553)")
print(f"  vehicle_age: {fs_rel['vehicle_age']:.4f} vs {np.exp(tw.params['vehicle_age']):.4f}"
      f"  (truth 0.980 x 0.970 = 0.951)")
