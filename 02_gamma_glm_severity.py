"""
Part 2: Gamma GLM for claim severity, then frequency x severity = pure premium.
Requires: numpy, pandas, statsmodels, scikit-learn. Uses insurance_data.py.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from insurance_data import make_auto_book

policies, claims = make_auto_book()
print(f"{len(claims):,} claims. Severity: mean ${claims.claim_amount.mean():,.0f}, "
      f"median ${claims.claim_amount.median():,.0f}, max ${claims.claim_amount.max():,.0f}")

# Split by POLICY so a policy's claims don't leak across train/test
train_ids, test_ids = train_test_split(policies.policy_id, test_size=0.25, random_state=0)
c_train = claims[claims.policy_id.isin(train_ids)]
c_test = claims[claims.policy_id.isin(test_ids)]

# ---------------------------------------------------------------
# 1. Gamma GLM, log link. One row per claim.
#    (If you only have aggregated loss per policy: response = loss / n_claims,
#     var_weights = n_claims.)
# ---------------------------------------------------------------
sev_formula = ("claim_amount ~ C(region, Treatment('Suburban'))"
               " + vehicle_age + np.log(vehicle_value / 25000)")
sev = smf.glm(sev_formula, data=c_train,
              family=sm.families.Gamma(link=sm.families.links.Log())).fit()
print(sev.summary().tables[1])

print("\nSeverity relativities   (truth: Urban 1.15, Rural 0.90, veh_age 0.970/yr, "
      "log-value elasticity 0.50)")
print(pd.DataFrame({"estimate": np.exp(sev.params).round(3),
                    "raw_coef": sev.params.round(3)}))
print(f"\nEstimated Gamma dispersion phi = {sev.scale:.3f} "
      f"(truth 1/shape = 0.5, so CV = sqrt(phi) = {np.sqrt(sev.scale):.2f})")

# ---------------------------------------------------------------
# 2. Why not plain OLS on claim_amount? Compare to a log-OLS
# ---------------------------------------------------------------
ols_log = smf.ols(sev_formula.replace("claim_amount", "np.log(claim_amount)"), data=c_train).fit()
naive = np.exp(ols_log.predict(c_test))                               # biased low
smeared = naive * np.mean(np.exp(ols_log.resid))                      # Duan smearing fix
gamma_pred = sev.predict(c_test)
print("\nTest-set total predicted vs actual claim dollars")
print(f"  Actual:                    ${c_test.claim_amount.sum():>14,.0f}")
print(f"  Gamma GLM:                 ${gamma_pred.sum():>14,.0f}")
print(f"  log-OLS, naive exp():      ${naive.sum():>14,.0f}   <- retransformation bias")
print(f"  log-OLS + smearing:        ${smeared.sum():>14,.0f}")

# ---------------------------------------------------------------
# 3. Frequency GLM on the same book (as in Part 1) and combine
# ---------------------------------------------------------------
p_train = policies[policies.policy_id.isin(train_ids)]
p_test = policies[policies.policy_id.isin(test_ids)].copy()
freq = smf.glm("claim_count ~ C(age_band, Treatment('41-60')) + C(region, Treatment('Suburban'))"
               " + vehicle_age + prior_claims + credit_score",
               data=p_train, family=sm.families.Poisson(),
               offset=np.log(p_train.exposure)).fit()

p_test["pred_freq"] = freq.predict(p_test, offset=np.zeros(len(p_test)))  # per policy-year
p_test["pred_sev"] = sev.predict(p_test)
p_test["pure_premium"] = p_test.pred_freq * p_test.pred_sev
p_test["expected_loss"] = p_test.pure_premium * p_test.exposure

print("\nPure premium = frequency x severity (test set)")
print(f"  Actual loss:   ${p_test.total_loss.sum():,.0f}")
print(f"  Expected loss: ${p_test.expected_loss.sum():,.0f}")
print(f"  A/E = {p_test.total_loss.sum() / p_test.expected_loss.sum():.3f}")

p_test["decile"] = pd.qcut(p_test.pure_premium, 10, labels=False) + 1
lift = p_test.groupby("decile").agg(exposure=("exposure", "sum"),
                                    actual=("total_loss", "sum"),
                                    expected=("expected_loss", "sum"))
lift["actual_LC"] = lift.actual / lift.exposure       # loss cost per policy-year
lift["expected_LC"] = lift.expected / lift.exposure
lift["A/E"] = lift.actual / lift.expected
print("\nLoss-cost lift by pure-premium decile")
print(lift[["actual_LC", "expected_LC", "A/E"]].round(2))

# ---------------------------------------------------------------
# 4. Price one policy
# ---------------------------------------------------------------
new = pd.DataFrame({"age_band": ["18-25"], "region": ["Urban"], "vehicle_age": [3],
                    "prior_claims": [1], "credit_score": [650], "vehicle_value": [35_000]})
f = freq.predict(new, offset=np.zeros(1)).iloc[0]
s = sev.predict(new).iloc[0]
print(f"\nNew policy: freq {f:.4f} x sev ${s:,.0f} = pure premium ${f * s:,.2f}")
