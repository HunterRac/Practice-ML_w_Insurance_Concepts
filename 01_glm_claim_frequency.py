"""
GLM for claim frequency (Poisson with log link and exposure offset).
Synthetic personal auto book. Requires: numpy, pandas, statsmodels, scikit-learn.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance

rng = np.random.default_rng(42)
n = 50_000

# ---------------------------------------------------------------
# 1. Simulate a personal auto portfolio
# ---------------------------------------------------------------
df = pd.DataFrame({
    "driver_age": rng.integers(18, 85, n),
    "vehicle_age": rng.integers(0, 20, n),
    "region": rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.4, 0.4, 0.2]),
    "prior_claims": rng.poisson(0.3, n).clip(0, 3),
    "exposure": rng.uniform(0.1, 1.0, n),          # policy-years in force
})
df["age_band"] = pd.cut(df["driver_age"], [17, 25, 40, 60, 120],
                        labels=["18-25", "26-40", "41-60", "61+"])

# "True" multiplicative risk structure (unknown to the modeler)
base_rate = 0.08                                   # claims per policy-year
age_rel = {"18-25": 1.9, "26-40": 1.1, "41-60": 1.0, "61+": 1.25}
region_rel = {"Urban": 1.35, "Suburban": 1.0, "Rural": 0.8}
true_rate = (base_rate
             * df["age_band"].map(age_rel).astype(float)
             * df["region"].map(region_rel)
             * np.exp(-0.02 * df["vehicle_age"])    # older cars: slightly fewer claims
             * np.exp(0.25 * df["prior_claims"]))   # claim history matters
df["claim_count"] = rng.poisson(true_rate * df["exposure"])

print("Portfolio snapshot")
print(df.head())
print(f"\nTotal exposure: {df.exposure.sum():,.0f} policy-years")
print(f"Total claims:   {df.claim_count.sum():,}")
print(f"Overall frequency: {df.claim_count.sum() / df.exposure.sum():.4f} claims per policy-year")

train, test = train_test_split(df, test_size=0.25, random_state=0)

# ---------------------------------------------------------------
# 2. Fit the Poisson GLM with log(exposure) as an offset
#    Reference levels: age 41-60, Suburban (the largest/most typical cells)
# ---------------------------------------------------------------
formula = ("claim_count ~ C(age_band, Treatment('41-60'))"
           " + C(region, Treatment('Suburban'))"
           " + vehicle_age + prior_claims")
glm = smf.glm(formula, data=train,
              family=sm.families.Poisson(),
              offset=np.log(train["exposure"])).fit()
print("\n", glm.summary())

# ---------------------------------------------------------------
# 3. Relativities: exp(coefficient) = multiplicative rating factor
# ---------------------------------------------------------------
rel = pd.DataFrame({
    "relativity": np.exp(glm.params),
    "ci_low": np.exp(glm.conf_int()[0]),
    "ci_high": np.exp(glm.conf_int()[1]),
    "p_value": glm.pvalues,
}).round(3)
print("\nRating relativities (Intercept = base frequency)")
print(rel)

# ---------------------------------------------------------------
# 4. Overdispersion check: Pearson chi2 / df should be near 1
# ---------------------------------------------------------------
print(f"\nPearson chi2 / df = {glm.pearson_chi2 / glm.df_resid:.3f}")

# ---------------------------------------------------------------
# 5. Out-of-sample evaluation
# ---------------------------------------------------------------
test = test.copy()
test["pred_count"] = glm.predict(test, offset=np.log(test["exposure"]))
test["pred_freq"] = test["pred_count"] / test["exposure"]

null_freq = train.claim_count.sum() / train.exposure.sum()
dev_model = mean_poisson_deviance(test.claim_count / test.exposure, test.pred_freq,
                                  sample_weight=test.exposure)
dev_null = mean_poisson_deviance(test.claim_count / test.exposure,
                                 np.full(len(test), null_freq),
                                 sample_weight=test.exposure)
print(f"\nTest Poisson deviance  null: {dev_null:.4f}   GLM: {dev_model:.4f}")
print(f"Deviance explained: {1 - dev_model / dev_null:.1%}")

# Actual vs expected by predicted-risk decile (a "lift" table)
test["decile"] = pd.qcut(test.pred_freq, 10, labels=False) + 1
lift = test.groupby("decile").agg(exposure=("exposure", "sum"),
                                  actual=("claim_count", "sum"),
                                  expected=("pred_count", "sum"))
lift["actual_freq"] = lift.actual / lift.exposure
lift["expected_freq"] = lift.expected / lift.exposure
lift["A/E"] = lift.actual / lift.expected
print("\nLift table (test set)")
print(lift.round(3))

# ---------------------------------------------------------------
# 6. Price a new policy
# ---------------------------------------------------------------
new = pd.DataFrame({"age_band": ["18-25"], "region": ["Urban"],
                    "vehicle_age": [3], "prior_claims": [1], "exposure": [1.0]})
lam = glm.predict(new, offset=np.log(new["exposure"])).iloc[0]
avg_severity = 4_500   # would come from a separate severity GLM (Gamma)
print(f"\nNew policy: 22-yr-old, Urban, 3-yr-old car, 1 prior claim")
print(f"  Expected frequency: {lam:.4f} claims/yr")
print(f"  Pure premium at ${avg_severity:,} severity: ${lam * avg_severity:,.2f}")
