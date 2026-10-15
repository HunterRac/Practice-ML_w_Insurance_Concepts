"""
Part 9: Survival analysis for policy retention (time to lapse/cancellation).
Kaplan-Meier + Cox proportional hazards, using statsmodels only.
(lifelines is the more common library: KaplanMeierFitter, CoxPHFitter. Same ideas.)
Requires: numpy, pandas, statsmodels.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

rng = np.random.default_rng(21)
n = 20_000

# ---------------------------------------------------------------
# 1. Simulate customers with a Weibull proportional-hazards lapse process
#    Time unit: months. Decreasing hazard (shape < 1): new customers churn most.
# ---------------------------------------------------------------
df = pd.DataFrame({
    "rate_change_pct": rng.normal(5, 8, n).round(1),        # last renewal rate change
    "bundled": rng.choice([0, 1], n, p=[0.55, 0.45]),        # home + auto
    "autopay": rng.choice([0, 1], n, p=[0.4, 0.6]),
    "claim_last_year": rng.choice([0, 1], n, p=[0.88, 0.12]),
    "age": rng.integers(18, 80, n),
})
TRUE_HR = {"rate_change_pct": 1.5 ** (1 / 10),   # +10 pts of rate -> 1.5x hazard
           "bundled": 0.55, "autopay": 0.75, "claim_last_year": 1.3, "age": 0.9 ** (1 / 10)}
lin = sum(np.log(TRUE_HR[c]) * df[c] for c in TRUE_HR)
shape, scale = 0.8, 70.0
u = rng.uniform(size=n)
df["true_time"] = scale * (-np.log(u) / np.exp(lin)) ** (1 / shape)   # inverse-CDF sampling

# Staggered entry: customers joined 0-48 months before the data cut -> right censoring
df["months_observed"] = rng.uniform(0, 48, n)
df["duration"] = np.minimum(df.true_time, df.months_observed)
df["lapsed"] = (df.true_time <= df.months_observed).astype(int)
print(f"Customers: {n:,}. Lapsed: {df.lapsed.mean():.1%}. Censored (still active): {1 - df.lapsed.mean():.1%}")

# ---------------------------------------------------------------
# 2. Kaplan-Meier retention curves by bundling
# ---------------------------------------------------------------
def km_at(d, months):
    sf = sm.SurvfuncRight(d.duration, d.lapsed)
    return [sf.surv_prob[np.searchsorted(sf.surv_times, m, side="right") - 1] for m in months]

months = [6, 12, 24, 36]
true_surv = lambda d, t: np.mean(d.true_time > t)
print("\nRetention (Kaplan-Meier) vs truth")
print("               " + "  ".join(f"{m:>4d}m" for m in months))
for label, d in [("All", df), ("Bundled", df[df.bundled == 1]), ("Monoline", df[df.bundled == 0])]:
    print(f"  {label:9s} KM " + "  ".join(f"{v:.3f}" for v in km_at(d, months)))
    print(f"  {'':9s} true " + "  ".join(f"{true_surv(d, m):.3f}" for m in months))

# ---------------------------------------------------------------
# 3. Why censoring matters: naive 12-month lapse rate estimates
# ---------------------------------------------------------------
true_12 = np.mean(df.true_time <= 12)
naive_all = np.mean((df.lapsed == 1) & (df.duration <= 12))         # censored counted as retained
seen12 = df[(df.months_observed >= 12) | (df.lapsed == 1)]          # drop the unobserved, keep early lapses
naive_drop = np.mean((seen12.lapsed == 1) & (seen12.duration <= 12))
km_12 = 1 - km_at(df, [12])[0]
print("\n12-month lapse rate")
print(f"  Truth:                                    {true_12:.3f}")
print(f"  Naive (treat censored as retained):       {naive_all:.3f}")
print(f"  Naive (drop short-tenure actives only):   {naive_drop:.3f}")
print(f"  Kaplan-Meier:                             {km_12:.3f}")

# ---------------------------------------------------------------
# 4. Cox proportional hazards: hazard ratios
# ---------------------------------------------------------------
cols = list(TRUE_HR)
cox = PHReg(df.duration, df[cols], status=df.lapsed, ties="efron").fit()
hr = pd.DataFrame({"HR": np.exp(cox.params), "truth": [TRUE_HR[c] for c in cols],
                   "ci_low": np.exp(cox.conf_int()[:, 0]), "ci_high": np.exp(cox.conf_int()[:, 1]),
                   "p": cox.pvalues}, index=cols)
print("\nCox hazard ratios (per unit; rate_change per 1 pt, age per 1 yr)")
print(hr.round(4))
print(f"  -> +10 pts of rate increase: HR {np.exp(10 * cox.params[0]):.2f} (truth 1.50)")

# Concordance (C-index): share of comparable pairs ranked correctly
risk = df[cols].values @ cox.params
def c_index(time, event, risk, m=4000, seed=0):
    r = np.random.default_rng(seed); i = r.integers(0, len(time), m * 50); j = r.integers(0, len(time), m * 50)
    ok = (event[i] == 1) & (time[i] < time[j])            # i lapsed before j's time
    return np.mean(risk[i][ok] > risk[j][ok])
print(f"  C-index: {c_index(df.duration.values, df.lapsed.values, risk):.3f}")

# ---------------------------------------------------------------
# 5. Predict retention curves and expected tenure for two customers
# ---------------------------------------------------------------
# Baseline cumulative hazard from Breslow estimator, then S(t|x) = exp(-H0(t) * exp(x'b))
times, cumhaz = cox.baseline_cumulative_hazard[0][0], cox.baseline_cumulative_hazard[0][1]
def surv_curve(x):
    return np.exp(-cumhaz * np.exp(np.dot(x, cox.params)))
customers = pd.DataFrame({
    "rate_change_pct": [15.0, 0.0], "bundled": [0, 1], "autopay": [0, 1],
    "claim_last_year": [1, 0], "age": [25, 55]}, index=["High-risk lapse", "Sticky"])
print("\nPredicted retention")
for name, x in customers.iterrows():
    s = surv_curve(x.values)
    at = lambda m: s[np.searchsorted(times, m, side="right") - 1]
    # Restricted mean tenure over 48 months = area under S(t)
    grid = np.linspace(0, 48, 481)
    s_grid = np.array([at(g) if g >= times[0] else 1.0 for g in grid])
    rmst = np.trapezoid(s_grid, grid)
    print(f"  {name:16s} 12m {at(12):.3f}  24m {at(24):.3f}  36m {at(36):.3f}   "
          f"expected months (of next 48): {rmst:.1f}")

# ---------------------------------------------------------------
# 6. Proportional-hazards check: does the bundled HR drift over time?
# ---------------------------------------------------------------
print("\nPH check: bundled HR estimated separately on early vs late follow-up")
early = df.assign(duration=np.minimum(df.duration, 12), lapsed=np.where(df.duration <= 12, df.lapsed, 0))
late = df[df.duration > 12].copy()     # survivors past 12m, left-truncated at 12
for name, d, entry in [("months 0-12", early, None), ("months 12+", late, np.full(len(late), 12.0))]:
    m = PHReg(d.duration, d[cols], status=d.lapsed, entry=entry).fit()
    print(f"  {name:12s} bundled HR {np.exp(m.params[1]):.3f}")
