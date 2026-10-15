"""
Part 10: Loss reserving. Chain ladder, its GLM equivalent (ODP), a bootstrap reserve
distribution, a GLM with a calendar-year trend, and an individual-claim GBM.
Requires: numpy, pandas, statsmodels, scikit-learn, lightgbm.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lightgbm as lgb
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(8)
N = 10                                                        # accident years x dev years
pd.set_option("display.width", 140)

# ---------------------------------------------------------------
# 1. Simulate a paid-loss triangle ($000s), with a recent inflation spike
# ---------------------------------------------------------------
pattern = np.array([0.25, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])
exposure = 1000 * 1.03 ** np.arange(N)                       # book grows 3%/yr
base_ult = 10.0 * exposure                                    # $10k of losses per exposure unit, at CY0 cost level
cy_rate = lambda k: 0.03 if k < 7 else 0.10                   # claims inflation: 3%, then 10% from CY7
infl = np.cumprod([1.0] + [1 + cy_rate(k) for k in range(1, 2 * N)])   # cost index by calendar year
PHI = 40                                                      # ODP dispersion

mean_inc = np.array([[base_ult[i] * pattern[j] * infl[i + j] for j in range(N)] for i in range(N)])
inc = PHI * rng.poisson(mean_inc / PHI).astype(float)         # over-dispersed Poisson draws
observed = np.add.outer(np.arange(N), np.arange(N)) < N       # upper-left triangle (i + j <= 9)
tri_inc = np.where(observed, inc, np.nan)
tri_cum = np.cumsum(np.nan_to_num(tri_inc), axis=1)
tri_cum[~observed] = np.nan

true_reserve_by_ay = np.where(~observed, mean_inc, 0).sum(axis=1)
print("Cumulative paid triangle ($000s)")
print(pd.DataFrame(tri_cum, index=[f"AY{i}" for i in range(N)], columns=[f"d{j+1}" for j in range(N)])
      .round(0).fillna("").to_string())

# ---------------------------------------------------------------
# 2. Chain ladder
# ---------------------------------------------------------------
def chain_ladder(cum):
    n = cum.shape[0]
    ldf = np.array([np.nansum(cum[: n - j - 1, j + 1]) / np.nansum(cum[: n - j - 1, j]) for j in range(n - 1)])
    latest = np.array([cum[i, n - 1 - i] for i in range(n)])
    cdf = np.append(np.cumprod(ldf[::-1])[::-1], 1.0)        # cumulative dev factor to ultimate
    ult = latest * cdf[::-1]
    return ldf, latest, ult, ult - latest

ldf, latest, ult, reserve_cl = chain_ladder(tri_cum)
print("\nAge-to-age factors:", np.round(ldf, 3))

# ---------------------------------------------------------------
# 3. ODP GLM: Poisson (quasi) GLM on incrementals with AY + dev factors == chain ladder
# ---------------------------------------------------------------
cells = pd.DataFrame([(i, j, inc[i, j], observed[i, j]) for i in range(N) for j in range(N)],
                     columns=["ay", "dev", "inc", "obs"])
cells["cy"] = cells.ay + cells.dev
cells["log_exp"] = np.log(exposure[cells.ay])
train = cells[cells.obs]
future = cells[~cells.obs].copy()

odp = smf.glm("inc ~ C(ay) + C(dev)", data=train, family=sm.families.Poisson()).fit(scale="X2")
future["odp"] = odp.predict(future)
reserve_odp = future.groupby("ay").odp.sum().reindex(range(N), fill_value=0).values
print(f"ODP dispersion estimate: {odp.scale:.1f} (truth {PHI})")

# ---------------------------------------------------------------
# 4. GLM with exposure offset + dev factors + piecewise calendar-year trend
#    (captures the inflation change that chain ladder averages away)
# ---------------------------------------------------------------
train = train.assign(cy_post=np.maximum(train.cy - 6, 0))
future["cy_post"] = np.maximum(future.cy - 6, 0)
cyglm = smf.glm("inc ~ C(dev) + cy + cy_post", data=train, family=sm.families.Poisson(),
                offset=train.log_exp).fit(scale="X2")
print(f"Calendar trend GLM: pre-CY7 trend {np.exp(cyglm.params['cy']) - 1:+.1%}/yr, "
      f"post {np.exp(cyglm.params['cy'] + cyglm.params['cy_post']) - 1:+.1%}/yr (truth +3% then +10%)")
future["cyglm"] = cyglm.predict(future, offset=future.log_exp)
reserve_cy = future.groupby("ay").cyglm.sum().reindex(range(N), fill_value=0).values

# The fitted post-CY7 trend rests on only 3 diagonals. Actuaries usually SELECT the future
# trend with judgment; here we keep the fitted levels but project the future at a selected +10%.
last_obs_cy = N - 1
mu_last = np.exp(cyglm.params["Intercept"]
                 + np.array([cyglm.params.get(f"C(dev)[T.{j}]", 0.0) for j in future.dev])
                 + cyglm.params["cy"] * last_obs_cy + cyglm.params["cy_post"] * (last_obs_cy - 6)
                 + future.log_exp)
future["cyglm_sel"] = mu_last * 1.10 ** (future.cy - last_obs_cy)
reserve_cy_sel = future.groupby("ay").cyglm_sel.sum().reindex(range(N), fill_value=0).values

summary = pd.DataFrame({"latest_paid": latest, "CL_reserve": reserve_cl, "ODP_GLM": reserve_odp,
                        "CY_trend_GLM": reserve_cy,
                        "CY_GLM_sel10": reserve_cy_sel, "true_reserve": true_reserve_by_ay},
                       index=[f"AY{i}" for i in range(N)])
summary.loc["Total"] = summary.sum()
print("\nReserves by accident year ($000s)")
print(summary.round(0).to_string())

# ---------------------------------------------------------------
# 5. Bootstrap ODP: distribution of the chain-ladder reserve
# ---------------------------------------------------------------
fit_inc = np.full((N, N), np.nan)
fit_inc[observed] = odp.predict(cells[cells.obs])
resid = (tri_inc[observed] - fit_inc[observed]) / np.sqrt(fit_inc[observed])
n_obs, n_par = observed.sum(), 2 * N - 1
resid *= np.sqrt(n_obs / (n_obs - n_par))                    # degrees-of-freedom adjustment
sims = []
for _ in range(2000):
    r_star = rng.choice(resid, size=n_obs, replace=True)
    pseudo = np.full((N, N), np.nan)
    pseudo[observed] = fit_inc[observed] + r_star * np.sqrt(fit_inc[observed])
    pc = np.cumsum(np.nan_to_num(pseudo), axis=1); pc[~observed] = np.nan
    ldf_b, latest_b, _, _ = chain_ladder(pc)
    # project future incrementals from the ACTUAL latest diagonal, then add process noise
    cum = np.array([tri_cum[i, N - 1 - i] for i in range(N)])
    total = 0.0
    for i in range(1, N):
        c = cum[i]
        for j in range(N - i, N):
            m = max(c * (ldf_b[j - 1] - 1), 1e-6)
            total += rng.gamma(m / odp.scale, odp.scale)   # mean m, variance phi*m
            c = c * ldf_b[j - 1]
    sims.append(total)
sims = np.array(sims)
print("\nBootstrap ODP reserve distribution ($000s)")
print(f"  mean {sims.mean():,.0f}   sd {sims.std():,.0f}   CV {sims.std() / sims.mean():.1%}")
print(f"  75th pct {np.percentile(sims, 75):,.0f}   95th {np.percentile(sims, 95):,.0f}   "
      f"99.5th {np.percentile(sims, 99.5):,.0f}")
print(f"  True expected reserve {true_reserve_by_ay.sum():,.0f} sits at the "
      f"{np.mean(sims < true_reserve_by_ay.sum()):.0%} percentile of the CL distribution")

# ---------------------------------------------------------------
# 6. Individual-claim ML: predict ultimate from claim features at 12 months
# ---------------------------------------------------------------
m = 15_000
claims = pd.DataFrame({
    "injury": rng.choice(["property_only", "minor", "moderate", "severe"], m, p=[0.5, 0.3, 0.15, 0.05]),
    "attorney": rng.choice([0, 1], m, p=[0.8, 0.2]),
    "claimant_age": rng.integers(16, 85, m),
    "paid_12m": np.round(rng.lognormal(7.5, 0.8, m)),
})
sev = claims.injury.map({"property_only": 1.05, "minor": 1.6, "moderate": 2.8, "severe": 6.0})
dev_mult = sev * (1 + 1.2 * claims.attorney) * (1 + 0.01 * np.maximum(claims.claimant_age - 50, 0))
claims["case_reserve_12m"] = np.round(claims.paid_12m * (dev_mult - 1) * rng.lognormal(0, 0.5, m))  # noisy adjuster view
claims["ultimate"] = np.round(claims.paid_12m * dev_mult * rng.lognormal(0, 0.25, m))
claims["injury"] = claims.injury.astype("category")
claims["incurred_12m"] = claims.paid_12m + claims.case_reserve_12m

closed, open_ = train_test_split(claims, test_size=0.3, random_state=0)   # pretend: closed = history
feats = ["injury", "attorney", "claimant_age", "paid_12m", "case_reserve_12m"]
# Target = development ratio (ultimate / paid at 12m): scale-free, easier for trees than raw $
gbm = lgb.LGBMRegressor(objective="gamma", n_estimators=800, learning_rate=0.03, num_leaves=15,
                        min_child_samples=50, verbose=-1).fit(closed[feats], closed.ultimate / closed.paid_12m)
agg_factor = closed.ultimate.sum() / closed.incurred_12m.sum()            # one incurred LDF for all
pred_agg = open_.incurred_12m * agg_factor
pred_gbm = gbm.predict(open_[feats]) * open_.paid_12m
print("\nIndividual-claim reserving on 4,500 hold-out claims")
print(f"  Actual total ultimate:          {open_.ultimate.sum():>14,.0f}")
print(f"  Aggregate incurred LDF ({agg_factor:.3f}): {pred_agg.sum():>14,.0f}   "
      f"claim-level MAE {np.mean(np.abs(pred_agg - open_.ultimate)):,.0f}")
print(f"  LightGBM (gamma loss):          {pred_gbm.sum():>14,.0f}   "
      f"claim-level MAE {np.mean(np.abs(pred_gbm - open_.ultimate)):,.0f}")
seg = open_.assign(agg_pred=pred_agg, gbm_pred=pred_gbm).groupby(["injury", "attorney"], observed=True)[
    ["ultimate", "agg_pred", "gbm_pred"]].sum()
seg["agg_A/E"] = seg.ultimate / seg.agg_pred
seg["gbm_A/E"] = seg.ultimate / seg.gbm_pred
print("\nA/E by segment (actual ultimate / predicted)")
print(seg[["agg_A/E", "gbm_A/E"]].round(2))
