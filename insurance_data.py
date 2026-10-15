"""
Shared synthetic personal auto book used by Parts 2-6 and 11.
The "true" risk structure is known so you can check what each model recovers.
"""
import numpy as np
import pandas as pd

AGE_REL = {"18-25": 1.9, "26-40": 1.1, "41-60": 1.0, "61+": 1.25}
REGION_FREQ_REL = {"Urban": 1.35, "Suburban": 1.0, "Rural": 0.8}
REGION_SEV_REL = {"Urban": 1.15, "Suburban": 1.0, "Rural": 0.9}


def make_auto_book(n=100_000, seed=42):
    """Return (policies, claims).

    policies: one row per policy, with claim_count and total_loss
    claims:   one row per claim, with claim_amount (for severity models)

    Truth:
      frequency = 0.08 x age x region x exp(-0.02*veh_age) x exp(0.25*prior)
                  x credit curve (non-linear) x 1.3 if (18-25 AND Urban)  <- interaction
      severity  ~ Gamma(shape=2), mean = 4,500 x region x exp(-0.03*veh_age)
                  x (vehicle_value / 25k)^0.5
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "driver_age": rng.integers(18, 85, n),
        "vehicle_age": rng.integers(0, 20, n),
        "vehicle_value": np.round(rng.lognormal(np.log(25_000), 0.45, n), -2),
        "region": rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.4, 0.4, 0.2]),
        "prior_claims": rng.poisson(0.3, n).clip(0, 3),
        "credit_score": rng.normal(700, 60, n).clip(450, 850).round(),
        "exposure": rng.uniform(0.1, 1.0, n),
    })
    df["age_band"] = pd.cut(df["driver_age"], [17, 25, 40, 60, 120],
                            labels=["18-25", "26-40", "41-60", "61+"]).astype(str)

    # Non-linear credit effect: flat above 700, rising steeply below
    credit_rel = np.exp(0.006 * np.clip(700 - df["credit_score"], 0, None))
    interaction = np.where((df["age_band"] == "18-25") & (df["region"] == "Urban"), 1.3, 1.0)
    df["true_freq"] = (0.08
                       * df["age_band"].map(AGE_REL)
                       * df["region"].map(REGION_FREQ_REL)
                       * np.exp(-0.02 * df["vehicle_age"])
                       * np.exp(0.25 * df["prior_claims"])
                       * credit_rel * interaction)
    df["claim_count"] = rng.poisson(df["true_freq"] * df["exposure"])

    df["true_sev"] = (4_500
                      * df["region"].map(REGION_SEV_REL)
                      * np.exp(-0.03 * df["vehicle_age"])
                      * (df["vehicle_value"] / 25_000) ** 0.5)

    # Explode to claim level; Gamma with shape k has mean = k * scale
    shape = 2.0
    idx = np.repeat(df.index.values, df["claim_count"].values)
    claims = df.loc[idx].reset_index().rename(columns={"index": "policy_id"})
    claims["claim_amount"] = rng.gamma(shape, claims["true_sev"] / shape).round(2)

    df["total_loss"] = claims.groupby("policy_id")["claim_amount"].sum().reindex(df.index).fillna(0.0)
    df.index.name = "policy_id"
    return df.reset_index(), claims


FEATURES = ["driver_age", "vehicle_age", "vehicle_value", "region",
            "prior_claims", "credit_score"]
