"""
Part 7: K-means clustering to build rating territories from zip-level experience.
Requires: numpy, pandas, scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_poisson_deviance

rng = np.random.default_rng(3)

# ---------------------------------------------------------------
# 1. 600 zip codes on a map with a smooth true risk surface
# ---------------------------------------------------------------
nz = 600
z = pd.DataFrame({"lat": rng.uniform(0, 100, nz), "lon": rng.uniform(0, 100, nz)})
d_city = np.hypot(z.lat - 35, z.lon - 40)             # distance to the city center
d_coast = z.lon                                        # coast on the west edge (lon=0)
z["true_rel"] = (np.exp(0.7 * np.exp(-d_city / 20))   # urban core: up to 2x
                 * np.exp(0.35 * np.exp(-d_coast / 10))  # coastal strip
                 * np.exp(rng.normal(0, 0.05, nz)))     # small local variation
z["true_rel"] /= np.average(z.true_rel)
# Exposure is highly uneven: dense near the city, sparse rurally
z["exposure_yr"] = np.round(rng.lognormal(np.log(150), 1.0, nz) * (1 + 3 * np.exp(-d_city / 25)))

BASE_FREQ = 0.09
def simulate_year(seed):
    r = np.random.default_rng(seed)
    return r.poisson(BASE_FREQ * z.true_rel * z.exposure_yr)

z["claims_y1"] = simulate_year(1)        # experience used to build territories
z["claims_y2"] = simulate_year(2)        # next year: out-of-time test
print(f"Zips: {nz}, total exposure/yr: {z.exposure_yr.sum():,.0f}, "
      f"median zip exposure: {z.exposure_yr.median():.0f}, median zip claims: {z.claims_y1.median():.0f}")

# ---------------------------------------------------------------
# 2. Raw and credibility-weighted zip relativities from year 1
#    (In practice these come from GLM residuals: actual / GLM-expected,
#     after removing driver and vehicle effects.)
# ---------------------------------------------------------------
overall = z.claims_y1.sum() / z.exposure_yr.sum()
z["raw_rel"] = (z.claims_y1 / z.exposure_yr) / overall
K = 120                                     # credibility constant, in expected claims
expected = overall * z.exposure_yr
z["Z"] = expected / (expected + K)
z["cred_rel"] = z.Z * z.raw_rel + (1 - z.Z) * 1.0

# ---------------------------------------------------------------
# 3. K-means on location + credibility relativity
#    Weighting the relativity column trades spatial contiguity vs risk homogeneity
# ---------------------------------------------------------------
feat = np.column_stack([z.lat, z.lon, np.log(z.cred_rel)])
Xs = StandardScaler().fit_transform(feat)
Xs[:, 2] *= 2.0                             # emphasize risk over geography

print("\n k   inertia   silhouette")
for k in range(3, 13):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs, sample_weight=z.exposure_yr)
    print(f"{k:2d}  {km.inertia_:9.0f}   {silhouette_score(Xs, km.labels_):.3f}")

K_TERR = 8
km = KMeans(n_clusters=K_TERR, n_init=20, random_state=0).fit(Xs, sample_weight=z.exposure_yr)
z["territory"] = km.labels_

terr = z.groupby("territory").agg(zips=("lat", "size"), exposure=("exposure_yr", "sum"),
                                  claims=("claims_y1", "sum"), true_rel=("true_rel", "mean"),
                                  lat=("lat", "mean"), lon=("lon", "mean"))
terr["terr_rel"] = (terr.claims / terr.exposure) / overall
terr = terr.sort_values("terr_rel")
# Renumber territories 1..K from lowest to highest risk, like a rate manual
rank = {t: i + 1 for i, t in enumerate(terr.index)}
z["territory"] = z.territory.map(rank)
terr.index = terr.index.map(rank)
print(f"\nTerritory table (K={K_TERR}, ranked low to high risk)")
print(terr.round(2))
z["terr_rel"] = z.territory.map(terr.terr_rel)

# ---------------------------------------------------------------
# 4. Out-of-time test: which relativity predicts year-2 claims best?
# ---------------------------------------------------------------
y2_freq = z.claims_y2 / z.exposure_yr
print("\nYear-2 Poisson deviance (exposure-weighted, lower = better)")
flat_dev = None
for name, rel in [("No territory (flat)", np.ones(nz)),
                  ("Raw zip relativity", z.raw_rel.clip(0.05)),
                  ("Credibility-weighted zip", z.cred_rel),
                  (f"K-means territories (K={K_TERR})", z.terr_rel),
                  ("True relativity (oracle)", z.true_rel)]:
    dev = mean_poisson_deviance(y2_freq, overall * rel, sample_weight=z.exposure_yr)
    flat_dev = flat_dev or dev
    print(f"  {name:32s} {dev:.5f}   ({1 - dev / flat_dev:+.0%} deviance reduction vs flat)")

# ---------------------------------------------------------------
# 5. Text map: territory number on a 10x10 grid (mode per cell)
# ---------------------------------------------------------------
z["gx"], z["gy"] = (z.lon // 10).astype(int), (z.lat // 10).astype(int)
grid = z.groupby(["gy", "gx"]).territory.agg(lambda s: s.mode().iloc[0]).unstack()
print("\nTerritory map (rows = lat bands from north to south, cols = lon from west/coast to east)")
print(grid.sort_index(ascending=False).fillna(-1).astype(int).replace(-1, ".").to_string())
