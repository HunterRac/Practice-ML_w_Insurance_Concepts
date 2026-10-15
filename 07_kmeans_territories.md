# Part 7: K-Means Clustering for Rating Territories

Script: [`07_kmeans_territories.py`](/mnt/project-files/ml-insurance/07_kmeans_territories.py)

## 1. Definition

**K-means** partitions $n$ points into $K$ clusters by minimizing within-cluster squared distance:

$$\min_{C_1..C_K}\sum_{k=1}^{K}\sum_{i\in C_k} w_i\,\lVert x_i-\mu_k\rVert^2$$

The algorithm alternates two steps: assign each point to the nearest centroid, then recompute centroids. Run it from several random starts (`n_init`) and keep the best. `sample_weight` lets heavy points (high-exposure zips) pull the centroids.

Choose $K$ with the **elbow** in inertia, **silhouette** score, and, most importantly for insurance, **business constraints and out-of-sample predictive power**.

**Why cluster for territories?** Rating by individual zip code fails: most zips have too few claims, so their raw loss experience is mostly noise. The standard actuarial approach:
1. Fit the GLM without territory and get each zip's **residual relativity** (actual / expected).
2. **Credibility-weight** it toward 1.0: $\hat r = Z\cdot r_{\text{raw}} + (1-Z)\cdot 1$, with $Z = \frac{E}{E+k}$ (E = expected claims).
3. **Cluster** zips on location and smoothed relativity into a manageable number of territories (5–20).
4. Put `territory` back into the GLM as a categorical variable.

Other insurance clustering uses: customer segmentation for marketing or retention, grouping vehicle models into symbol groups, and grouping claims by adjuster workflow.

## 2. Worked example: building 8 territories

**Data:** 600 zip codes on a map. The true frequency relativity has an **urban core** (up to 2×) and a **coastal strip** on the west edge. Exposure is very uneven (median 243 car-years and only 22 claims per zip). Year 1 builds the territories; year 2 is the **out-of-time test**.

**Out-of-time results (year-2 Poisson deviance):**
| Zip relativity used | Deviance reduction vs flat |
|---|---|
| Raw zip experience | **−2%** (worse than no territory at all) |
| Credibility-weighted zip | +27% |
| **K-means, 8 territories** | **+26%** |
| True relativity (oracle) | +47% |

- **Raw zip rates are worse than nothing**: they price last year's noise.
- Credibility weighting fixes most of it.
- 8 territories keep almost all of the credibility-weighted benefit with **8 parameters instead of 600**. That makes them stable, fileable, and easy for agents to understand.

**Territory table** (ranked low to high risk, like a rate manual):
| Territory | Zips | Exposure | Fitted rel | True rel (avg) | Centroid |
|---|---|---|---|---|---|
| 1 | 13 | 19.6k | 0.76 | 0.90 | inland north |
| 2 | 65 | 35.8k | 0.78 | 0.90 | far east |
| 3 | 118 | 38.5k | 0.88 | 0.98 | northwest |
| 4 | 153 | 50.8k | 0.97 | 0.91 | northeast |
| 5 | 136 | 51.4k | 1.02 | 1.04 | south |
| 6 | 86 | 48.6k | 1.16 | 1.12 | near city / coast |
| 7 | 24 | 19.2k | 1.32 | 1.26 | city ring |
| **8** | **5** | 7.8k | **1.48** | **1.47** | **city center (35, 40)** |

The top territories sit right at the true city center. Territory 1 at 0.76 vs a true 0.90 shows a **winner's curse**: clustering on the experience itself picks up favorable noise. That is why you validate out of time and often temper the extremes.

**Choosing K:** silhouette was fairly flat (0.24–0.31 across K = 3–12), which is typical, since risk varies smoothly and there are no natural clumps. In territory work K is usually set by business needs and validated out of sample, not by silhouette.

**Geography vs risk weight.** The script scales the risk column by 2× after standardizing. A higher weight gives more homogeneous risk but scattered "territories"; a lower weight gives contiguous but mixed-risk ones. Many carriers require contiguity, which is where spatially constrained methods come in (below).

## 3. Python code (core)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1-2. Credibility-weighted zip relativity (here vs overall mean; in practice vs GLM expected)
expected = overall_freq * z.exposure
z["Z"] = expected / (expected + 120)                  # k = 120 expected claims
z["cred_rel"] = z.Z * z.raw_rel + (1 - z.Z) * 1.0

# 3. Cluster on location + log relativity
X = StandardScaler().fit_transform(np.column_stack([z.lat, z.lon, np.log(z.cred_rel)]))
X[:, 2] *= 2.0                                        # emphasize risk vs geography
km = KMeans(n_clusters=8, n_init=20, random_state=0).fit(X, sample_weight=z.exposure)
z["territory"] = km.labels_

# 4. Re-estimate the territory relativity (or add C(territory) to the GLM)
terr_rel = z.groupby("territory").apply(lambda d: d.claims.sum() / d.exposure.sum()) / overall_freq
```

## 4. When to use it, and when not to

**Use K-means when:** you need a small number of groups from many noisy units (zips, vehicle models, occupations) and want something simple and explainable.

**Limitations and alternatives:**
- **Assumes round, similar-size clusters** in Euclidean space and is sensitive to feature scaling. Try Gaussian mixtures for elliptical clusters, or DBSCAN/HDBSCAN when there's noise and odd shapes.
- **Not spatially contiguous.** For contiguous territories, use agglomerative clustering with a neighbor connectivity matrix (`AgglomerativeClustering(connectivity=...)`), or SKATER-style regionalization (`spopt` package).
- **Smoothing instead of clustering:** spatial smoothing of residuals (kriging, or a GAM/GBM on lat/lon) can then be banded into territories.
- **Re-validate yearly**: territory boundaries are filed and sticky, but risk moves (new development, weather patterns).
