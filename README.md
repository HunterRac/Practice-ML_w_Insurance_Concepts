# ML for P&C Insurance: Index

Each part has a write-up (definition → worked example → Python) and a runnable script.
Install: `pip install -r requirements.txt`
Run any script from this folder, e.g. `python 04_gbm_frequency.py`. Parts 2–4 and 11 import `insurance_data.py`.

| # | Topic | Use case | Write-up | Script |
|---|---|---|---|---|
| 1 | Poisson GLM | Claim frequency, rating relativities | [md](01_glm_claim_frequency.md) | [py](01_glm_claim_frequency.py) |
| 2 | Gamma GLM | Claim severity, pure premium | [md](02_gamma_glm_severity.md) | [py](02_gamma_glm_severity.py) |
| 3 | Tweedie GLM | Pure premium in one model | [md](03_tweedie_glm.md) | [py](03_tweedie_glm.py) |
| 4 | Gradient boosting (LightGBM) | Frequency with interactions, vs GLM | [md](04_gbm_frequency.md) | [py](04_gbm_frequency.py) |
| 5 | Random forest | Litigation triage at FNOL | [md](05_random_forest_claim_triage.md) | [py](05_random_forest_claim_triage.py) |
| 6 | Logistic regression vs GBM classifier | Fraud / SIU referral | [md](06_fraud_classifier.md) | [py](06_fraud_classifier.py) |
| 7 | K-means clustering | Rating territories | [md](07_kmeans_territories.md) | [py](07_kmeans_territories.py) |
| 8 | Isolation Forest / LOF | Unsupervised fraud screening | [md](08_anomaly_detection_fraud.md) | [py](08_anomaly_detection_fraud.py) |
| 9 | Kaplan-Meier / Cox PH | Policy retention | [md](09_survival_retention.md) | [py](09_survival_retention.py) |
| 10 | Chain ladder, ODP GLM, bootstrap, claim-level GBM | Loss reserving | [md](10_reserving.md) | [py](10_reserving.py) |
| 11 | SHAP | Explaining GBMs for filings | [md](11_explainability_shap.md) | [py](11_explainability_shap.py) |

All data is synthetic with a known "truth", so every write-up compares what the model recovered against the real answer.
