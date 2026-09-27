# Tourism package purchase propensity - MLOps project

Pipeline: `.github/workflows/pipeline.yml` (GitHub Actions)

| Stage | Script | Output |
|---|---|---|
| Data registration | `model_building/data_register.py` | validation summary (workflow artifact) |
| Data preparation | `model_building/prep.py` | train/test splits (workflow artifact) |
| Training + MLflow tracking | `model_building/train.py` | `deployment/model.joblib` (committed by the workflow) |
| App | `deployment/app.py` | Streamlit Community Cloud |

The model is a historical purchase-propensity model trained on past tourism package
purchases. It uses only pre-contact customer attributes and has not been validated on
Wellness Tourism Package outcomes.
