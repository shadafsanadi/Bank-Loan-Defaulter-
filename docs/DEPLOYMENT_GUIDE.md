# Deployment Guide

**Project:** Home Loan Default Prediction  
**Last Updated:** 2026-06-04

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Conda or virtualenv
- ~8 GB RAM (for loading all 7 tables)
- All 7 CSV files in data/raw/

### Setup

```bash
# Clone repo
git clone https://github.com/your-username/HomeLoanDefault.git
cd HomeLoanDefault

# Create environment
conda create -n homeloan python=3.11 -y
conda activate homeloan

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Option 1: Make
make train

# Option 2: Direct
python src/models/train.py
```

This will:
1. Load all 7 CSV tables from data/raw/
2. Build feature aggregations for all supplementary tables
3. Join all features into a master DataFrame
4. Run 5-fold cross-validation for model comparison
5. Train the final model on the full training set
6. Save artifacts to models/v2/

Training time: ~15–30 minutes depending on hardware (histogram XGBoost on 307K rows with ~300 features).

### Running the Streamlit App

```bash
# Option 1: Make
make app

# Option 2: Direct
streamlit run app/app.py
```

App will be available at: http://localhost:8501

### Running the FastAPI Service

```bash
# Option 1: Make
make api

# Option 2: Direct
uvicorn api.main:app --reload --port 8000
```

API docs at: http://localhost:8000/docs  
Health check: http://localhost:8000/health

---

## Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Start all services (API + Streamlit)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Services

| Service | Port | URL |
|---|---|---|
| Streamlit app | 8501 | http://localhost:8501 |
| FastAPI service | 8000 | http://localhost:8000 |
| API docs | 8000 | http://localhost:8000/docs |

---

## API Reference

### POST /predict

Predict default risk for a loan applicant.

**Request body:**
```json
{
  "amt_income_total": 150000,
  "amt_credit": 500000,
  "amt_annuity": 25000,
  "days_birth": -12000,
  "days_employed": -1825,
  "ext_source_2": 0.65,
  "ext_source_3": 0.52,
  "code_gender": "F",
  "name_income_type": "Working",
  "name_education_type": "Higher education",
  "name_family_status": "Married",
  "name_housing_type": "House / apartment"
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.073,
  "risk_band": "Low",
  "top_risk_factors": [
    {"feature": "EXT_SOURCE_2", "impact": -0.142},
    {"feature": "CREDIT_INCOME_RATIO", "impact": 0.089},
    {"feature": "AGE_YEARS", "impact": -0.061}
  ]
}
```

### GET /health

```json
{
  "status": "ok",
  "model_version": "v2",
  "model_roc_auc": 0.804,
  "trained_at": "2026-06-04T10:30:00"
}
```

---

## Environment Variables

All environment variables are optional — defaults are set in configs/model_config.yaml.

| Variable | Default | Description |
|---|---|---|
| MODEL_VERSION | v2 | Which model version to load (models/{MODEL_VERSION}/) |
| DATA_DIR | data/raw | Path to raw CSV files |
| LOG_LEVEL | INFO | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| API_PORT | 8000 | FastAPI server port |
| APP_PORT | 8501 | Streamlit app port |

---

## Makefile Commands

```bash
make train      # Train model with full pipeline
make app        # Launch Streamlit dashboard
make api        # Launch FastAPI service
make test       # Run pytest test suite
make lint       # Run flake8 linting
make clean      # Remove compiled Python files
make docker-up  # Start all Docker services
make docker-down # Stop all Docker services
```

---

## Production Deployment Checklist

Before deploying to production:

- [ ] Model trained with all 7 tables (not just application_train.csv)
- [ ] 5-fold CV AUC > 0.78 documented in models/v2/metadata.json
- [ ] Test set AUC evaluated exactly once and documented
- [ ] Threshold documented and justified (not default 0.5)
- [ ] All tests passing (`make test`)
- [ ] Docker images built and tested
- [ ] Health endpoint returning correct model version
- [ ] SHAP explanations rendering in app
- [ ] Input validation catching out-of-range values
- [ ] Logging to file configured
- [ ] Model artifact backed up (not just local files)

---

## Cloud Deployment Options

### Streamlit Cloud (Simplest — Portfolio Demo)

1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect GitHub repo
4. Set main file: app/app.py
5. Note: Large CSV files cannot be uploaded — use pre-trained model artifacts only

### Hugging Face Spaces

1. Create Space with Streamlit SDK
2. Upload code and model artifacts
3. Suitable for demos with pre-trained model

### AWS / GCP / Azure (Production)

1. Build Docker image → push to container registry
2. Deploy FastAPI container to ECS/Cloud Run/AKS
3. Deploy Streamlit as separate container or serve via reverse proxy
4. Store model artifacts in S3/GCS/Blob Storage
5. Set up CloudWatch/Stackdriver for monitoring

---

## Model Versioning

Model artifacts are stored in versioned directories:

```
models/
├── v1/              # MVP artifacts (XGBoost, main table only)
│   ├── xgb_model.pkl
│   ├── model_features.pkl
│   └── scaler.pkl
└── v2/              # Full pipeline (all tables, sklearn Pipeline)
    ├── pipeline.pkl         # preprocessor + model, single serializable object
    ├── feature_names.pkl    # ordered list of features
    ├── shap_explainer.pkl   # pre-computed TreeExplainer
    └── metadata.json        # AUC, threshold, training date, feature count
```

When retraining, always create a new version directory rather than overwriting existing artifacts. Roll back by changing MODEL_VERSION environment variable.
