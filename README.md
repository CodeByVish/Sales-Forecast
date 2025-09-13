# Sales Forecast — Upload & Predict

A Streamlit app and ML pipeline that predicts **win probability** and **days-to-close** for sales opportunities, then aggregates a **monthly expected revenue forecast**.

> **This repository contains a safe, public version**: synthetic data only, example configs, and a baseline model you can train locally. No confidential data or proprietary models are included.

## What this project demonstrates
- End‑to‑end workflow: data cleaning → feature engineering → model training (RF Classifier + RF Regressor) → inference → monthly forecast
- Production‑style app (`Streamlit`) for uploading snapshots and exporting CSV/Excel
- Clear separation of concerns (features, model bundle, training scripts)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py    # trains a baseline model to artifacts/public/
streamlit run app.py
```

Copy `config.example.yaml` to `config.yaml` and adjust paths if needed.
