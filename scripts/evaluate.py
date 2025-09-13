from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
import joblib
from sales_forecast.features import basic_clean, select_features

DATA = Path("data/examples/synthetic_pipeline.csv")
OUTDIR = Path("artifacts/public")

def main():
    df = pd.read_csv(DATA)
    df = basic_clean(df)
    X = select_features(df, [])

    clf = joblib.load(OUTDIR / "rf_classifier.pkl")
    reg = joblib.load(OUTDIR / "rf_regressor.pkl")

    auc = roc_auc_score((X["prob"]>0.5).astype(int), clf.predict_proba(X)[:,1])
    mae = mean_absolute_error(60 - 50*X["prob"], reg.predict(X))
    print(f"AUC (rough): {auc:.3f}  |  MAE days-to-close (rough): {mae:.2f}")

if __name__ == "__main__":
    main()
