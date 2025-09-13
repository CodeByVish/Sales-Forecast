from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from sales_forecast.features import basic_clean, select_features

DATA = Path("data/examples/synthetic_pipeline.csv")
OUTDIR = Path("artifacts/public")
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    df = basic_clean(df)
    X = select_features(df, [])
    # Synthetic targets (for demo)
    y_clf = (X["prob"]*0.6 + (X["tcv"]>X["tcv"].median())*0.2 + (X["days_span"]<45)*0.2 + np.random.rand(len(X))*0.05 > 0.5).astype(int)
    y_reg = (60 - 50*X["prob"] + 0.00001*X["tcv"].clip(0, 1e6) + np.random.randn(len(X))*5).clip(5, 120)

    Xtr, Xte, ytr_c, yte_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr_c)
    reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr_r, ytr_r)

    joblib.dump(clf, OUTDIR / "rf_classifier.pkl")
    joblib.dump(reg, OUTDIR / "rf_regressor.pkl")
    print("Saved models to", OUTDIR)

if __name__ == "__main__":
    main()
