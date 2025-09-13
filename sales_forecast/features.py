import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Total Contract Value (converted)", "Probability (%)"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["Created Date", "Close Date"]:
        if col not in out.columns:
            out[col] = pd.NaT
    return out

def select_features(df: pd.DataFrame, cols):
    X = pd.DataFrame(index=df.index)
    if "Total Contract Value (converted)" in df.columns:
        X["tcv"] = pd.to_numeric(df["Total Contract Value (converted)"], errors="coerce").fillna(0)
    else:
        X["tcv"] = 0.0
    if "Probability (%)" in df.columns:
        X["prob"] = pd.to_numeric(df["Probability (%)"], errors="coerce").fillna(0) / 100.0
    else:
        X["prob"] = 0.0
    cd = pd.to_datetime(df.get("Created Date"), errors="coerce", dayfirst=True)
    cl = pd.to_datetime(df.get("Close Date"), errors="coerce", dayfirst=True)
    X["days_span"] = (cl - cd).dt.days.fillna(0)
    return X.fillna(0)
