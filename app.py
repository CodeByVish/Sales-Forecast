# app.py
from pathlib import Path
import io
import yaml
import pandas as pd
import numpy as np
import streamlit as st

from sales_forecast.features import basic_clean, select_features
from sales_forecast.model_loader import ModelBundle

st.set_page_config(page_title="Sales Forecast", layout="wide")

@st.cache_data
def load_cfg(cfg_path: Path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

CFG = load_cfg(Path("config.yaml"))

@st.cache_resource
def load_models(model_dir: str):
    bundle = ModelBundle(model_dir).load()
    return bundle

MODEL = load_models(CFG["model_dir"])

def read_any_table(uploaded_file: io.BytesIO, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        try:
            return pd.read_excel(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)

def make_monthly_forecast(df_pred: pd.DataFrame, dayfirst: bool) -> pd.DataFrame:
    created = pd.to_datetime(df_pred.get("Created Date"), errors="coerce", dayfirst=dayfirst)
    close   = pd.to_datetime(df_pred.get("Close Date"),   errors="coerce", dayfirst=dayfirst)
    if "predicted_days_to_close" in df_pred.columns:
        exp_close = close.fillna(created + pd.to_timedelta(df_pred["predicted_days_to_close"].fillna(0), unit="D"))
    else:
        exp_close = close
    df_pred = df_pred.copy()
    df_pred["Expected Close Date"] = exp_close
    df_pred["Expected Close Month"] = pd.to_datetime(exp_close).dt.to_period("M").dt.to_timestamp("M")
    value_col = "Total Contract Value (converted)" if "Total Contract Value (converted)" in df_pred.columns else "Total Contract Value"
    df_pred["Expected Value"] = df_pred.get(value_col, 0).fillna(0) * df_pred.get("win_probability", 0).fillna(0)
    monthly = (
        df_pred.dropna(subset=["Expected Close Month"])
               .groupby("Expected Close Month", as_index=False)["Expected Value"].sum()
               .sort_values("Expected Close Month")
    )
    monthly.rename(columns={"Expected Close Month":"Month","Expected Value":"Expected Revenue"}, inplace=True)
    return df_pred, monthly

def to_excel_bytes(pred_df: pd.DataFrame, monthly_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        pred_df.to_excel(xw, sheet_name="Predictions", index=False)
        monthly_df.to_excel(xw, sheet_name="Monthly Forecast", index=False)
    bio.seek(0)
    return bio.read()

st.title("📈 Sales Forecast (Upload & Predict)")
st.caption("Upload a pipeline snapshot (Excel/CSV). The app predicts win probability and days-to-close, then aggregates a monthly revenue forecast.")

col_left, col_right = st.columns([2,1], gap="large")
with col_left:
    uploaded = st.file_uploader("Upload pipeline file (.xlsx, .xls, or .csv)", type=["xlsx","xls","csv"])
with col_right:
    dayfirst = st.checkbox("Dates are in DD/MM/YYYY", value=True)
    run_btn = st.button("Run Forecast", type="primary", use_container_width=True, disabled=(uploaded is None))

if run_btn and uploaded is not None:
    try:
        raw_df = read_any_table(uploaded, uploaded.name)
        clean_df = basic_clean(raw_df)
        X = select_features(clean_df, CFG["schema"]["feature_cols"])
        preds = MODEL.predict(X)
        pred_df = pd.concat([clean_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        pred_df, monthly = make_monthly_forecast(pred_df, dayfirst=dayfirst)

        total_rows = len(pred_df)
        total_exp_rev = monthly["Expected Revenue"].sum() if not monthly.empty else 0.0
        avg_prob = pred_df["win_probability"].mean() if "win_probability" in pred_df.columns else np.nan

        k1, k2, k3 = st.columns(3)
        k1.metric("Opportunities", f"{total_rows:,}")
        k2.metric("Sum Expected Revenue", f"{total_exp_rev:,.0f}")
        k3.metric("Avg Win Probability", f"{avg_prob:0.2f}")

        st.subheader("Monthly Expected Revenue")
        if monthly.empty:
            st.info("No valid dates to aggregate. Check Close/Created Date columns in your file.")
        else:
            st.bar_chart(monthly.set_index("Month")["Expected Revenue"])

        st.subheader("Preview (first 20 rows)")
        st.dataframe(pred_df.head(20), use_container_width=True)

        st.subheader("Download outputs")
        fname_stem = Path(uploaded.name).stem
        pred_csv = pred_df.to_csv(index=False).encode("utf-8")
        monthly_csv = monthly.to_csv(index=False).encode("utf-8")
        xlsx_bytes = to_excel_bytes(pred_df, monthly)

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Predictions CSV", pred_csv, file_name=f"predictions_{fname_stem}.csv", mime="text/csv", use_container_width=True)
        c2.download_button("⬇️ Monthly Forecast CSV", monthly_csv, file_name=f"monthly_forecast_{fname_stem}.csv", mime="text/csv", use_container_width=True)
        c3.download_button("⬇️ Excel Dashboard (2 sheets)", xlsx_bytes, file_name="ForecastDashboard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        with st.expander("What the model is doing (quick recap)"):
            st.markdown("""
- **RandomForestClassifier** → predicts **win probability** for each opportunity.
- **RandomForestRegressor** → predicts **days to close**.
- We compute **Expected Close Date** (given Close Date, else Created Date + predicted days).
- **Expected Revenue** = (Total Contract Value (converted) or Total Contract Value) × win probability.
- Then we **sum Expected Revenue by month** to get the monthly forecast.
            """)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a file, tick the date format if needed, then click **Run Forecast**.")
