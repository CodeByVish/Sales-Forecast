from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

# -----------------------
# Helpers (date features)
# -----------------------
def _to_datetime(x):
    return pd.to_datetime(x, errors="coerce")


def _add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Created/Close/Snapshot Year/Month so inputs match what the intern trained on.
    We DO NOT modify models; we only prepare inputs.
    """
    df = df.copy()

    # Created date -> year/month
    if "Created Date" in df.columns:
        d = _to_datetime(df["Created Date"])
        if "Created Year" not in df.columns:
            df["Created Year"] = d.dt.year
        if "Created Month" not in df.columns:
            df["Created Month"] = d.dt.month

    # Close date -> year/month
    if "Close Date" in df.columns:
        d = _to_datetime(df["Close Date"])
        if "Close Year" not in df.columns:
            df["Close Year"] = d.dt.year
        if "Close Month" not in df.columns:
            df["Close Month"] = d.dt.month

    # Snapshot year/month: locally infer sensible defaults if absent
    if "Snapshot Year" not in df.columns:
        if "Close Year" in df.columns:
            df["Snapshot Year"] = df["Close Year"]
        elif "Created Year" in df.columns:
            df["Snapshot Year"] = df["Created Year"]
    if "Snapshot Month" not in df.columns:
        if "Close Month" in df.columns:
            df["Snapshot Month"] = df["Close Month"]
        elif "Created Month" in df.columns:
            df["Snapshot Month"] = df["Created Month"]

    return df


# -----------------------
# One-hot fallback (no saved encoder or mismatch)
# -----------------------
def _enc_fallback_build(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Build a DataFrame with exactly the columns listed in expected_cols, efficiently (no fragmentation).
    - Columns starting with 'ENC_' are treated as one-hot for the pattern: ENC_<Feature>_<Category>
    - Numeric/date passthrough columns are copied from X; missing become 0
    - Unknown/missing columns are filled with 0
    """
    X = X.copy()

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    cols_map = {norm(c): c for c in X.columns}
    def getcol(name): return cols_map.get(norm(name))

    col_data = {}  # collect arrays then build once

    for col in expected_cols:
        if col.startswith("ENC_"):
            # Parse: ENC_<Feature>_<Category>  (split at the last underscore)
            base = col[4:]
            if "_" in base:
                i = base.rfind("_")
                feat = base[:i]
                cat = base[i + 1:]
            else:
                col_data[col] = np.zeros(len(X), dtype=np.int8)
                continue

            feat_src = getcol(feat)
            if feat_src is None:
                col_data[col] = np.zeros(len(X), dtype=np.int8)
            else:
                v = X[feat_src].astype(str).fillna("")
                col_data[col] = (v == str(cat)).astype(np.int8).to_numpy()
        else:
            # numeric/date passthrough
            src = getcol(col)
            if src is None:
                col_data[col] = np.zeros(len(X), dtype=np.float32)
            else:
                col_data[col] = pd.to_numeric(X[src], errors="coerce").fillna(0).astype(np.float32).to_numpy()

    # Build the frame in one shot to avoid fragmentation warnings and speed up
    out = pd.DataFrame(col_data, index=X.index)
    return out


# -----------------------
# Loader
# -----------------------
class ModelBundle:
    """
    Uses the intern's 5-file bundle:
      - cl_encoder.pkl                (optional but preferred)
      - cl_feature_columns.pkl        (may exist; not strictly relied on)
      - cl_model.pkl                  (RandomForestClassifier)
      - master_regression_encoder.pkl (optional)
      - regression_model.pkl          (RandomForestRegressor)

    Strategy (NO retraining, NO artifact edits):
      * For classifier:
        1) Add date parts (Created/Close/Snapshot Year/Month)
        2) Try saved encoder.transform(all cols) -> DataFrame with names
        3) Align to model.feature_names_in_: add missing=0, drop extras, reorder
        4) If encoder missing or names mismatch -> build exact encoded DF via fallback
      * For regressor: same idea.
    """
    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self.cl_encoder = None
        self.cl_model = None
        self.cl_feature_columns = None

        self.reg_encoder = None
        self.reg_model = None

    def _safe_load(self, name: str):
        p = self.model_dir / name
        return joblib.load(p) if p.exists() else None

    def load(self):
        self.cl_encoder = self._safe_load("cl_encoder.pkl")
        self.cl_model = self._safe_load("cl_model.pkl")
        self.cl_feature_columns = self._safe_load("cl_feature_columns.pkl")

        self.reg_encoder = self._safe_load("master_regression_encoder.pkl")
        self.reg_model = self._safe_load("regression_model.pkl")

        # Fallback: if only one big artifact exists, pick it up
        if self.cl_model is None and self.reg_model is None:
            cands = list(self.model_dir.glob("*.pkl")) + list(self.model_dir.glob("*.joblib"))
            if not cands:
                raise FileNotFoundError(f"No model artifacts found under {self.model_dir}")
            largest = max(cands, key=lambda p: p.stat().st_size)
            obj = joblib.load(largest)
            if hasattr(obj, "predict_proba"):
                self.cl_model = obj
            else:
                self.reg_model = obj
        return self

    # ---- Classifier prep ----
    def _prepare_cl(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        Xp = _add_date_parts(X_raw)
        expected = getattr(self.cl_model, "feature_names_in_", None)

        if expected is None:
            if self.cl_encoder is not None:
                Xt = self.cl_encoder.transform(Xp)
                if isinstance(Xt, pd.DataFrame):
                    return Xt
                try:
                    names = self.cl_encoder.get_feature_names_out()
                    return pd.DataFrame(Xt, columns=names, index=Xp.index)
                except Exception:
                    return pd.DataFrame(Xt, index=Xp.index)
            return Xp.select_dtypes(include=[np.number]).fillna(0)

        if self.cl_encoder is not None:
            try:
                Xt = self.cl_encoder.transform(Xp)
                if isinstance(Xt, pd.DataFrame):
                    Xt_df = Xt.copy()
                else:
                    try:
                        names = self.cl_encoder.get_feature_names_out()
                        Xt_df = pd.DataFrame(Xt, columns=names, index=Xp.index)
                    except Exception:
                        raise RuntimeError("Encoder returned unnamed matrix; switching to fallback encoding.")
                # Align to expected (add missing=0, drop extras)
                for col in expected:
                    if col not in Xt_df.columns:
                        Xt_df[col] = 0
                Xt_df = Xt_df[list(expected)]
                return Xt_df
            except Exception:
                pass  # fall through

        # Fallback: build exactly the expected columns
        return _enc_fallback_build(Xp, list(expected))

    # ---- Regressor prep ----
    def _prepare_reg(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        Xp = _add_date_parts(X_raw)
        expected = getattr(self.reg_model, "feature_names_in_", None)

        if self.reg_encoder is not None:
            try:
                Xt = self.reg_encoder.transform(Xp)
                if isinstance(Xt, pd.DataFrame):
                    Xt_df = Xt.copy()
                else:
                    try:
                        names = self.reg_encoder.get_feature_names_out()
                        Xt_df = pd.DataFrame(Xt, columns=names, index=Xp.index)
                    except Exception:
                        Xt_df = pd.DataFrame(Xt, index=Xp.index)
                if expected is not None:
                    for col in expected:
                        if col not in Xt_df.columns:
                            Xt_df[col] = 0
                    Xt_df = Xt_df[list(expected)]
                return Xt_df
            except Exception:
                if expected is not None:
                    return _enc_fallback_build(Xp, list(expected))
                return Xp.select_dtypes(include=[np.number]).fillna(0)

        if expected is not None:
            return _enc_fallback_build(Xp, list(expected))
        return Xp.select_dtypes(include=[np.number]).fillna(0)

    # ---- Public API ----
    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X_raw.index)

        if self.cl_model is not None:
            Xc = self._prepare_cl(X_raw)
            if hasattr(self.cl_model, "predict_proba"):
                proba = self.cl_model.predict_proba(Xc)
                prob1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.ravel(proba)
            else:
                # Rare: no proba -> normalize scores
                scores = self.cl_model.predict(Xc)
                smin, smax = np.min(scores), np.max(scores)
                prob1 = (scores - smin) / (smax - smin + 1e-9)
            out["win_probability"] = np.asarray(prob1)

        if self.reg_model is not None:
            Xr = self._prepare_reg(X_raw)
            out["predicted_days_to_close"] = self.reg_model.predict(Xr)

        if out.empty:
            raise RuntimeError("No usable model found. Ensure artifacts are in artifacts/model/")
        return out
