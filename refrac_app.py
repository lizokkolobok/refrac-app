"""
Re-Frac Candidate Screening — web interface for the v1 (CBP) model.
Upload a CSV of wells (feature columns, like training_data.csv) and the app
ranks them, flags missing important features, and returns the top candidates.

Run locally / in Colab:   streamlit run refrac_app.py
Requires (see requirements.txt): streamlit, pandas, numpy, scikit-learn, joblib
The two model files must sit next to this script:
    final_quantile_models.joblib
    training_data.csv
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
MODEL_PATH = os.path.join(HERE, "final_quantile_models.joblib")
TRAIN_PATH = os.path.join(HERE, "training_data.csv")
SUCCESS_BOE = 15000

# important features — a missing one makes predictions unreliable
CRITICAL = ["last12_oil_rate", "last6_oil_rate", "peak_oil",
            "cum_oil_at_refrac", "cum_gas_at_refrac", "months_on_prod_at_refrac",
            "Proppant_LBS", "PerfInterval_FT", "frac_water_bbl"]

st.set_page_config(page_title="Re-Frac Candidate Screening", page_icon="️", layout="wide")


# ----------------------------- model loading -----------------------------
@st.cache_resource
def load_model():
    models = joblib.load(MODEL_PATH)
    train = pd.read_csv(TRAIN_PATH)
    feats = list(models[0.50].feature_names_in_)
    return models, train, feats


def num(f, c):
    return pd.to_numeric(f[c], errors="coerce") if c in f.columns else pd.Series(np.nan, index=f.index)


def add_eng(df):
    df = df.copy()
    df["eng_depletion_rate"] = num(df, "cum_oil_at_refrac") / (num(df, "months_on_prod_at_refrac") + 1)
    df["eng_off_peak_ratio"] = num(df, "last6_oil_rate") / (num(df, "peak_oil") + 1)
    df["eng_decline_ratio"] = num(df, "last6_oil_rate") / (num(df, "last12_oil_rate") + 1)
    df["eng_gas_fraction"] = num(df, "cum_gas_at_refrac") / (num(df, "cum_oil_at_refrac") * 6 + num(df, "cum_gas_at_refrac") + 1)
    df["eng_proppant_per_perf"] = num(df, "Proppant_LBS") / (num(df, "PerfInterval_FT") + 1)
    return df.replace([np.inf, -np.inf], np.nan)


def check_columns(df_new, feats):
    have = set(df_new.columns)
    raw_needed = set(feats) | {"cum_oil_at_refrac", "months_on_prod_at_refrac", "last6_oil_rate",
                               "peak_oil", "last12_oil_rate", "cum_gas_at_refrac", "Proppant_LBS", "PerfInterval_FT"}
    ENG = {"eng_decline_ratio","eng_off_peak_ratio","eng_depletion_rate",
           "eng_gas_fraction","eng_proppant_per_perf"}
    missing = sorted(((raw_needed - have) & (set(feats) | set(CRITICAL))) - ENG)
    extra = sorted(have - raw_needed)
    crit_missing = [c for c in missing if c in CRITICAL and c not in ENG]
    # rename hints
    hints = []
    for m in crit_missing:
        key = m.replace("_", "").lower()
        for e in extra:
            el = e.replace("_", "").lower()
            if key[:5] in el or el[:5] in key:
                hints.append((e, m))
    return missing, extra, crit_missing, hints


def score(df_raw, models, train, feats):
    train_e = add_eng(train)
    med = {c: pd.to_numeric(train_e[c], errors="coerce").median() for c in feats if c in train_e.columns}
    freq = {}
    for fc, src in [("county_freq", "county"), ("Field_freq", "Field")]:
        if src in train.columns:
            freq[fc] = train[src].value_counts().to_dict()

    df = add_eng(df_raw)
    for fc, src in [("county_freq", "county"), ("Field_freq", "Field")]:
        if fc in feats and src in df.columns:
            df[fc] = df[src].map(freq.get(fc, {})).fillna(0)
    X = df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    for c in feats:
        X[c] = X[c].fillna(med.get(c, 0))

    out = df_raw.copy()
    out["pred_low_p05"] = models[0.05].predict(X).round(0)
    out["pred_central_p50"] = models[0.50].predict(X).round(0)
    out["pred_upside_p95"] = models[0.95].predict(X).round(0)
    out["model_rank"] = out["pred_upside_p95"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("model_rank").reset_index(drop=True)


# ----------------------------- UI -----------------------------
st.title("Re-Frac Candidate Screening")
st.caption("Upload a CSV of wells and the model ranks them by predicted re-frac upside. "
           "Model: v1 (Central Basin Platform, Permian).")

if not (os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_PATH)):
    st.error("Model files not found. Place **final_quantile_models.joblib** and "
             "**training_data.csv** in the same folder as this app.")
    st.stop()

models, train, feats = load_model()

with st.sidebar:
    st.header("Settings")
    top_n = st.number_input("How many top candidates to show", 10, 1000, 100, step=10)
    st.markdown("---")
    st.markdown(f"**Model expects {len(feats)} feature columns.**")
    with st.expander("See required columns"):
        st.write(sorted(feats))
    st.markdown("**Reading the output**")
    st.markdown("- **p50** — best single estimate (BOE)\n"
                "- **p95** — upper bound / ranking score\n"
                "- **p05–p95** — uncertainty range\n"
                "- **rank** — 1 = strongest candidate")

uploaded = st.file_uploader("Upload wells CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV with one row per well. Columns should match the model's features "
            "(same format as training_data.csv). Missing columns are filled with typical values; "
            "important missing columns are flagged below after upload.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

st.success(f"Loaded {len(df_raw):,} wells with {df_raw.shape[1]} columns.")

# ---- column check ----
missing, extra, crit_missing, hints = check_columns(df_raw, feats)
c1, c2 = st.columns(2)
c1.metric("Features the model needs", len(feats))
c2.metric("Missing (median-filled)", len(missing), delta=f"{len(crit_missing)} important" if crit_missing else "none important",
          delta_color="inverse" if crit_missing else "off")

if crit_missing:
    st.warning("⚠️ **Important features are missing — predictions will be less reliable:**\n\n"
               + "\n".join(f"- `{c}`" for c in crit_missing))
    for e, m in hints:
        st.info(f"💡 Your column **`{e}`** looks like it might be **`{m}`** renamed. "
                f"Rename it to match so the model uses it.")
else:
    st.success("✅ All important features present.")

if st.button("Run screening", type="primary"):
    with st.spinner("Scoring wells..."):
        ranked = score(df_raw, models, train, feats)
    st.subheader(f"Top {min(top_n, len(ranked))} candidates")

    show_cols = [c for c in ["model_rank", "well_API14", "API14", "operator", "Formation",
                             "pred_central_p50", "pred_upside_p95", "pred_low_p05"]
                 if c in ranked.columns]
    top = ranked.head(int(top_n))
    st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

    # quick visual: predicted upside of the top candidates
    if "pred_upside_p95" in top.columns:
        st.bar_chart(top.set_index(show_cols[1] if len(show_cols) > 1 else "model_rank")["pred_upside_p95"].head(30))

    st.download_button("⬇️ Download full ranked CSV",
                       ranked.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_wells.csv", mime="text/csv")
