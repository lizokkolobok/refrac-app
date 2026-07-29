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

# known aliases: obvious alternative spellings -> canonical model name
KNOWN_ALIASES = {
    "tvd": "TVD_FT", "tvd_ft": "TVD_FT", "true_vertical_depth": "TVD_FT",
    "md": "MD_FT", "measured_depth": "MD_FT",
    "proppant": "Proppant_LBS", "proppant_lbs": "Proppant_LBS", "total_proppant": "Proppant_LBS",
    "perf_interval": "PerfInterval_FT", "perforation_interval": "PerfInterval_FT",
    "frac_water": "frac_water_bbl", "water_bbl": "frac_water_bbl",
    "last12": "last12_oil_rate", "oil_last12": "last12_oil_rate", "last_12_oil": "last12_oil_rate", "oillast12": "last12_oil_rate",
    "last6": "last6_oil_rate", "oil_last6": "last6_oil_rate",
    "peak": "peak_oil", "peak_oil_rate": "peak_oil",
    "cum_oil": "cum_oil_at_refrac", "cumulative_oil": "cum_oil_at_refrac",
    "cum_gas": "cum_gas_at_refrac", "cumulative_gas": "cum_gas_at_refrac",
}

def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _tokens(s):
    import re
    return set(t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t)

def suggest_matches(missing, extra):
    """For each missing model column, propose similar unused columns from the upload,
    ranked best-first. Alias hits rank top; then token overlap; then string similarity."""
    from difflib import SequenceMatcher
    out = {}
    for m in missing:
        mn = _norm(m); mtok = _tokens(m)
        scored = []
        for e in extra:
            en = _norm(e); etok = _tokens(e)
            if KNOWN_ALIASES.get(en) == m:
                score = 1.0
            else:
                overlap = len(mtok & etok) / max(1, len(mtok | etok))     # token Jaccard
                ratio   = SequenceMatcher(None, mn, en).ratio()
                contains = mn in en or en in mn
                score = max(overlap, ratio, 0.85 if contains else 0)
            if score >= 0.5:
                scored.append((round(score, 3), e))
        scored.sort(reverse=True)
        if scored:
            out[m] = [e for _, e in scored]
    return out


def suggest_by_content(missing, extra, df_new, train, feats):
    """Content-based hint: for each missing model column, find unused numeric columns
    in the upload whose value distribution resembles that column in the training data.
    Returns {missing_col: [candidate_cols ranked by distribution similarity]}."""
    import numpy as np
    def stats(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 5: return None
        return np.array([s.median(), s.quantile(0.1), s.quantile(0.9)], float)
    train_stats = {m: stats(train[m]) for m in missing if m in train.columns}
    extra_stats = {e: stats(df_new[e]) for e in extra}
    out = {}
    for m, ts in train_stats.items():
        if ts is None: continue
        scale = abs(ts[0]) + 1.0
        scored = []
        for e, es in extra_stats.items():
            if es is None: continue
            # normalized distance between the two 3-number summaries
            dist = float(np.mean(np.abs(ts - es)) / scale)
            if dist < 0.5:                      # within ~50% of the training scale
                scored.append((round(1 - dist, 3), e))
        scored.sort(reverse=True)
        if scored:
            out[m] = [e for _, e in scored]
    return out

st.set_page_config(page_title="Re-Frac Candidate Screening", page_icon="", layout="wide")


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

# initialise the confirmed rename map for this upload
if "colmap" not in st.session_state:
    st.session_state.colmap = {}

if crit_missing:
    st.warning("**Important features are missing.** If your file uses different column "
               "names for the same thing, match them below — nothing is renamed until you confirm.")
    name_sug = suggest_matches(crit_missing, extra)
    content_sug = suggest_by_content(crit_missing, extra, df_raw, train, feats)
    with st.form("colmap_form"):
        chosen = {}
        for miss in crit_missing:
            by_name = name_sug.get(miss, [])
            by_content = [e for e in content_sug.get(miss, []) if e not in by_name]
            ranked = by_name + by_content                       # name matches first
            options = ["— (leave missing) —"] + ranked + \
                      [e for e in extra if e not in ranked]
            default_ix = 1 if ranked else 0
            # label shows why each candidate is suggested
            tag = ""
            if by_name:    tag = f"  (best guess by name: {by_name[0]})"
            elif by_content: tag = f"  (guess by data values: {by_content[0]})"
            pick = st.selectbox(f"Model needs **{miss}** — which of your columns is this?{tag}",
                                options, index=default_ix, key=f"map_{miss}")
            if pick != options[0]:
                chosen[pick] = miss
        applied = st.form_submit_button("Apply column matches")
    if applied:
        st.session_state.colmap = chosen
        if chosen:
            st.success("Matched: " + ", ".join(f"`{k}` → `{v}`" for k, v in chosen.items()))
        else:
            st.info("No matches applied — missing columns will be filled with typical values.")
    # re-check after any confirmed mapping
    if st.session_state.colmap:
        df_raw = df_raw.rename(columns=st.session_state.colmap)
        missing, extra, crit_missing, hints = check_columns(df_raw, feats)
        if not crit_missing:
            st.success("All important features now present after matching.")
        else:
            st.warning("Still missing: " + ", ".join(f"`{c}`" for c in crit_missing))
else:
    st.success("All important features present.")

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

    st.download_button("Download full ranked CSV",
                       ranked.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_wells.csv", mime="text/csv")
