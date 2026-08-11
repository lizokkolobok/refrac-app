"""
Re-Frac Candidate Screening - HYBRID web interface.
Scores uploaded wells using each basin's best model:
  - dead basins (no successful re-fracs) are excluded from results
  - strong basins use their own dedicated model
  - all other basins use the national model (v2.1)
Self-contained: reproduces the training feature pipeline from the bundles.

Files that must sit next to this script:
  national_model_v2_1.joblib
  basin_models/model_<BASIN>.joblib   (one per basin with its own model)

Run:  streamlit run refrac_app.py
Requirements: streamlit, pandas, numpy, scikit-learn, joblib
"""
import os, glob
import numpy as np
import pandas as pd
import joblib
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
NATIONAL_PATH = os.path.join(HERE, "national_model_v2_1.joblib")
BASIN_DIR = os.path.join(HERE, "basin_models")
SHORTLIST_PATH = os.path.join(HERE, "national_shortlist_v2_1.csv")   # historical data for the calibration table

# ---- CONFIG ----------------------
OWN_MODEL_BASINS = {"ARK-LA-TX", "MIDLAND", "FORT WORTH", "PERMIAN OTHER"}  # own model clearly wins OOF
# Basins where the dedicated basin model beats the national model out-of-fold.
# (Delaware excluded: basin wins by only +0.006 ROC on 168 wells - within noise, national safer.
#  Mid-Continent excluded: national actually wins there, 0.947 vs 0.932.)
DEAD_BASINS = {"SAN JUAN", "RATON"}                              # 0% success -> excluded
# Weak basins: uplift is non-zero but <=2% of wells clear the 10k breakeven -> not worth screening.
WEAK_BASINS = {"ARKOMA", "ROCKIES OTHER", "SACRAMENTO", "PICEANCE", "MID-CONTINENT OTHER"}
# (San Juan/Raton already covered by DEAD_BASINS; both groups can be dropped or kept via a toggle.)
REFRAC_COST_USD = 400_000
PROFIT_PER_BOE_USD = 40.0
BREAKEVEN_BOE = REFRAC_COST_USD / PROFIT_PER_BOE_USD             # 10,000

SUCCESS_BOE = 15000
CRITICAL = ["last12_oil_rate", "last6_oil_rate", "peak_oil",
            "cum_oil_at_refrac", "cum_gas_at_refrac", "months_on_prod_at_refrac",
            "Proppant_LBS", "PerfInterval_FT", "frac_water_bbl"]
# TOP5 most important input features - if any is missing, the prediction is NOT valid.
TOP5 = ["last12_oil_rate", "last6_oil_rate", "peak_oil", "cum_oil_at_refrac", "Proppant_LBS"]

# ---- feature lists ----------------------------
STRUCTURAL = ["well_age_yrs", "job_year", "refrac_seq", "well_n_refracs",
              "tvd_ft", "frac_water_bbl", "is_injector",
              "dist_nearest_ft", "n_offsets_660ft", "n_offsets_1320ft",
              "n_offsets_2640ft", "n_offsets_2640ft_pre_refrac",
              "known_injector_within_radius", "lat", "lon"]
PRODUCTION = ["months_on_prod_at_refrac", "cum_oil_at_refrac", "cum_gas_at_refrac",
              "cum_water_at_refrac", "last6_oil_rate", "last12_oil_rate",
              "ip30_oil", "ip90_oil", "peak_oil", "months_to_peak",
              "underperf_ratio", "pre_arps_Di", "pre_arps_b",
              "water_cut_at_refrac", "gor_at_refrac"]
WELLS = ["TVD_FT", "MD_FT", "PerfInterval_FT", "LateralLength_FT", "FracStages",
         "AverageStageSpacing_FT", "Proppant_LBS", "ProppantIntensity_LBSPerFT",
         "TotalFluidPumped_BBL", "FluidIntensity_BBLPerFT", "AcidVolume_BBL",
         "Bottom_Hole_Temp_DEGF", "NumberOfStrings", "OilTestRate_BBLPerDAY",
         "First3MonthOil_BBL", "First12MonthOil_BBL", "WHLiquids_PCT", "GOR_ScfPerBbl",
         "orig_proppant_lbs", "orig_fluid_bbl", "orig_frac_stages",
         "orig_proppant_intensity", "n_formation_tops", "formation_column_ft",
         "shallowest_top_ft", "deepest_top_ft"]
NUMERIC = STRUCTURAL + PRODUCTION + WELLS
ONEHOT = ["Trajectory", "ENVWellboreType", "ENVWellStatus", "ENVProdWellType",
          "ENVWellType", "Conventional", "state", "ENVProducingMethod"]
ENG_COLS = ["eng_depletion_rate", "eng_off_peak_ratio", "eng_decline_ratio",
            "eng_gas_fraction", "eng_proppant_per_perf"]

# ---- column-matching helpers (renamed-column handling) ----------------------
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
                overlap = len(mtok & etok) / max(1, len(mtok | etok))
                ratio   = SequenceMatcher(None, mn, en).ratio()
                contains = mn in en or en in mn
                score = max(overlap, ratio, 0.85 if contains else 0)
            if score >= 0.5:
                scored.append((round(score, 3), e))
        scored.sort(reverse=True)
        if scored:
            out[m] = [e for _, e in scored]
    return out

def suggest_by_content(missing, extra, df_new):
    """Content hint using the national model's stored basin medians as the reference scale."""
    def stats(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) < 5: return None
        return np.array([s.median(), s.quantile(0.1), s.quantile(0.9)], float)
    # reference scale for each missing col: global median from national bundle basin_medians
    ref = {}
    for m in missing:
        bm = national.get("basin_medians", {}).get(m)
        if bm and "global" in bm:
            g = bm["global"]; ref[m] = np.array([g, g*0.5, g*1.5], float)
    extra_stats = {e: stats(df_new[e]) for e in extra}
    out = {}
    for m, ts in ref.items():
        scale = abs(ts[0]) + 1.0
        scored = []
        for e, es in extra_stats.items():
            if es is None: continue
            dist = float(np.mean(np.abs(ts - es)) / scale)
            if dist < 0.5:
                scored.append((round(1 - dist, 3), e))
        scored.sort(reverse=True)
        if scored:
            out[m] = [e for _, e in scored]
    return out

def check_columns(df_new, feats):
    have = set(df_new.columns)
    raw_needed = set(feats) | {"cum_oil_at_refrac", "months_on_prod_at_refrac", "last6_oil_rate",
                               "peak_oil", "last12_oil_rate", "cum_gas_at_refrac", "Proppant_LBS", "PerfInterval_FT"}
    ENG = set(ENG_COLS)
    missing = sorted(((raw_needed - have) & (set(feats) | set(CRITICAL))) - ENG)
    extra = sorted(have - raw_needed)
    crit_missing = [c for c in missing if c in CRITICAL and c not in ENG]
    return missing, extra, crit_missing


def flag_recent_refracs(df, years):
    """Return a boolean Series: True if the well's last re-frac was within `years`.
    Uses refrac_date if present, else job_year. Wells with no date -> False (unknown)."""
    import datetime
    today = pd.Timestamp(datetime.date.today())
    flag = pd.Series(False, index=df.index)
    if "refrac_date" in df.columns:
        dt = pd.to_datetime(df["refrac_date"], errors="coerce")
        age_yrs = (today - dt).dt.days / 365.25
        flag = flag | (age_yrs < years)
    elif "job_year" in df.columns:
        yr = pd.to_numeric(df["job_year"], errors="coerce")
        flag = flag | ((today.year - yr) < years)
    return flag.fillna(False)


@st.cache_data
def load_calibration_data():
    """Load the historical shortlist once. Returns (DataFrame, p50col, actcol, basincol) or Nones."""
    if not os.path.exists(SHORTLIST_PATH):
        return None, None, None, None
    df = pd.read_csv(SHORTLIST_PATH)
    p50c = next((c for c in ["pred_central_p50","pred_mid_p50"] if c in df.columns), None)
    actc = next((c for c in ["actual_boe","actual"] if c in df.columns), None)
    basc = next((c for c in ["ENVBasin","basin"] if c in df.columns), None)
    if not (p50c and actc):
        return None, None, None, None
    df = df.copy()
    df[p50c] = pd.to_numeric(df[p50c], errors="coerce")
    df[actc] = pd.to_numeric(df[actc], errors="coerce")
    df = df.dropna(subset=[p50c, actc])
    if basc:
        df[basc] = df[basc].astype(str).str.upper().str.strip()
    return df, p50c, actc, basc

def _cal_bins(max_p50):
    """Uniform 5k-wide P50 bins from <0 up past the data max."""
    top = int(np.ceil(max(max_p50, 5000) / 5000.0) * 5000)
    edges = [-np.inf, 0] + list(range(5000, top + 5000, 5000))
    labels = ["< 0"]
    lo = 0
    for hi in edges[2:]:
        labels.append(f"{lo//1000}-{hi//1000}k")
        lo = hi
    return edges, labels

def calibration_for(d, p50c, actc, threshold=BREAKEVEN_BOE):
    """Calibration table for a wells subset, in uniform 5k P50 buckets."""
    if d is None or len(d) == 0:
        return None, None
    edges, labels = _cal_bins(float(d[p50c].max()))
    b = pd.cut(d[p50c], bins=edges, labels=labels)
    ok = (d[actc] >= threshold).astype(int)
    rows = []
    for lab in labels:
        m = (b == lab).values
        if m.sum() == 0: continue
        acc = ok[m].mean()
        rows.append({"model_predicted_P50": lab, "n_wells": int(m.sum()),
                     "actually_exceeded_breakeven": f"{acc*100:.0f}%",
                     "median_actual_BOE": int(d[actc][m].median())})
    return pd.DataFrame(rows), float(ok.mean())

@st.cache_data
def basin_calibration_prob(min_bucket=5):
    """Per-basin lookup: {basin: {bucket_label: success_fraction}} in 5k buckets.
    Only buckets with >= min_bucket wells are kept (small buckets are too noisy)."""
    d, p50c, actc, basc = load_calibration_data()
    if d is None or not basc:
        return None, None, None
    edges, labels = _cal_bins(float(d[p50c].max()))
    d = d.copy()
    d["_b"] = pd.cut(d[p50c], bins=edges, labels=labels)
    d["_ok"] = (d[actc] >= BREAKEVEN_BOE).astype(int)
    table = {}
    for basin, g in d.groupby(basc):
        m = {}
        for lab, gg in g.groupby("_b", observed=True):
            if len(gg) >= min_bucket:
                m[str(lab)] = float(gg["_ok"].mean())
        if m:
            table[basin] = m
    return table, edges, labels

def calib_prob_for_wells(ranked):
    """For each well, look up the historical success rate of its basin+P50 bucket."""
    tbl, edges, labels = basin_calibration_prob()
    if tbl is None:
        return None
    basin = ranked["ENVBasin"].astype(str).str.upper().str.strip() if "ENVBasin" in ranked.columns else None
    p50 = pd.to_numeric(ranked.get("pred_central_p50"), errors="coerce")
    if basin is None:
        return None
    buckets = pd.cut(p50, bins=edges, labels=labels).astype(str)
    out = []
    for bsn, bkt in zip(basin, buckets):
        out.append(tbl.get(bsn, {}).get(bkt, np.nan))
    return pd.Series(out, index=ranked.index)

st.set_page_config(page_title="Re-Frac Screening", page_icon="oil", layout="wide")

# ---- light professional theme: clean background + teal accent ----
st.markdown("""
<style>
:root {
  --bg: #fffffa;
  --panel: #edf4fa;
  --teal: #edf4fa;
  --teal-dark: #090D73;
  --teal-soft: #090d73;
  --text: #1F2A2E;
  --muted: #5F6E73;
  --line: #DCE4E6;
}
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
h1, h2, h3 { color: var(--teal-dark); font-family: 'Segoe UI', Helvetica, Arial, sans-serif; letter-spacing: .2px; }
h1 { border-bottom: 3px solid var(--teal); padding-bottom: .35rem; }
.stApp p, .stApp label, .stApp span, .stApp li { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
/* buttons */
.stButton>button, .stDownloadButton>button {
  background: var(--teal); color: #FFFFFF; border: none; font-weight: 600; border-radius: 6px;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: var(--teal-dark); color:#FFFFFF; }
.stButton>button[kind="primary"] { background: var(--teal); color:#FFFFFF; }
/* dataframe + expander cards */
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
/* subtle teal tint on sidebar headers */
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--teal-dark); }
[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def _find_basin_model_files():
    """Look for basin models in basin_models/ AND next to the app (root)."""
    files = glob.glob(os.path.join(BASIN_DIR, "model_*.joblib"))
    files += glob.glob(os.path.join(HERE, "model_*.joblib"))   # loaded loose in the repo root
    return sorted(set(files))

@st.cache_resource
def load_models():
    national = joblib.load(NATIONAL_PATH)
    basins, load_errors = {}, []
    for mf in _find_basin_model_files():
        try:
            b = joblib.load(mf)
            name = b.get("version", "").replace("basin-", "").upper().strip()
            if name:
                basins[name] = b
        except Exception as e:
            load_errors.append((os.path.basename(mf), f"{type(e).__name__}: {e}"))
    return national, basins, load_errors


def _num(f, c):
    if c not in f.columns:
        return pd.Series(np.nan, index=f.index)
    col = f[c]
    if isinstance(col, pd.DataFrame):   # duplicate column names -> take the first
        col = col.iloc[:, 0]
    return pd.to_numeric(col, errors="coerce")

def add_eng(df):
    df = df.copy()
    if df.columns.duplicated().any():                 # guard: duplicate names break _num
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df["eng_depletion_rate"] = _num(df, "cum_oil_at_refrac") / (_num(df, "months_on_prod_at_refrac") + 1)
    df["eng_off_peak_ratio"] = _num(df, "last6_oil_rate") / (_num(df, "peak_oil") + 1)
    df["eng_decline_ratio"] = _num(df, "last6_oil_rate") / (_num(df, "last12_oil_rate") + 1)
    df["eng_gas_fraction"] = _num(df, "cum_gas_at_refrac") / (_num(df, "cum_oil_at_refrac") * 6 + _num(df, "cum_gas_at_refrac") + 1)
    df["eng_proppant_per_perf"] = _num(df, "Proppant_LBS") / (_num(df, "PerfInterval_FT") + 1)
    return df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

def build_features(df):
    if df.columns.duplicated().any():                 # guard: keep first of any duplicate
        df = df.loc[:, ~df.columns.duplicated()].copy()
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        if c in df.columns:
            col = df[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            X[c] = pd.to_numeric(col, errors="coerce")
    present = [c for c in ONEHOT if c in df.columns]
    if present:
        oh = pd.get_dummies(df[present].astype(str), prefix=present, dummy_na=True)
        X = pd.concat([X, oh.set_index(X.index)], axis=1)
    return X.astype(float)

def features_for_bundle(df_raw, B, basin_series):
    df = add_eng(df_raw)
    X = build_features(df)
    X = X.loc[:, ~X.columns.duplicated()]
    for c in ENG_COLS:
        if c in df.columns and c not in X.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
    for c, bm in B.get("basin_medians", {}).items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            med = basin_series.map(bm["per_basin"]).fillna(bm["global"])
            X[f"rel_{c}"] = v / (med + 1)
    X["basin_n_wells"] = np.log1p(basin_series.map(B.get("basin_counts", {})).fillna(1))
    for c, fmap in B.get("freq_maps", {}).items():
        if c in df.columns:
            X[f"{c}_freq"] = df[c].astype(str).map(fmap).fillna(0)
    FEATS = B["feats"]
    Xs = X.reindex(columns=FEATS).apply(pd.to_numeric, errors="coerce")
    for c in FEATS:
        Xs[c] = Xs[c].fillna(B.get("train_medians", {}).get(c, 0))
    return Xs

def predict_bundle(df_raw, B, basin_series):
    Xs = features_for_bundle(df_raw, B, basin_series)
    LEV = np.array(B["qgrid"]); models = B["models"]
    Q = np.sort(np.vstack([models[q].predict(Xs) for q in LEV]).T, axis=1)
    p50 = Q[:, list(LEV).index(0.50)]
    if "classifier" in B:
        prob = B["classifier"].predict_proba(Xs.fillna(-999))[:, 1]
    else:
        T = BREAKEVEN_BOE
        prob = np.array([1 - LEV[0] if T <= qs[0] else (1 - LEV[-1] if T >= qs[-1]
                        else 1 - np.interp(T, qs, LEV)) for qs in Q])
    c = float(B.get("conformal", {}).get("__GLOBAL__", 0))
    return p50, Q[:, 0] - c, Q[:, -1] + c, prob


def hybrid_score(df_raw, national, basins, drop_weak=True):
    basin = df_raw["ENVBasin"].astype(str).str.upper().str.strip() if "ENVBasin" in df_raw.columns \
            else pd.Series("UNKNOWN", index=df_raw.index)
    df_raw = df_raw.copy(); df_raw["_basin"] = basin

    p50 = np.full(len(df_raw), np.nan); lo = np.full(len(df_raw), np.nan)
    hi = np.full(len(df_raw), np.nan); prob = np.full(len(df_raw), np.nan)
    model_used = np.array(["national"] * len(df_raw), dtype=object)

    p50[:], lo[:], hi[:], prob[:] = predict_bundle(df_raw, national, basin)

    for b in OWN_MODEL_BASINS:
        if b in basins:
            mask = (basin == b).values
            if mask.any():
                sub = df_raw[mask]
                p, l, h, pr = predict_bundle(sub, basins[b], sub["_basin"])
                p50[mask], lo[mask], hi[mask], prob[mask] = p, l, h, pr
                model_used[mask] = f"basin:{b}"

    out = df_raw.copy()
    out["ENVBasin"] = basin
    out["prob_exceeds_breakeven"] = np.round(prob, 3)
    out["expected_profit_USD"] = np.round(p50 * PROFIT_PER_BOE_USD - REFRAC_COST_USD, 0)
    # risk-adjusted profit: weight the upside by the probability of clearing breakeven,
    # so a big P50 with a low probability is discounted vs a reliable smaller well.
    out["risk_adjusted_profit_USD"] = np.round(prob * p50 * PROFIT_PER_BOE_USD - REFRAC_COST_USD, 0)
    out["pred_central_p50"] = np.round(p50, 0)
    out["band_low"] = np.round(lo, 0)
    out["band_high"] = np.round(hi, 0)
    # relative uncertainty: interval width as a fraction of the prediction (scale-free)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (hi - lo) / np.where(np.abs(p50) < 1, np.nan, p50)
    out["relative_uncertainty"] = np.round(np.abs(rel), 2)
    out["model_used"] = model_used

    dead_mask = basin.isin(DEAD_BASINS).values
    weak_mask = basin.isin(WEAK_BASINS).values if drop_weak else np.zeros(len(out), bool)
    remove_mask = dead_mask | weak_mask
    dropped = out[remove_mask].copy()
    dropped["drop_reason"] = np.where(basin[remove_mask].isin(DEAD_BASINS), "dead basin", "weak basin")
    out = out[~remove_mask].copy()
    out["rank"] = out["prob_exceeds_breakeven"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("rank").reset_index(drop=True)
    return out, dropped


# ----------------------------- UI -----------------------------
st.markdown("""
<div style="background: #090D73; padding: 2rem 2.2rem; border-radius: 14px;
            margin-bottom: 1.6rem; box-shadow: 0 4px 22px rgba(9,13,115,0.28);
            position: relative; overflow: hidden;">
  <div style="position:absolute; left:0; top:0; width:5px; height:100%; background:#5EC8D8;"></div>
  <div style="position:absolute; top:0; right:0; width:200px; height:100%;
              background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(94,200,216,0.08) 100%);"></div>
  <div style="font-size: .72rem; font-weight: 600; color: #7FB0E8; letter-spacing: 2.5px;
              text-transform: uppercase; margin-bottom: .5rem;">
    Re-Frac Analytics
  </div>
  <div style="font-family: Georgia, 'Times New Roman', serif; font-size: 2.3rem;
              font-weight: 700; color: #FFFFFF; line-height: 1.1;">
    Candidate Screening
  </div>
  <div style="width: 46px; height: 3px; background: #5EC8D8; border-radius: 2px;
              margin: .9rem 0 .8rem 0;"></div>
  <div style="font-size: 1rem; color: #C6D4F0; font-weight: 400; max-width: 640px;
              line-height: 1.5;">
    Basin-aware model that ranks wells by their predicted re-frac uplift and the
    probability of clearing breakeven.
  </div>
</div>
""", unsafe_allow_html=True)

missing_files = []
if not os.path.exists(NATIONAL_PATH):
    missing_files.append("national_model_v2_1.joblib")
if not _find_basin_model_files():
    missing_files.append("basin model files (model_<BASIN>.joblib, in basin_models/ or the repo root)")
if missing_files:
    st.error("Missing model files: " + ", ".join(f"**{m}**" for m in missing_files) +
             ". Place them next to this app.")
    st.stop()

national, basins, _load_errors = load_models()
if _load_errors:
    st.warning("Some basin models could not be loaded and were skipped "
               "(those basins will use the national model):\n\n"
               + "\n".join(f"- {f}: {msg}" for f, msg in _load_errors))

with st.sidebar:
    st.header("Setup")
    st.markdown(f"**National model:** loaded ({national.get('version','v2.1')})")
    st.markdown(f"**Basin models:** {len(basins)} loaded")
    st.markdown(f"**Basins scored by their own model:** {', '.join(sorted(b for b in OWN_MODEL_BASINS if b in basins)) or 'none'}")
    st.markdown(f"**Dead basins (always excluded):** {', '.join(sorted(DEAD_BASINS))}")
    st.markdown(f"**Weak basins (uplift below breakeven):** {', '.join(sorted(WEAK_BASINS))}")
    drop_weak = st.checkbox("Exclude weak basins from screening", value=True,
                            help="Weak basins have some uplift but <=2% of wells clear breakeven. "
                                 "Uncheck to keep them in the results.")
    st.caption("A basin is 'weak' when 2% or fewer of its wells clear breakeven - so there's almost "
               "nothing worth screening.")
    st.markdown("---")
    st.markdown(f"**Economics:** ${REFRAC_COST_USD:,.0f} cost, ${PROFIT_PER_BOE_USD:.0f}/BOE "
                f"-> breakeven {BREAKEVEN_BOE:,.0f} BOE")
    top_n = st.number_input("Top candidates to show", 10, 2000, 100, step=10)
    recent_years = st.slider("Recent re-frac window (years)", 0, 10, 3,
                             help="Wells whose last re-frac is more recent than this are handled "
                                  "per the option below. 0 disables it.")
    recent_action = st.radio("For recently re-fraced wells:", ["Flag only", "Remove from results"],
                             index=0, help="Flag keeps them in the list with a marker; "
                                           "Remove drops them like dead-basin wells.")
    st.markdown("---")
    st.caption("Input needs an ENVBasin column (basin routing). A refrac_date or job_year column "
               "enables the recent-refrac option.")

# reference calibration tables (historical), per basin: how reliable is a given P50 prediction?
_cal_df, _p50c, _actc, _basc = load_calibration_data()
if _cal_df is not None:
    with st.expander("Prediction reliability (calibration table, historical data)"):
        st.caption(f"Across historical wells, how often did each P50 range actually exceed "
                   f"breakeven ({BREAKEVEN_BOE:,.0f} BOE)? Pick a basin, or All.")
        if _basc:
            opts = ["All basins"] + sorted(b for b in _cal_df[_basc].unique()
                                           if (_cal_df[_basc] == b).sum() >= 20)
            pick = st.selectbox("Basin", opts, key="cal_basin")
            sub = _cal_df if pick == "All basins" else _cal_df[_cal_df[_basc] == pick]
        else:
            sub = _cal_df
        tbl, base = calibration_for(sub, _p50c, _actc)
        if tbl is not None:
            st.caption(f"n = {len(sub)} wells | base rate: {base*100:.0f}%")
            st.dataframe(tbl, width='stretch', hide_index=True)
            st.caption("Read: higher predicted P50 -> more reliable. Find the row where "
                       "'actually_exceeded_breakeven' reaches 100% - that's the prediction level "
                       "you can fully trust in this basin.")
        else:
            st.caption("Not enough labelled wells in this basin for a table.")

uploaded = st.file_uploader("Upload wells CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV with one row per well, including an ENVBasin column. "
            "Each well is scored by its basin's best model; dead basins are dropped automatically.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

# drop duplicate column names (keep first) - duplicates break numeric conversion
if df_raw.columns.duplicated().any():
    dups = sorted(set(df_raw.columns[df_raw.columns.duplicated()]))
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
    st.warning("Your file has duplicate column names; keeping the first of each: "
               + ", ".join(dups))

st.success(f"Loaded {len(df_raw):,} wells with {df_raw.shape[1]} columns.")

# ---- renamed-column handling (suggest, user confirms) ----
if "colmap" not in st.session_state:
    st.session_state.colmap = {}
_feats = national["feats"]
missing, extra, crit_missing = check_columns(df_raw, _feats)
if crit_missing:
    st.warning("Some important columns are missing. If your file uses different names for the "
               "same thing, match them below - nothing is renamed until you confirm.")
    name_sug = suggest_matches(crit_missing, extra)
    content_sug = suggest_by_content(crit_missing, extra, df_raw)
    with st.form("colmap_form"):
        chosen = {}
        for miss in crit_missing:
            by_name = name_sug.get(miss, [])
            by_content = [e for e in content_sug.get(miss, []) if e not in by_name]
            ranked_opts = by_name + by_content
            options = ["- (leave missing) -"] + ranked_opts + [e for e in extra if e not in ranked_opts]
            tag = ""
            if by_name:      tag = f"  (best guess by name: {by_name[0]})"
            elif by_content: tag = f"  (guess by data values: {by_content[0]})"
            pick = st.selectbox(f"Model needs {miss} - which of your columns is this?{tag}",
                                options, index=1 if ranked_opts else 0, key=f"map_{miss}")
            if pick != options[0]:
                chosen[pick] = miss
        applied = st.form_submit_button("Apply column matches")
        # RED alert if a TOP5 feature was left unmatched in the form
        top5_unmatched = [m for m in crit_missing if m in TOP5 and m not in chosen.values()]
        if applied and top5_unmatched:
            st.error("PREDICTION NOT VALID - you left a top-5 critical feature without a matching "
                     "column: **" + ", ".join(top5_unmatched) + "**. The model relies heavily on "
                     "these; results will be unreliable until they are matched to a real column.")
    if applied:
        st.session_state.colmap = chosen
        st.success("Matched: " + (", ".join(f"{k} -> {v}" for k, v in chosen.items()) or "nothing"))
    if st.session_state.colmap:
        df_raw = df_raw.rename(columns=st.session_state.colmap)
        missing, extra, crit_missing = check_columns(df_raw, _feats)
        if not crit_missing:
            st.success("All important features present after matching.")

# RED alert if any TOP5 feature is still absent from the data (after any matching)
_top5_missing = [c for c in TOP5 if c not in df_raw.columns]
st.session_state.top5_missing = _top5_missing
if _top5_missing:
    st.error("PREDICTION NOT VALID - these top-5 critical features are missing from the data: **"
             + ", ".join(_top5_missing) + "**. The model depends on them, so any scores produced "
             "will be filled with placeholder values and should not be trusted. Add or match these "
             "columns before relying on the results.")

# heads-up about dead-basin wells BEFORE scoring
if "ENVBasin" in df_raw.columns:
    _b = df_raw["ENVBasin"].astype(str).str.upper().str.strip()
    _excl = DEAD_BASINS | WEAK_BASINS
    _excl_counts = _b[_b.isin(_excl)].value_counts()
    if len(_excl_counts):
        _lines = "\n".join(f"- {name}: {n} well(s)" for name, n in _excl_counts.items())
        st.warning(f"WARNING - {int(_excl_counts.sum())} well(s) are in dead/weak basins and may be "
                   f"DROPPED from the results (uplift there rarely clears breakeven):\n\n{_lines}")

if "ENVBasin" not in df_raw.columns:
    st.warning("No ENVBasin column found - every well will use the national model, and dead "
               "basins can't be excluded. Add an ENVBasin column for full hybrid routing.")

if st.button("Run screening", type="primary"):
    with st.spinner("Scoring wells with per-basin models..."):
        ranked, dropped = hybrid_score(df_raw, national, basins, drop_weak=drop_weak)
    # recent-refrac handling: flag or remove, per the toggle
    if recent_years > 0:
        recent_mask = flag_recent_refracs(ranked, recent_years).values
        n_recent = int(recent_mask.sum())
        if recent_action == "Remove from results":
            if n_recent:
                recent_removed = ranked[recent_mask].copy()
                ranked = ranked[~recent_mask].copy()
                ranked["rank"] = ranked["prob_exceeds_breakeven"].rank(ascending=False, method="first").astype(int)
                ranked = ranked.sort_values("rank").reset_index(drop=True)
                st.session_state.recent_removed = recent_removed
            else:
                st.session_state.recent_removed = None
            st.session_state.recent_flagged_n = n_recent
        else:  # Flag only
            ranked["recently_refraced"] = recent_mask
            st.session_state.recent_removed = None
            st.session_state.recent_flagged_n = n_recent
    else:
        st.session_state.recent_removed = None
        st.session_state.recent_flagged_n = 0
    # calibration probability: historical success rate of each well's basin + P50 bucket
    cp = calib_prob_for_wells(ranked)
    if cp is not None:
        ranked["calib_prob_from_table"] = cp.round(2)
    # traffic-light signals: P50 size, classifier probability, calibration-table probability.
    # green = strong, yellow = middling, red = weak, grey = no data. One glance covers all three.
    def _light_p50(x):
        if pd.isna(x): return "\u26AA"
        return "\U0001F534" if x < BREAKEVEN_BOE else ("\U0001F7E1" if x < 2 * BREAKEVEN_BOE else "\U0001F7E2")
    def _light_prob(x):
        if pd.isna(x): return "\u26AA"
        return "\U0001F534" if x < 0.4 else ("\U0001F7E1" if x < 0.7 else "\U0001F7E2")
    _p50 = pd.to_numeric(ranked.get("pred_central_p50"), errors="coerce")
    _clf = pd.to_numeric(ranked.get("prob_exceeds_breakeven"), errors="coerce")
    _cal = pd.to_numeric(ranked.get("calib_prob_from_table"), errors="coerce") \
           if "calib_prob_from_table" in ranked.columns else pd.Series(np.nan, index=ranked.index)
    ranked["signals"] = [f"{_light_p50(a)}{_light_prob(b)}{_light_prob(c)}"
                         for a, b, c in zip(_p50, _clf, _cal)]
    # persist results so filter widgets don't wipe them on rerun
    st.session_state.ranked = ranked
    st.session_state.dropped = dropped
    st.session_state.recent_action_used = recent_action if recent_years > 0 else None
    st.session_state.recent_years_used = recent_years

# ---- results (from session_state, survives widget reruns) ----
if st.session_state.get("ranked") is not None:
    ranked = st.session_state.ranked
    dropped = st.session_state.dropped
    n_recent = st.session_state.get("recent_flagged_n", 0)
    recent_removed = st.session_state.get("recent_removed")
    _raction = st.session_state.get("recent_action_used")
    _ryears = st.session_state.get("recent_years_used", 0)
    if _raction == "Remove from results":
        if n_recent:
            st.warning(f"WARNING - removed {n_recent} well(s) re-fraced within the last "
                       f"{_ryears} years (a repeat re-frac may not make sense yet).")
            if recent_removed is not None:
                with st.expander(f"See the {n_recent} removed (recently re-fraced) wells"):
                    rshow = [c for c in ["well_API14","API14","ENVBasin","refrac_date","job_year"] if c in recent_removed.columns]
                    st.dataframe(recent_removed[rshow], width='stretch', hide_index=True)
        else:
            st.caption(f"No wells re-fraced within the last {_ryears} years.")
    elif _raction == "Flag only":
        if n_recent:
            st.warning(f"WARNING - {n_recent} well(s) were re-fraced within the last {_ryears} "
                       f"years and are flagged (recently_refraced = True) but kept in the list. "
                       f"A repeat re-frac may not make sense yet - review before acting.")
        else:
            st.caption(f"No wells re-fraced within the last {_ryears} years.")
    if len(dropped):
        counts = dropped.groupby(["drop_reason","ENVBasin"]).size()
        lines = "\n".join(f"- {basin} ({reason}): {n} well(s)" for (reason, basin), n in counts.items())
        st.warning(f"WARNING - dropped {len(dropped)} well(s) in dead/weak basins "
                   f"(excluded from the ranking below):\n\n{lines}")
        with st.expander(f"See the {len(dropped)} dropped wells"):
            dshow = [c for c in ["well_API14","API14","ENVBasin","drop_reason","operator"] if c in dropped.columns]
            st.dataframe(dropped[dshow], width='stretch', hide_index=True)
            st.download_button("Download dropped wells", dropped.to_csv(index=False).encode("utf-8"),
                               file_name="dropped_wells.csv", mime="text/csv")
    # RED banner repeated here so it can't be missed next to the results
    _t5m = st.session_state.get("top5_missing", [])
    if _t5m:
        st.error("PREDICTION NOT VALID - the data is missing top-5 critical feature(s): **"
                 + ", ".join(_t5m) + "**. These scores are filled with placeholder values and "
                 "should not be trusted. Add or match these columns, then re-run.")

    # summary metric cards
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    n_scored = len(ranked)
    n_dropped = len(dropped)
    n_recent = int(ranked["recently_refraced"].sum()) if "recently_refraced" in ranked.columns else 0
    n_profit = int((ranked["expected_profit_USD"] > 0).sum()) if "expected_profit_USD" in ranked.columns else 0
    m1.metric("Wells scored", f"{n_scored:,}")
    m2.metric("Profitable (P50)", f"{n_profit:,}")
    m3.metric("Excluded (dead/weak)", f"{n_dropped:,}")
    m4.metric("Recently re-fraced", f"{n_recent:,}")

    st.subheader("Top candidates")

    # ---- result filters ----
    fc1, fc2, fc3 = st.columns([1.2, 1.2, 1.6])
    with fc1:
        if "ENVBasin" in ranked.columns:
            basin_opts = ["All basins"] + sorted(ranked["ENVBasin"].dropna().unique())
            fbasin = st.selectbox("Basin", basin_opts, key="result_basin")
        else:
            fbasin = "All basins"
    with fc2:
        op_col = next((c for c in ["operator", "Operator", "ENVOperator"] if c in ranked.columns), None)
        if op_col:
            op_opts = ["All operators"] + sorted(ranked[op_col].dropna().astype(str).unique())
            fop = st.selectbox("Operator", op_opts, key="result_operator")
        else:
            fop = "All operators"
    with fc3:
        api_col = next((c for c in ["well_API14", "API14", "api10"] if c in ranked.columns), None)
        api_query = st.text_input("Search well API", "", key="api_search",
                                  placeholder="e.g. 3305304904") if api_col else ""

    min_prob = st.slider("Min probability of exceeding breakeven", 0.0, 1.0, 0.0, 0.05,
                         help="Hide wells below this probability. 0 shows everything.")
    only_profitable = st.checkbox("Only profitable wells (expected profit > 0)", value=False)

    sort_by = st.radio("Sort by",
                       ["Probability of exceeding breakeven", "P50 (predicted uplift)",
                        "Expected profit", "Risk-adjusted profit"],
                       horizontal=True, index=0,
                       help="Probability ranks real successes best (validated 0.99 vs 0.91 for P50, "
                            "and it wins in every basin), so it's the default. Switch if you prefer.")

    view = ranked
    if fbasin != "All basins":
        view = view[view["ENVBasin"] == fbasin]
    if op_col and fop != "All operators":
        view = view[view[op_col].astype(str) == fop]
    if api_col and api_query.strip():
        view = view[view[api_col].astype(str).str.contains(api_query.strip(), na=False)]
    if min_prob > 0 and "prob_exceeds_breakeven" in view.columns:
        view = view[view["prob_exceeds_breakeven"] >= min_prob]
    if only_profitable and "expected_profit_USD" in view.columns:
        view = view[view["expected_profit_USD"] > 0]

    # apply the chosen sort
    sort_col = {"Probability of exceeding breakeven": "prob_exceeds_breakeven",
                "P50 (predicted uplift)": "pred_central_p50",
                "Expected profit": "expected_profit_USD",
                "Risk-adjusted profit": "risk_adjusted_profit_USD"}.get(sort_by)
    if sort_col and sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=False).reset_index(drop=True)
        view["rank"] = np.arange(1, len(view) + 1)

    show = [c for c in ["rank", "well_API14", "API14", "ENVBasin", "model_used", "signals",
                        "pred_central_p50", "calib_prob_from_table",
                        "prob_exceeds_breakeven", "expected_profit_USD", "risk_adjusted_profit_USD",
                        "relative_uncertainty", "recently_refraced"]
            if c in view.columns]
    top = view.head(int(top_n))
    st.caption(f"Showing {min(top_n, len(view))} of {len(view)} wells")

    # color highlighting: green = high probability / profit, red = low
    sty = top[show].style
    try:
        if "prob_exceeds_breakeven" in show:
            sty = sty.background_gradient(subset=["prob_exceeds_breakeven"], cmap="RdYlGn", vmin=0, vmax=1)
        if "calib_prob_from_table" in show:
            sty = sty.background_gradient(subset=["calib_prob_from_table"], cmap="RdYlGn", vmin=0, vmax=1)
        if "expected_profit_USD" in show:
            sty = sty.background_gradient(subset=["expected_profit_USD"], cmap="RdYlGn")
        if "risk_adjusted_profit_USD" in show:
            sty = sty.background_gradient(subset=["risk_adjusted_profit_USD"], cmap="RdYlGn")
        if "relative_uncertainty" in show:
            sty = sty.background_gradient(subset=["relative_uncertainty"], cmap="RdYlGn_r")
        fmt = {}
        if "pred_central_p50" in show:       fmt["pred_central_p50"] = "{:,.0f}"
        if "expected_profit_USD" in show:    fmt["expected_profit_USD"] = "${:,.0f}"
        if "risk_adjusted_profit_USD" in show: fmt["risk_adjusted_profit_USD"] = "${:,.0f}"
        if "prob_exceeds_breakeven" in show: fmt["prob_exceeds_breakeven"] = "{:.0%}"
        if "calib_prob_from_table" in show:   fmt["calib_prob_from_table"] = "{:.0%}"
        if "relative_uncertainty" in show:   fmt["relative_uncertainty"] = "{:.2f}"
        sty = sty.format(fmt)
        st.dataframe(sty, width='stretch', hide_index=True)
    except Exception:
        st.dataframe(top[show], width='stretch', hide_index=True)
    st.caption("Model routing: " + " | ".join(
        f"{k}: {v}" for k, v in ranked["model_used"].value_counts().items()))
    st.download_button("Download full ranked CSV",
                       ranked.to_csv(index=False).encode("utf-8"),
                       file_name="ranked_wells_hybrid.csv", mime="text/csv")

    # ---- 3x3 matrix: P50 size  x  probability of exceeding breakeven ----
    if {"pred_central_p50", "prob_exceeds_breakeven"}.issubset(view.columns) and len(view) >= 3:
        st.subheader("Size vs reliability matrix (3x3)")
        st.caption("Columns = predicted size (P50, split into low / mid / high thirds). "
                   "Rows = probability of clearing breakeven, in three bands. "
                   "The strongest wells are top-right: big predicted uplift and high probability.")

        v = view.copy()
        p50 = pd.to_numeric(v["pred_central_p50"], errors="coerce")
        clf = pd.to_numeric(v["prob_exceeds_breakeven"], errors="coerce")

        # size thirds by P50 terciles
        try:
            q33, q66 = p50.quantile([1/3, 2/3])
        except Exception:
            q33 = q66 = p50.median()
        def size_bucket(x):
            if pd.isna(x): return "mid"
            return "low" if x <= q33 else ("high" if x > q66 else "mid")

        # probability bands - thresholds chosen from the calibration data (see note below):
        #   < 40%  -> low   (historically these almost never clear breakeven)
        #   40-70% -> mid   (they clear it often, but it isn't a sure thing)
        #   >= 70% -> high  (they clear it almost every time)
        PROB_LOW, PROB_HIGH = 0.40, 0.70
        def prob_bucket(p):
            if pd.isna(p): return "mid"
            return "low" if p < PROB_LOW else ("high" if p >= PROB_HIGH else "mid")

        v["_size"] = p50.map(size_bucket)
        v["_prob"] = clf.map(prob_bucket)

        sizes = ["low", "mid", "high"]
        rows  = ["high", "mid", "low"]          # high probability on top
        size_hdr = {"low": "Low P50", "mid": "Mid P50", "high": "High P50"}
        row_hdr = {"high": "High prob (>=70%)", "mid": "Mid prob (40-70%)",
                   "low": "Low prob (<40%)"}
        # background tint per row (green good, amber caution, grey skip)
        row_bg = {"high": "#EAF6EA", "mid": "#FBF5E7", "low": "#F1F1F3"}

        counts = {(s, r): int(((v["_size"] == s) & (v["_prob"] == r)).sum())
                  for s in sizes for r in rows}

        # render as an HTML grid
        html = ['<div style="display:grid; grid-template-columns:170px 1fr 1fr 1fr; gap:6px; align-items:stretch;">']
        html.append('<div></div>')
        for s in sizes:
            html.append(f'<div style="text-align:center; font-weight:700; color:#090D73; '
                        f'padding:.3rem;">{size_hdr[s]}</div>')
        for r in rows:
            html.append(f'<div style="font-weight:700; color:#090D73; display:flex; '
                        f'align-items:center; padding:.3rem;">{row_hdr[r]}</div>')
            for s in sizes:
                n = counts[(s, r)]
                strong = (r == "high" and s == "high")
                border = "2px solid #0F766E" if strong else "1px solid #D8DEE8"
                html.append(
                    f'<div style="background:{row_bg[r]}; border:{border}; border-radius:8px; '
                    f'padding:1rem .5rem; text-align:center;">'
                    f'<div style="font-size:1.5rem; font-weight:700; color:#1F2A2E;">{n}</div>'
                    f'<div style="font-size:.72rem; color:#5F6E73;">wells</div></div>')
        html.append('</div>')
        st.markdown("".join(html), unsafe_allow_html=True)
        st.caption("High P50 + high probability (top-right) = strongest candidates. "
                   "High P50 + low probability = big predicted uplift the model doubts; treat as a "
                   "risky bet and look closer before acting.")

        # explain WHY the probability thresholds are 40% and 70%
        with st.expander("Why the probability bands are 40% and 70%"):
            st.markdown("""
The thresholds come from where the model's predicted probability actually changes real-world behaviour, measured on historical
wells with known outcomes. Grouping those wells by predicted probability and checking how often they really cleared breakeven gives:

| Predicted probability | Actually cleared breakeven |
|---|---|
| 0-20% | 0% |
| 20-40% | 15% |
| 40-60% | 74% |
| 60-80% | 86% |
| 80-100% | 99% |

One extra reason these bands are conservative: the model tends to understate
probability (a well it calls 45% really succeeds about 68% of the time), so a well landing in
the high band is safer than the label suggests.
""")

        # add matrix labels as downloadable columns
        v["size_bucket"] = v["_size"]
        v["prob_band"] = v["_prob"]

        # let the user inspect which wells sit in any cell
        st.markdown("**See the wells in a cell**")
        cc1, cc2 = st.columns(2)
        with cc1:
            pick_size = st.selectbox("Size (P50)", ["High P50", "Mid P50", "Low P50"],
                                     key="mx_size")
        with cc2:
            pick_row = st.selectbox("Probability of exceeding breakeven",
                                    ["High prob (>=70%)", "Mid prob (40-70%)", "Low prob (<40%)"],
                                    key="mx_row")
        size_key = {"Low P50": "low", "Mid P50": "mid", "High P50": "high"}[pick_size]
        row_key  = {"High prob (>=70%)": "high", "Mid prob (40-70%)": "mid",
                    "Low prob (<40%)": "low"}[pick_row]
        cell = v[(v["_size"] == size_key) & (v["_prob"] == row_key)]
        st.caption(f"{len(cell)} well(s) in '{pick_size} x {pick_row}'")
        if len(cell):
            cell_cols = [c for c in ["rank", "well_API14", "API14", "ENVBasin", "model_used",
                                     "signals", "pred_central_p50", "calib_prob_from_table",
                                     "prob_exceeds_breakeven", "expected_profit_USD",
                                     "risk_adjusted_profit_USD"] if c in cell.columns]
            st.dataframe(cell[cell_cols], width='stretch', hide_index=True)
            st.download_button(f"Download these {len(cell)} wells",
                               cell.to_csv(index=False).encode("utf-8"),
                               file_name=f"matrix_{size_key}_{row_key}.csv", mime="text/csv",
                               key="mx_dl")

        view = v.drop(columns=["_size", "_prob"])

    # ---- well map (if coordinates are present) ----
    lat_col = next((c for c in ["lat", "Latitude", "SurfaceLatitude"] if c in view.columns), None)
    lon_col = next((c for c in ["lon", "Longitude", "SurfaceLongitude"] if c in view.columns), None)
    if lat_col and lon_col:
        st.subheader("Well map")
        mp = view.loc[:, ~view.columns.duplicated()].copy()
        mp["_lat"] = pd.to_numeric(mp[lat_col], errors="coerce")
        mp["_lon"] = pd.to_numeric(mp[lon_col], errors="coerce")
        mp = mp.dropna(subset=["_lat", "_lon"])
        if len(mp):
            st.caption(f"{len(mp)} wells shown. Green = higher probability of exceeding breakeven, "
                       "red = lower.")
            try:
                import pydeck as pdk
                p = mp["prob_exceeds_breakeven"].fillna(0).clip(0, 1) if "prob_exceeds_breakeven" in mp.columns else 0.5
                mp["_r"] = ((1 - p) * 220).astype(int)          # red channel high when prob low
                mp["_g"] = (p * 180 + 40).astype(int)           # green channel high when prob high
                mp["_api"] = mp[next((c for c in ["well_API14","API14","api10"] if c in mp.columns), mp.columns[0])].astype(str)
                mp["_prob_txt"] = (mp["prob_exceeds_breakeven"]*100).round(0).astype("Int64").astype(str)+"%" \
                                  if "prob_exceeds_breakeven" in mp.columns else ""
                layer = pdk.Layer(
                    "ScatterplotLayer", data=mp,
                    get_position=["_lon", "_lat"],
                    get_fill_color=["_r", "_g", 60, 160],
                    get_radius=6000, pickable=True, radius_min_pixels=3, radius_max_pixels=12)
                view_state = pdk.ViewState(latitude=float(mp["_lat"].mean()),
                                           longitude=float(mp["_lon"].mean()), zoom=4.2)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                                         tooltip={"text": "API {_api}\\nBasin {ENVBasin}\\nProb {_prob_txt}"}))
            except Exception:
                # fallback: simple built-in map (no color) if pydeck unavailable
                st.map(mp.rename(columns={"_lat": "lat", "_lon": "lon"})[["lat", "lon"]])
        else:
            st.caption("No valid coordinates to map in the current selection.")

    # ---- portfolio calculator ----
    st.subheader("Portfolio calculator")
    st.caption("If you re-frac the top N wells from the filtered list above, what does the "
               "portfolio look like?")
    pc1, pc2 = st.columns([1, 2.4])
    with pc1:
        port_n = st.number_input("Re-frac the top", 1, max(1, len(view)), min(20, max(1, len(view))),
                                 step=5, key="port_n")
    port = view.head(int(port_n))
    total_cost = int(port_n) * REFRAC_COST_USD
    exp_profit = float(port["expected_profit_USD"].sum()) if "expected_profit_USD" in port.columns else 0.0
    exp_uplift = float(port["pred_central_p50"].sum()) if "pred_central_p50" in port.columns else 0.0
    n_confident = int((port["prob_exceeds_breakeven"] >= 0.5).sum()) if "prob_exceeds_breakeven" in port.columns else 0
    with pc2:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total cost", f"${total_cost/1e6:,.1f}M")
        k2.metric("Expected profit", f"${exp_profit/1e6:,.1f}M")
        k3.metric("Expected uplift", f"{exp_uplift:,.0f} BOE")
        k4.metric("Wells with prob >= 50%", f"{n_confident} / {int(port_n)}")
    if "expected_profit_USD" in port.columns and port_n > 0:
        roi = exp_profit / total_cost * 100 if total_cost else 0
        st.caption(f"Expected portfolio ROI: {roi:,.0f}%  (expected profit / total cost; "
                   f"based on P50 central estimates and ${REFRAC_COST_USD:,.0f} per re-frac)")

    # column legend
    with st.expander("What do these columns mean?"):
        st.markdown("""
- **signals** - three traffic lights at a glance: P50 size | classifier probability | calibration-table probability. Green = strong, yellow = middling, red = weak, grey = no data. Three greens = the strongest, most agreed-upon candidates.
- **pred_central_p50** - the model's central estimate of uplift (barrels of oil equivalent).
- **calib_prob_from_table** - historical success rate for wells in the same basin and P50 range (from the calibration table). Blank if that basin/range has too few historical wells.
- **prob_exceeds_breakeven** - probability the well clears the breakeven threshold (10,000 BOE).
- **expected_profit_USD** - P50 x $40 per BOE, minus the $400,000 re-frac cost.
- **risk_adjusted_profit_USD** - the same profit but weighted by the probability of clearing breakeven (probability x P50 x $40 - $400,000). Discounts big-but-unlikely wells; use it to compare a large risky well against a smaller reliable one.
- **relative_uncertainty** - interval width divided by P50; smaller = more confident. Comparable across wells of any size.
- **model_used** - which model scored the well: the national model, or a basin's own model.
- **recently_refraced** - flagged if the last re-frac was within the chosen window (a repeat may not make sense yet).
""")
