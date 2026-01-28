# pages/4_Results.py
# Results — Fairness Judgment, Evidence & Verdict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Results — Fairness Assessment & Verdict")
st.markdown(
    """
    <style>
    /* =========================
       Global app surface
       ========================= */
    .stApp {
        background-color: #f5f7fa;
        font-family: "Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
    }

    /* =========================
       Page title
       ========================= */
    h1 {
        font-size: 2.1rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-bottom: 0.25rem;
    }

    /* =========================
       Section headings
       ========================= */
    h2 {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.8rem;
        margin-bottom: 0.4rem;
    }

    h3 {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 1.4rem;
        margin-bottom: 0.3rem;
    }

    /* =========================
       Caption / helper text
       ========================= */
    .stCaption {
        font-size: 0.88rem;
        color: #6b7280;
        line-height: 1.4;
    }

    /* =========================
       Card containers
       (st.container(border=True))
       ========================= */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    /* =========================
       Inputs & text areas
       ========================= */
    textarea, input {
        font-size: 0.95rem !important;
        line-height: 1.45 !important;
    }

    textarea {
        padding: 0.6rem !important;
    }

    /* =========================
       Labels
       ========================= */
    label {
        font-weight: 500 !important;
        color: #111827 !important;
    }

    /* =========================
       Tabs
       ========================= */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.55rem 0.9rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #1f2937;
        font-weight: 600;
    }

    /* =========================
       Status panel badges
       ========================= */
    .status-complete {
        color: #065f46;
        font-weight: 600;
    }

    .status-pending {
        color: #92400e;
        font-weight: 600;
    }

    /* =========================
       Primary action button
       ========================= */
    button[kind="primary"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* =========================
       Typography refinement
       ========================= */

    /* Body text */
    html, body, [class*="css"] {
        font-size: 15px;
        line-height: 1.55;
        letter-spacing: -0.005em;
    }

    /* Main page title */
    h1 {
        font-size: 2.15rem;
        line-height: 1.25;
        letter-spacing: -0.015em;
    }

    /* Section headers */
    h2 {
        font-size: 1.45rem;
        line-height: 1.35;
        letter-spacing: -0.01em;
    }

    h3 {
        font-size: 1.15rem;
        line-height: 1.35;
        letter-spacing: -0.005em;
    }

    /* Labels above inputs */
    label {
        font-size: 0.92rem !important;
        letter-spacing: -0.005em;
    }

    /* Text areas & inputs — make them feel less "form-like" */
    textarea, input {
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
        border-radius: 6px !important;
    }

    /* Reduce visual density inside text areas */
    textarea {
        padding-top: 0.55rem !important;
        padding-bottom: 0.55rem !important;
    }

    /* Help text (tooltips & captions) */
    .stCaption, .stTooltipContent {
        font-size: 0.85rem;
        line-height: 1.45;
        color: #6b7280;
    }

    /* =========================
       Make long-form text readable
       ========================= */

    /* Paragraphs inside markdown */
    p {
        max-width: 68ch;
    }

    /* Reduce harsh contrast for long reading */
    p, li {
        color: #1f2937;
    }

    /* =========================
       Tables (metrics) polish
       ========================= */
    table {
        font-size: 0.9rem;
    }

    thead tr th {
        font-weight: 600;
        background-color: #f9fafb;
    }

    /* =========================
       Subtle divider refinement
       ========================= */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.8rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Preconditions
# --------------------------------------------------
inference = st.session_state.get("inference")
if not inference or not inference.get("completed"):
    st.warning("Inference results not available. Please complete the Inference step.")
    st.stop()

df = inference["results_df"].copy()

# --------------------------------------------------
# Explicit fairness metric definitions (DECLARED)
# --------------------------------------------------
FAIRNESS_METRICS = [
    ("Statistical Parity Difference", 0.0),
    ("Disparate Impact", 1.0),
    ("Average Odds Difference", 0.0),
    ("Equal Opportunity Difference", 0.0),
    ("Error Rate Difference", 0.0),
]

AVAILABLE_FAIRNESS_METRICS = [
    (m, ideal) for m, ideal in FAIRNESS_METRICS if m in df.columns
]

# --------------------------------------------------
# Compute Bias Index (BI) and Fairness Score (FS)
# --------------------------------------------------
def compute_BI_FS(row: pd.Series):
    diffs = []
    metric_vals = {}

    for name, ideal in AVAILABLE_FAIRNESS_METRICS:
        val = row.get(name, np.nan)
        if pd.notna(val):
            metric_vals[name] = float(val)
            diffs.append((float(val) - float(ideal)) ** 2)

    if not diffs:
        return np.nan, np.nan, metric_vals

    bi = float(np.sqrt(np.mean(diffs)))
    fs = float(1.0 - bi)
    return bi, fs, metric_vals

if "BI" not in df.columns or "FS" not in df.columns:
    BI_vals, FS_vals, metric_maps = [], [], []
    for _, r in df.iterrows():
        bi, fs, mvals = compute_BI_FS(r)
        BI_vals.append(bi)
        FS_vals.append(fs)
        metric_maps.append(mvals)

    df["BI"] = BI_vals
    df["FS"] = FS_vals
    df["_fairness_metric_map"] = metric_maps

# --------------------------------------------------
# Model selection
# --------------------------------------------------
st.markdown("### Model under evaluation")
model = st.selectbox("Select model", df["Model"].tolist())
row = df[df["Model"] == model].iloc[0]

FS_val = row["FS"]
BI_val = row["BI"]
fairness_metric_map = row["_fairness_metric_map"]

# --------------------------------------------------
# Verdict rules (EXPLICIT)
# --------------------------------------------------
FS_PASS = 0.85
FS_CONDITIONAL = 0.70
BI_MAX = 0.15

if pd.isna(FS_val) or pd.isna(BI_val):
    verdict = "INSUFFICIENT DATA"
    verdict_color = "#9E9E9E"
elif FS_val >= FS_PASS and BI_val <= BI_MAX:
    verdict = "PASS"
    verdict_color = "#2E7D32"
elif FS_val >= FS_CONDITIONAL:
    verdict = "CONDITIONAL"
    verdict_color = "#ED6C02"
else:
    verdict = "FAIL"
    verdict_color = "#C62828"

# --------------------------------------------------
# Decision cards (VISUAL UPGRADE)
# --------------------------------------------------
st.markdown("### Decision Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    label="Fairness Score (FS)",
    value=f"{FS_val:.3f}" if pd.notna(FS_val) else "N/A",
    help="FS = 1 − Bias Index (higher is better)"
)

c2.metric(
    label="Bias Index (BI)",
    value=f"{BI_val:.3f}" if pd.notna(BI_val) else "N/A",
    help="Root-mean-square deviation of fairness metrics from ideal values"
)

c3.markdown(
    f"""
    <div style="
        padding: 1.1rem;
        border-radius: 0.75rem;
        background-color: {verdict_color};
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
    ">
        {verdict}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Fairness Metrics Summary Plot (EVIDENCE LAYER)
# --------------------------------------------------
st.markdown("### Fairness Metrics — Component View")

if fairness_metric_map:
    metric_names = list(fairness_metric_map.keys())
    values = [fairness_metric_map[m] for m in metric_names]
    ideals = [ideal for m, ideal in AVAILABLE_FAIRNESS_METRICS if m in metric_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(metric_names, values)

    # Ideal reference lines
    for m, ideal in AVAILABLE_FAIRNESS_METRICS:
        if m in metric_names:
            ax.axhline(
                y=ideal,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                color="black"
            )

    ax.set_ylabel("Metric value")
    ax.set_title("Individual Fairness Metrics for Selected Model")
    ax.set_xticklabels(metric_names, rotation=30, ha="right")

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Dashed reference lines indicate ideal values. "
        "These metrics collectively contribute to the Bias Index (BI)."
    )
else:
    st.info("No fairness metrics available for visualization.")

# --------------------------------------------------
# Decision rationale (TEXTUAL, AUDIT-FRIENDLY)
# --------------------------------------------------
st.markdown("### Decision Rationale")

st.markdown(
    f"""
- **Bias Index (BI)** is computed as the root-mean-square deviation of the selected
  fairness metrics from their ideal values.
- **Fairness Score (FS)** is derived as **FS = 1 − BI**.

**Decision thresholds applied**:
- FS ≥ {FS_PASS} and BI ≤ {BI_MAX} → **PASS**
- FS ≥ {FS_CONDITIONAL} → **CONDITIONAL**
- Otherwise → **FAIL**

**Final verdict for `{model}`**: **{verdict}**
"""
)

# --------------------------------------------------
# Persist results for Final Report
# --------------------------------------------------
st.session_state["results"] = {
    "selected_model": model,
    "FS": FS_val,
    "BI": BI_val,
    "verdict": verdict,
    "fairness_metrics": fairness_metric_map,
    "thresholds": {
        "FS_pass": FS_PASS,
        "FS_conditional": FS_CONDITIONAL,
        "BI_max": BI_MAX,
    },
}

st.success("Fairness verdict recorded and ready for final report.")
