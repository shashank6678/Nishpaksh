# pages/3_Inference.py
# Inference & Fairness Evaluation — STORY-DRIVEN + CONTRACT-SAFE

import streamlit as st
import pandas as pd
import numpy as np


import hashlib

from utils.two_class_metrics import GroupMetrics, FairnessMetrics
from utils.viz_utils import (
    _as01,
    plot_bar_single_metric,
    plot_line_single_metric,
    plot_fairness_error_bars,
    plot_by_group_bars,
    plot_models_groups_heatmap,
    plot_disparity_in_performance,
    plot_group_error_panel,
    plot_fairness_accuracy_scatter,
)

# --------------------------------------------------
# Page setup (UNCHANGED)
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Inference & Fairness Evaluation")

# --------------------------------------------------
# Preconditions
# --------------------------------------------------
data = st.session_state.get("uploaded_data")
label_col = st.session_state.get("ground_truth_col")
sensitive_col = st.session_state.get("sensitive_col")
privileged_value = st.session_state.get("privileged_value")

if not isinstance(data, pd.DataFrame) or data.empty:
    st.warning("No dataset available. Complete earlier steps first.")
    st.stop()

if None in (label_col, sensitive_col, privileged_value):
    st.error("Ground truth, sensitive attribute, and privileged group must be set.")
    st.stop()

# --------------------------------------------------
# Detect prediction columns
# --------------------------------------------------
pred_cols = [
    c for c in data.columns
    if c not in {label_col, sensitive_col}
    and set(pd.Series(data[c]).dropna().unique()).issubset({0, 1})
]

if not pred_cols:
    st.error("No prediction columns detected.")
    st.stop()

# --------------------------------------------------
# Positive class selectors (UNCHANGED)
# --------------------------------------------------
def _guess_positive(vals):
    for v in vals:
        if str(v).lower() in {"1", "true", "yes", "approved", "positive"}:
            return v
    return sorted(vals)[-1]

label_vals = data[label_col].astype(str).unique().tolist()
pred_vals = sorted({str(v) for c in pred_cols for v in data[c].dropna().unique()})

c1, c2 = st.columns(2)
POS_TRUE = c1.selectbox(
    "Positive class (ground truth)",
    label_vals,
    index=label_vals.index(_guess_positive(label_vals)),
)
POS_PRED = c2.selectbox(
    "Positive class (predictions)",
    pred_vals,
    index=pred_vals.index(_guess_positive(pred_vals)),
)

# --------------------------------------------------
# Deterministic computation fingerprint
# --------------------------------------------------
def make_compute_key():
    h = hashlib.sha256()
    h.update(str(label_col).encode())
    h.update(str(sensitive_col).encode())
    h.update(str(privileged_value).encode())
    h.update(str(POS_TRUE).encode())
    h.update(str(POS_PRED).encode())
    h.update(",".join(pred_cols).encode())
    h.update(str(data.shape).encode())
    return h.hexdigest()

compute_key = make_compute_key()

# --------------------------------------------------
# Heavy computation (PURE FUNCTION, NO CACHE)
# --------------------------------------------------
def compute_all_metrics(
    df, label_col, sensitive_col, pred_cols,
    privileged_value, pos_true, pos_pred, B=20
):
    y_true = _as01(df[label_col].values, positive=pos_true)
    sens_arr = df[sensitive_col].astype(str).values

    rows = []
    fairness_bootstrap = {}

    for col in pred_cols:
        y_pred = _as01(df[col].values, positive=pos_pred)

        perf = GroupMetrics(y_true, y_pred).get_all()
        fair = FairnessMetrics(
            y_true, y_pred, sens_arr,
            privileged_value=privileged_value
        ).get_all()

        rows.append({"Model": col, **perf, **fair})

        for m in fair.keys():
            fairness_bootstrap.setdefault(m, {})[col] = []
            for _ in range(B):
                idx = np.random.choice(len(y_true), len(y_true), replace=True)
                fb = FairnessMetrics(
                    y_true[idx], y_pred[idx], sens_arr[idx],
                    privileged_value=privileged_value
                ).get_all()
                fairness_bootstrap[m][col].append(fb.get(m))

    return pd.DataFrame(rows), fairness_bootstrap, y_true, sens_arr

# --------------------------------------------------
# Compute trigger (ARCHITECTURALLY FIXED)
# --------------------------------------------------
if st.button("Compute metrics"):
    with st.spinner("Computing metrics…"):
        results_df, fairness_bootstrap, y_true, sens_arr = compute_all_metrics(
            data, label_col, sensitive_col, pred_cols,
            privileged_value, POS_TRUE, POS_PRED
        )

        def safe_table(df, cols):
            keep = [c for c in cols if c in df.columns]
            return df[["Model"] + keep]

        PERFORMANCE_COLS = [
            "TP", "TN", "FP", "FN",
            "Accuracy", "TPR (Recall)", "TNR",
            "FPR", "FNR", "Precision (PPV)",
            "NPV", "FDR", "FOR", "F1"
        ]

        FAIRNESS_COLS = [
            "Statistical Parity Difference",
            "Disparate Impact",
            "Average Odds Difference",
            "Error Rate Difference",
            "Equal Opportunity Difference",
            "Equalized Odds",
            "Thiel Index",
        ]

        # 🔒 CONTRACT PRESERVED
        st.session_state["inference"] = {
            "completed": True,
            "compute_key": compute_key,
            "results_df": results_df,
            "fairness_bootstrap": fairness_bootstrap,
            "y_true": y_true,
            "sensitive_attr": sens_arr,
            "positive_class": {
                "ground_truth": POS_TRUE,
                "prediction": POS_PRED,
            },
            "tables": {
                "performance": safe_table(results_df, PERFORMANCE_COLS),
                "fairness": safe_table(results_df, FAIRNESS_COLS),
            },
        }

        st.session_state["metrics_ready"] = True

# --------------------------------------------------
# Guard: freeze UI until computed
# --------------------------------------------------
if not st.session_state.get("metrics_ready"):
    st.stop()

# --------------------------------------------------
# Unpack (READ-ONLY, SAFE)
# --------------------------------------------------
inference = st.session_state["inference"]

# Recompute protection
if inference.get("compute_key") != compute_key:
    st.warning("Inputs changed. Please recompute metrics.")
    st.stop()

results_df = inference["results_df"]
fairness_bootstrap = inference["fairness_bootstrap"]
y_true = inference["y_true"]
sens_arr = inference["sensitive_attr"]

AVAILABLE_FAIRNESS_METRICS = [
    c for c in [
        "Statistical Parity Difference",
        "Disparate Impact",
        "Average Odds Difference",
        "Error Rate Difference",
        "Equal Opportunity Difference",
        "Equalized Odds",
        "Thiel Index",
    ]
    if c in results_df.columns
]

# --------------------------------------------------
# Sidebar navigation (UNCHANGED)
# --------------------------------------------------
subpage = st.sidebar.radio(
    "Inference & Fairness Story",
    [
        "1️⃣ Model Readiness",
        "2️⃣ Outcome Distribution & Parity",
        "3️⃣ Error Disparities",
        "4️⃣ Fairness–Performance Tradeoffs",
        "5️⃣ Model Comparison & Risk Summary",
    ],
)

# ==================================================
# 1️⃣ Model Readiness
# ==================================================
if subpage == "1️⃣ Model Readiness":
    st.dataframe(inference["tables"]["performance"].set_index("Model"), use_container_width=True)
    st.pyplot(plot_bar_single_metric(results_df, "Accuracy"))
    st.pyplot(plot_line_single_metric(results_df, "Accuracy"))

# ==================================================
# 2️⃣ Outcome Distribution & Parity
# ==================================================
elif subpage == "2️⃣ Outcome Distribution & Parity":
    st.dataframe(inference["tables"]["fairness"].set_index("Model"), use_container_width=True)

    metric = st.selectbox("Outcome fairness metric", AVAILABLE_FAIRNESS_METRICS)
    st.pyplot(plot_bar_single_metric(results_df, metric))

    model = st.selectbox("Model for group outcomes", pred_cols)
    group_df = (
        pd.DataFrame({
            sensitive_col: sens_arr,
            "selection_rate": _as01(data[model].values, POS_PRED),
        })
        .groupby(sensitive_col, as_index=False)
        .mean()
    )

    st.pyplot(
        plot_by_group_bars(
            group_df,
            sensitive_col=sensitive_col,
            value_col="selection_rate",
            title="Selection Rate by Group",
        )
    )

# ==================================================
# 3️⃣ Error Disparities
# ==================================================
elif subpage == "3️⃣ Error Disparities":
    model = st.selectbox("Model", pred_cols)

    fig, stats = plot_disparity_in_performance(
        y_true, data[model].values, sens_arr,
        positive_true=POS_TRUE, positive_pred=POS_PRED
    )
    st.pyplot(fig)
    st.dataframe(stats["per_group"])

    st.pyplot(
        plot_group_error_panel(
            y_true, data[model].values, sens_arr,
            group_name=sensitive_col,
            positive_true=POS_TRUE, positive_pred=POS_PRED
        )
    )

# ==================================================
# 4️⃣ Fairness–Performance Tradeoffs
# ==================================================
elif subpage == "4️⃣ Fairness–Performance Tradeoffs":
    st.pyplot(
        plot_fairness_accuracy_scatter(
            results_df,
            fairness_metric=st.selectbox("Fairness metric", AVAILABLE_FAIRNESS_METRICS),
            performance_metric=st.selectbox(
                "Performance metric", ["Accuracy", "TPR (Recall)", "FPR", "FNR"]
            ),
        )
    )

# ==================================================
# 5️⃣ Model Comparison & Risk Summary
# ==================================================
elif subpage == "5️⃣ Model Comparison & Risk Summary":
    st.pyplot(plot_fairness_error_bars(fairness_bootstrap))

    heatmap_df = results_df.set_index("Model")[AVAILABLE_FAIRNESS_METRICS]
    st.pyplot(plot_models_groups_heatmap(heatmap_df))

    st.dataframe(
        inference["tables"]["fairness"].set_index("Model"),
        use_container_width=True
    )

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Inference & Fairness Evaluation")
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
data = st.session_state.get("uploaded_data")
label_col = st.session_state.get("ground_truth_col")
sensitive_col = st.session_state.get("sensitive_col")
privileged_value = st.session_state.get("privileged_value")

if not isinstance(data, pd.DataFrame) or data.empty:
    st.warning("No dataset available. Complete earlier steps first.")
    st.stop()

if None in (label_col, sensitive_col, privileged_value):
    st.error("Ground truth, sensitive attribute, and privileged group must be set.")
    st.stop()

# --------------------------------------------------
# Detect prediction columns
# --------------------------------------------------
pred_cols = [
    c for c in data.columns
    if c not in {label_col, sensitive_col}
    and set(pd.Series(data[c]).dropna().unique()).issubset({0, 1})
]

if not pred_cols:
    st.error("No prediction columns detected.")
    st.stop()

# --------------------------------------------------
# Positive class selectors
# --------------------------------------------------
def _guess_positive(vals):
    for v in vals:
        if str(v).lower() in {"1", "true", "yes", "approved", "positive"}:
            return v
    return sorted(vals)[-1]

label_vals = data[label_col].astype(str).unique().tolist()
pred_vals = sorted({str(v) for c in pred_cols for v in data[c].dropna().unique()})

c1, c2 = st.columns(2)
POS_TRUE = c1.selectbox(
    "Positive class (ground truth)",
    label_vals,
    index=label_vals.index(_guess_positive(label_vals)),
)
POS_PRED = c2.selectbox(
    "Positive class (predictions)",
    pred_vals,
    index=pred_vals.index(_guess_positive(pred_vals)),
)

# --------------------------------------------------
# Metric computation
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def compute_all_metrics(
    df, label_col, sensitive_col, pred_cols,
    privileged_value, pos_true, pos_pred, B=100
):
    y_true = _as01(df[label_col].values, positive=pos_true)
    sens_arr = df[sensitive_col].astype(str).values

    rows = []
    fairness_bootstrap = {}

    for col in pred_cols:
        y_pred = _as01(df[col].values, positive=pos_pred)

        perf = GroupMetrics(y_true, y_pred).get_all()
        fair = FairnessMetrics(
            y_true, y_pred, sens_arr,
            privileged_value=privileged_value
        ).get_all()

        rows.append({"Model": col, **perf, **fair})

        for m in fair.keys():
            fairness_bootstrap.setdefault(m, {})[col] = []
            for _ in range(B):
                idx = np.random.choice(len(y_true), len(y_true), replace=True)
                fb = FairnessMetrics(
                    y_true[idx], y_pred[idx], sens_arr[idx],
                    privileged_value=privileged_value
                ).get_all()
                fairness_bootstrap[m][col].append(fb.get(m))

    return pd.DataFrame(rows), fairness_bootstrap, y_true, sens_arr


# --------------------------------------------------
# Compute trigger
# --------------------------------------------------
if st.button("Compute metrics"):
    results_df, fairness_bootstrap, y_true, sens_arr = compute_all_metrics(
        data, label_col, sensitive_col, pred_cols,
        privileged_value, POS_TRUE, POS_PRED
    )

    def safe_table(df, cols):
        keep = [c for c in cols if c in df.columns]
        return df[["Model"] + keep]

    PERFORMANCE_COLS = [
        "TP", "TN", "FP", "FN",
        "Accuracy", "TPR (Recall)", "TNR",
        "FPR", "FNR", "Precision (PPV)",
        "NPV", "FDR", "FOR", "F1"
    ]

    FAIRNESS_COLS = [
        "Statistical Parity Difference",
        "Disparate Impact",
        "Average Odds Difference",
        "Error Rate Difference",
        "Equal Opportunity Difference",
        "Equalized Odds",
        "Thiel Index",
    ]

    # 🔒 CONTRACT: DO NOT CHANGE THESE KEYS
    st.session_state["inference"] = {
        "completed": True,
        "results_df": results_df,
        "fairness_bootstrap": fairness_bootstrap,
        "y_true": y_true,
        "sensitive_attr": sens_arr,
        "positive_class": {
            "ground_truth": POS_TRUE,
            "prediction": POS_PRED,
        },
        "tables": {
            "performance": safe_table(results_df, PERFORMANCE_COLS),
            "fairness": safe_table(results_df, FAIRNESS_COLS),
        },
    }

    st.session_state["metrics_ready"] = True

if not st.session_state.get("metrics_ready"):
    st.stop()

# --------------------------------------------------
# Unpack (read-only)
# --------------------------------------------------
inference = st.session_state["inference"]
results_df = inference["results_df"]
fairness_bootstrap = inference["fairness_bootstrap"]
y_true = inference["y_true"]
sens_arr = inference["sensitive_attr"]

AVAILABLE_FAIRNESS_METRICS = [
    c for c in [
        "Statistical Parity Difference",
        "Disparate Impact",
        "Average Odds Difference",
        "Error Rate Difference",
        "Equal Opportunity Difference",
        "Equalized Odds",
        "Thiel Index",
    ]
    if c in results_df.columns
]

# --------------------------------------------------
# Sidebar navigation (story-based, safe)
# --------------------------------------------------
subpage = st.sidebar.radio(
    "Inference & Fairness Story",
    [
        "1️⃣ Model Readiness",
        "2️⃣ Outcome Distribution & Parity",
        "3️⃣ Error Disparities",
        "4️⃣ Fairness–Performance Tradeoffs",
        "5️⃣ Model Comparison & Risk Summary",
    ],
)

# ==================================================
# 1️⃣ Model Readiness
# ==================================================
if subpage == "1️⃣ Model Readiness":
    st.dataframe(inference["tables"]["performance"].set_index("Model"), use_container_width=True)
    st.pyplot(plot_bar_single_metric(results_df, "Accuracy"))
    st.pyplot(plot_line_single_metric(results_df, "Accuracy"))

# ==================================================
# 2️⃣ Outcome Distribution & Parity
# ==================================================
elif subpage == "2️⃣ Outcome Distribution & Parity":
    st.dataframe(inference["tables"]["fairness"].set_index("Model"), use_container_width=True)

    metric = st.selectbox("Outcome fairness metric", AVAILABLE_FAIRNESS_METRICS)
    st.pyplot(plot_bar_single_metric(results_df, metric))

    model = st.selectbox("Model for group outcomes", pred_cols)
    group_df = (
        pd.DataFrame({
            sensitive_col: sens_arr,
            "selection_rate": _as01(data[model].values, POS_PRED),
        })
        .groupby(sensitive_col, as_index=False)
        .mean()
    )

    st.pyplot(
        plot_by_group_bars(
            group_df,
            sensitive_col=sensitive_col,
            value_col="selection_rate",
            title="Selection Rate by Group",
        )
    )

# ==================================================
# 3️⃣ Error Disparities
# ==================================================
elif subpage == "3️⃣ Error Disparities":
    model = st.selectbox("Model", pred_cols)

    fig, stats = plot_disparity_in_performance(
        y_true, data[model].values, sens_arr,
        positive_true=POS_TRUE, positive_pred=POS_PRED
    )
    st.pyplot(fig)
    st.dataframe(stats["per_group"])

    st.pyplot(
        plot_group_error_panel(
            y_true, data[model].values, sens_arr,
            group_name=sensitive_col,
            positive_true=POS_TRUE, positive_pred=POS_PRED
        )
    )

# ==================================================
# 4️⃣ Fairness–Performance Tradeoffs
# ==================================================
elif subpage == "4️⃣ Fairness–Performance Tradeoffs":
    st.pyplot(
        plot_fairness_accuracy_scatter(
            results_df,
            fairness_metric=st.selectbox("Fairness metric", AVAILABLE_FAIRNESS_METRICS),
            performance_metric=st.selectbox(
                "Performance metric", ["Accuracy", "TPR (Recall)", "FPR", "FNR"]
            ),
        )
    )

# ==================================================
# 5️⃣ Model Comparison & Risk Summary
# ==================================================
elif subpage == "5️⃣ Model Comparison & Risk Summary":
    st.pyplot(plot_fairness_error_bars(fairness_bootstrap))

    heatmap_df = results_df.set_index("Model")[AVAILABLE_FAIRNESS_METRICS]
    st.pyplot(plot_models_groups_heatmap(heatmap_df))

    st.dataframe(inference["tables"]["fairness"].set_index("Model"), use_container_width=True)
