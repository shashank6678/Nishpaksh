# pages/4_Metrics_and_Thresholds.py
"""
Standalone Metrics & Thresholds page (improved UX)
- User selects problem type (classification/regression/clustering/recommendation)
- Metrics selection is a checkbox grid (with Select all / Clear selection)
- Threshold numeric inputs appear inline when a metric is checked
- Optional notes removed (per request)
- Thresholds saved to st.session_state['thresholds'] but not applied in Results (standalone)
"""

import streamlit as st
import pandas as pd
import math

st.set_page_config(layout="centered", page_title="Metrics & Thresholds")

st.title("Model type and group fairness metrics")

st.markdown(
    "Choose the problem type, then tick the group/fairness metrics you want to configure. "
    "When a metric is checked, its threshold input appears inline. "
    
)

# Problem selection
problem = st.radio(
    "Which kind of problem is this model solving?",
    ("classification", "regression", "clustering", "recommendation"),
    index=0,
    horizontal=True
)

st.markdown(f"**Problem type:** `{problem}`")

# Group fairness metrics (kept same for simplicity)
GROUP_METRICS = [
    "Statistical Parity Difference",
    "Disparate Impact",
    "Average Odds Difference",
    "Equal Opportunity Difference",
    "Error Rate Difference",
    "Calibration Difference (global)"
]

# sensible defaults
DEFAULTS = {
    "Statistical Parity Difference": 0.10,
    "Disparate Impact": 0.80,
    "Average Odds Difference": 0.10,
    "Equal Opportunity Difference": 0.10,
    "Error Rate Difference": 0.10,
    "Calibration Difference (global)": 0.05,
}

# Load existing thresholds if present
saved_thresholds = st.session_state.get("thresholds", {}) if "thresholds" in st.session_state else {}

# Keep an internal selection state so Select all / Clear selection work smoothly
if "metric_selection" not in st.session_state or st.session_state.get("_metric_selection_problem") != problem:
    # initialize selection: checked if present in saved thresholds OR default to checked
    st.session_state["metric_selection"] = {m: (m in saved_thresholds or True) for m in GROUP_METRICS}
    st.session_state["_metric_selection_problem"] = problem

# header controls: select all / clear all
col_a, col_b, _ = st.columns([1, 1, 6])
with col_a:
    if st.button("Select all metrics"):
        for m in GROUP_METRICS:
            st.session_state["metric_selection"][m] = True
with col_b:
    if st.button("Clear all"):
        for m in GROUP_METRICS:
            st.session_state["metric_selection"][m] = False

st.markdown("### Choose group/fairness metrics to configure")

# Render checkboxes in two columns for compactness
ncols = 2
cols = st.columns(ncols)
for i, m in enumerate(GROUP_METRICS):
    col = cols[i % ncols]
    key_chk = f"chk_{problem}_{m}"
    # default checked state from session state
    default_checked = st.session_state["metric_selection"].get(m, True)
    checked = col.checkbox(label=m, value=default_checked, key=key_chk)
    # persist into metric_selection map so select-all/clear-all reflect current state
    st.session_state["metric_selection"][m] = checked

    # inline threshold input appears immediately below the checkbox (indented)
    if checked:
        # Use a small number_input; keep format flexible for different metric types
        key_th = f"th_{problem}_{m}"
        # decide initial value: saved thresholds -> saved, else DEFAULTS
        init_val = saved_thresholds.get(m, {}).get("value", DEFAULTS.get(m, 0.0))
        # If init_val might be non-finite, fallback
        try:
            init_val = float(init_val)
            if math.isnan(init_val) or math.isinf(init_val):
                init_val = float(DEFAULTS.get(m, 0.0))
        except Exception:
            init_val = float(DEFAULTS.get(m, 0.0))
        # layout: place threshold input immediately after the checkbox
        col.number_input(f"Threshold for {m}", value=init_val, step=0.01, format="%.4f", key=key_th)

st.markdown("---")
# Build thresholds dict from current checkbox states + inputs
thresholds = {}
for m in GROUP_METRICS:
    if st.session_state["metric_selection"].get(m, False):
        key_th = f"th_{problem}_{m}"
        # number_input always exists as a widget when metric is checked; try to read. If missing, fallback.
        val = st.session_state.get(key_th, DEFAULTS.get(m, 0.0))
        thresholds[m] = {"value": float(val), "problem": problem}

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Save thresholds to session"):
        st.session_state["thresholds"] = thresholds
        st.success("Saved thresholds to session.")
with col2:
    if st.button("Clear thresholds from session"):
        if "thresholds" in st.session_state:
            del st.session_state["thresholds"]
        st.success("Cleared thresholds from session.")


