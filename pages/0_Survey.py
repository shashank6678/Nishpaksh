"""
Survey page — INPUT ONLY (governance risk assessment)

Responsibilities:
- Render governance / audit survey using utils.survey.render_survey()
- Interpret results correctly as RISK (not performance)
- Store normalized outputs in st.session_state for downstream reporting
- Generate on-screen PREVIEW visualizations (ephemeral)
- Lock survey once completed

Hard rules:
- No DOCX / PDF generation
- No filesystem writes
- No report logic
"""

import streamlit as st
import matplotlib.pyplot as plt
import traceback
from typing import Dict, Any

# --------------------------------------------------
# Import survey renderer (unchanged)
# --------------------------------------------------
try:
    from utils.survey import render_survey
except Exception:
    render_survey = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def extract_governance_outputs(submission: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize governance-risk outputs from survey submission.
    This function is CONTRACT-ALIGNED with utils/survey.py.
    """
    if not isinstance(submission, dict):
        raise ValueError("Invalid survey submission format")

    total_risk = submission.get("total_risk_score")
    risk_category = submission.get("risk_category")
    proxy_subscores = submission.get("subscores", {})
    answers = submission.get("answers", {})

    # ---- compute section-level average risk (1–5 scale) ----
    section_avg_risk: Dict[str, float] = {}
    for section, responses in answers.items():
        risk_sum = 0
        count = 0
        for resp in responses.values():
            # utils.survey already excluded "Not Applicable" during scoring,
            # but we stay defensive here.
            from utils.survey import get_risk_score
            score = get_risk_score(resp)
            if score > 0:
                risk_sum += score
                count += 1
        section_avg_risk[section] = (risk_sum / count) if count > 0 else 0.0

    return {
        "total_risk_score": float(total_risk) if total_risk is not None else None,
        "risk_category": risk_category,
        "proxy_subscores": proxy_subscores,
        "section_avg_risk": section_avg_risk,
        "raw_submission": submission,
    }


def make_proxy_risk_plot(proxy_scores: Dict[str, float]):
    """
    Preview plot: Proxy-bucket risk contributions (0–20 each).
    This matches the governance aggregation logic exactly.
    """
    labels = list(proxy_scores.keys())
    values = [proxy_scores[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    y = range(len(labels))

    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels([l.replace("_", " ").title() for l in labels])
    ax.set_xlim(0, 20)
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.4, i, f"{v:.0f}", va="center")

    ax.set_title("Governance Risk Drivers (Proxy Contributions)")
    ax.set_xlabel("Risk Contribution (0–20 per proxy)")
    plt.tight_layout()
    return fig


# --------------------------------------------------
# Streamlit Page
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Survey — Governance & Audit Risk Assessment")

if render_survey is None:
    st.error("Survey renderer not available. Ensure utils/survey.py exports render_survey().")
    st.stop()

# --------------------------------------------------
# Render Survey
# --------------------------------------------------
try:
    submission = render_survey(embedded=False, require_identity=False)
except Exception:
    st.error("Survey renderer raised an exception.")
    st.code(traceback.format_exc())
    st.stop()

# Survey not finished yet
if not submission:
    st.info("Please complete and submit the survey to proceed.")
    st.stop()

st.success("Survey completed successfully.")

# --------------------------------------------------
# Normalize + Store Outputs (ONCE)
# --------------------------------------------------
if not st.session_state.get("survey_completed"):

    outputs = extract_governance_outputs(submission)

    st.session_state["survey_outputs"] = outputs

    # minimal plotting payload (no raw submission duplication)
    st.session_state["survey_plot_data"] = {
        "proxy_subscores": outputs["proxy_subscores"]
    }

    st.session_state["survey_completed"] = True

# --------------------------------------------------
# Display Stored Summary
# --------------------------------------------------
survey_outputs = st.session_state.get("survey_outputs", {})

st.subheader("Governance Risk Summary")

st.metric(
    label="Aggregate Risk Index (0–100)",
    value=survey_outputs.get("total_risk_score", "N/A")
)

st.metric(
    label="Overall Risk Category",
    value=survey_outputs.get("risk_category", "N/A")
)

st.caption(
    "ℹ️ Higher values indicate higher governance and fairness risk. "
    "This is a risk index, not a performance score."
)

# --------------------------------------------------
# Preview: Proxy Risk Drivers (Ephemeral)
# --------------------------------------------------
plot_data = st.session_state.get("survey_plot_data")

if plot_data and plot_data.get("proxy_subscores"):
    st.subheader("Risk Driver Preview")
    fig = make_proxy_risk_plot(plot_data["proxy_subscores"])
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No proxy-level risk data available for visualization.")

# --------------------------------------------------
# Lock Notice
# --------------------------------------------------
st.markdown("---")
st.caption(
    "✔ Survey results are locked and stored for downstream reporting. "
    "To modify responses, reset the session."
)
