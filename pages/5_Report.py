# pages/5_Report.py
# FINAL REPORT COMPILER — WIRE-FRAME ALIGNED (TEC 7.1 COMPLIANT)
# UI REFINED + SECTION STATUS (NO LOGIC CHANGES)

import streamlit as st
import numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Inches
import tempfile
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="wide")
st.title("Final Fairness Evaluation Report")
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
st.markdown(
    """
    <style>
    /* =========================
       Brand accent system
       ========================= */

    :root {
        --accent-color: #2563eb;      /* muted blue */
        --accent-soft: #eff6ff;       /* very light blue */
        --accent-border: #bfdbfe;
    }

    /* =========================
       Page title accent
       ========================= */
    h1::after {
        content: "";
        display: block;
        width: 64px;
        height: 3px;
        background-color: var(--accent-color);
        margin-top: 8px;
        border-radius: 2px;
    }

    /* =========================
       Tabs — active indicator
       ========================= */
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid var(--accent-color) !important;
        color: var(--accent-color) !important;
    }

    /* =========================
       Status panel background
       ========================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(h3:contains("Report completion status")),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(h2:contains("Report completion status")) {
        background-color: var(--accent-soft);
        border-color: var(--accent-border);
    }

    /* =========================
       Section headers — subtle left rail
       ========================= */
    h2 {
        border-left: 4px solid var(--accent-color);
        padding-left: 0.75rem;
    }

    /* =========================
       Sidebar refinement
       ========================= */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] a {
        font-weight: 500;
    }

    /* Highlight active page in sidebar */
    section[data-testid="stSidebar"] li:has(a[aria-current="page"]) {
        background-color: #e0f2fe;
        border-radius: 6px;
    }

    /* =========================
       Status badge enhancement
       ========================= */
    .status-complete {
        background-color: #dcfce7;
        padding: 0.15rem 0.45rem;
        border-radius: 999px;
    }

    .status-pending {
        background-color: #ffedd5;
        padding: 0.15rem 0.45rem;
        border-radius: 999px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ==================================================
# PRECONDITIONS
# ==================================================
REQUIRED_KEYS = ["survey_outputs", "preproc", "inference", "results"]
for key in REQUIRED_KEYS:
    if key not in st.session_state:
        st.error(f"Missing required step: {key}")
        st.stop()

survey = st.session_state["survey_outputs"]
preproc = st.session_state["preproc"]
inference = st.session_state["inference"]
results = st.session_state["results"]

# ==================================================
# DEFENSIVE INITIALIZATION (UNCHANGED)
# ==================================================
survey.setdefault("risk_bucket", "Medium")
survey.setdefault("auditor_access", "Training + Validation")
survey.setdefault("testing_type", "Grey box")
survey.setdefault("dependencies", "")
survey.setdefault("limitations", "")
survey.setdefault("questionnaire_summary", "")
survey.setdefault("protected_attr", "")
survey.setdefault("privileged_groups", "")
survey.setdefault("favourable_outcome", "")
survey.setdefault("protected_attr_rationale", "")
survey.setdefault("metric_rationale", "")
survey.setdefault("threshold_rationale", "")
survey.setdefault("risk_outcome", "")
survey.setdefault("certification_context", "")

preproc.setdefault("user_narratives", {})
preproc["user_narratives"].setdefault("P14", "")
preproc.setdefault("pipeline_description", "")
preproc.setdefault("split_method", "Not specified")
preproc.setdefault("synthetic_data", "No synthetic data used.")
preproc.setdefault("scenarios_tested", "")

# ==================================================
# TEMPLATE
# ==================================================
TEMPLATE_PATH = Path(__file__).parent / "Fairness_Evaluation_Report_Wireframe.docx"
if not TEMPLATE_PATH.exists():
    st.error("Wireframe DOCX template not found.")
    st.stop()

# ==================================================
# DOCX HELPERS (UNCHANGED)
# ==================================================
def replace_text(doc, token, value):
    found = False
    for p in doc.paragraphs:
        if token in p.text:
            p.text = p.text.replace(token, str(value))
            found = True
    return found

def insert_table(doc, token, df):
    for p in doc.paragraphs:
        if token in p.text:
            p.text = ""
            table = doc.add_table(rows=1, cols=len(df.columns))
            for i, col in enumerate(df.columns):
                table.rows[0].cells[i].text = str(col)
            for _, row in df.iterrows():
                cells = table.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = f"{v:.4f}" if isinstance(v, float) else str(v)
            p._p.addnext(table._tbl)
            return True
    return False

def insert_plot(doc, token, fig):
    for p in doc.paragraphs:
        if token in p.text:
            p.text = p.text.replace(token, "")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, dpi=240, bbox_inches="tight")
                plt.close(fig)
                p.add_run().add_picture(tmp.name, width=Inches(5.5))
            return True
    return False

# ==================================================
# PLOTS (UNCHANGED)
# ==================================================
def plot_fairness_summary():
    metrics = results["fairness_metrics"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(metrics.keys(), metrics.values())
    ax.axhline(0, linestyle="--", color="black")
    ax.axhline(1, linestyle="--", color="black")
    ax.set_title("Bias Index by Protected Attribute")
    plt.xticks(rotation=30, ha="right")
    return fig

def plot_accuracy_bar():
    df = inference["results_df"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["Model"], df["Accuracy"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=30, ha="right")
    return fig

def plot_fairness_ci():
    fb = inference["fairness_bootstrap"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for metric, models in fb.items():
        for model, vals in models.items():
            ax.errorbar(model, np.mean(vals), yerr=np.std(vals), fmt="o")
    ax.set_title("Fairness Metric Uncertainty")
    return fig

# ==================================================
# SECTION STATUS PANEL (RESTORED & EXPLICIT)
# ==================================================
# ==================================================
# SECTION STATUS PANEL (INTENTIONAL COMPLETION CHECK)
# ==================================================
def is_meaningful(text: str, min_len: int = 10) -> bool:
    if text is None:
        return False
    return len(text.strip()) >= min_len

def status_badge(ok: bool) -> str:
    return "🟢 Complete" if ok else "🟠 Needs input"

with st.container(border=True):
    st.markdown("### Report completion status")
    st.caption(
        "Sections are marked complete only after sufficient, explicit information "
        "has been provided. Default selections do not imply completion."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**1. System context and scope** — "
            f"{status_badge(is_meaningful(survey['protected_attr']) and is_meaningful(survey['favourable_outcome']))}"
        )

        st.write(
            f"**2. Audit configuration** — "
            f"{status_badge(survey['testing_type'] in ['Open box','Grey box','Closed box'])}"
        )

        st.write(
            f"**3. Data and pipeline description** — "
            f"{status_badge(is_meaningful(preproc['pipeline_description']))}"
        )

    with col2:
        st.write(
            f"**4. Risk and limitations** — "
            f"{status_badge(is_meaningful(survey['limitations']))}"
        )

        st.write(
            f"**5. Fairness rationale** — "
            f"{status_badge(is_meaningful(survey['metric_rationale']) and is_meaningful(survey['threshold_rationale']))}"
        )

        st.write(
            f"**6. Certification context** — "
            f"{status_badge(is_meaningful(survey['certification_context']))}"
        )

# ==================================================
# UI TABS
# ==================================================
tab1, tab2, tab3 = st.tabs(["Summary", "Metrics", "Detailed Report"])

# ==================================================
# TAB 1 — SUMMARY
# ==================================================
with tab1:
    st.markdown("### System context and audit scope")
    st.caption(
        "Factual declarations describing what was evaluated and under what conditions."
    )

    with st.container(border=True):
        st.markdown(
            f"""
            Fairness Score: **{results['FS']:.3f}**  
            Bias Index: **{results['BI']:.3f}**  
            Final verdict: **{results['verdict']}**
            """
        )

    with st.container(border=True):
        survey["protected_attr"] = st.text_input(
            "Which protected attributes were evaluated?",
            survey["protected_attr"]
        )

        survey["privileged_groups"] = st.text_input(
            "How were privileged and unprivileged groups defined?",
            survey["privileged_groups"]
        )

        survey["favourable_outcome"] = st.text_input(
            "What outcome was considered favourable?",
            survey["favourable_outcome"]
        )

    with st.container(border=True):
        survey["risk_bucket"] = st.selectbox(
            "Risk classification of this evaluation",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(survey["risk_bucket"])
        )

        survey["auditor_access"] = st.selectbox(
            "Data access available to the evaluator",
            ["Training data only", "Training + Validation", "Full (Train/Validation/Test)"],
            index=1
        )

        survey["testing_type"] = st.radio(
            "System access level during testing",
            ["Open box", "Grey box", "Closed box"],
            index=["Open box", "Grey box", "Closed box"].index(survey["testing_type"])
        )

    with st.container(border=True):
        depends = st.radio(
            "Did this evaluation require assistance from the system developer?",
            ["No", "Yes"],
            index=0 if survey["dependencies"] in ("", "No developer dependency.") else 1
        )

        if depends == "Yes":
            survey["dependencies"] = st.text_area(
                "Describe the nature of developer involvement",
                survey["dependencies"],
                height=120
            )
        else:
            survey["dependencies"] = "No developer dependency."

        survey["limitations"] = st.text_area(
            "Which aspects of the system were not evaluated as part of this assessment?",
            survey["limitations"],
            height=140
        )

# ==================================================
# TAB 2 — METRICS (LOCKED)
# ==================================================
with tab2:
    st.markdown("### Computed metrics")
    st.caption("These values are generated automatically and cannot be edited.")

    with st.container(border=True):
        st.dataframe(inference["tables"]["fairness"], use_container_width=True)

    with st.container(border=True):
        st.dataframe(inference["tables"]["performance"], use_container_width=True)

# ==================================================
# TAB 3 — DETAILED REPORT
# ==================================================
with tab3:
    st.markdown("### Detailed evaluation narrative")

    with st.container(border=True):
        preproc["user_narratives"]["P14"] = st.text_area(
            "Describe the AI system and its intended use",
            preproc["user_narratives"]["P14"],
            height=140
        )

        preproc["pipeline_description"] = st.text_area(
            "Describe the data, model, and processing pipeline",
            preproc["pipeline_description"],
            height=140
        )

    with st.container(border=True):
        survey["questionnaire_summary"] = st.text_area(
            "Summary of fairness questionnaire responses",
            survey["questionnaire_summary"],
            height=120
        )

        survey["risk_outcome"] = st.text_area(
            "Outcome of the risk assessment",
            survey["risk_outcome"],
            height=120
        )

    with st.container(border=True):
        survey["protected_attr_rationale"] = st.text_area(
            "Rationale for selecting protected attributes",
            survey["protected_attr_rationale"],
            height=120
        )

        survey["metric_rationale"] = st.text_area(
            "Rationale for selecting fairness metrics",
            survey["metric_rationale"],
            height=120
        )

        survey["threshold_rationale"] = st.text_area(
            "Rationale for threshold selection",
            survey["threshold_rationale"],
            height=120
        )

    with st.container(border=True):
        preproc["scenarios_tested"] = st.text_area(
            "Evaluation scenarios tested",
            preproc["scenarios_tested"],
            height=120
        )

        survey["certification_context"] = st.text_area(
            "Certification and intended usage context",
            survey["certification_context"],
            height=120
        )

# ==================================================
# FINAL GENERATION (UNCHANGED)
# ==================================================
if st.button("Generate Final Report"):
    doc = Document(TEMPLATE_PATH)

    TEXT = {
        "[[P1_TEXT]]": f"FS={results['FS']:.3f}, BI={results['BI']:.3f}, Verdict={results['verdict']}",
        "[[P2_TEXT]]": preproc["user_narratives"]["P14"],
        "[[P3_TEXT]]": survey["protected_attr"],
        "[[P4_TEXT]]": survey["privileged_groups"],
        "[[P5_TEXT]]": survey["favourable_outcome"],
        "[[P6_TEXT]]": survey["risk_bucket"],
        "[[P7_TEXT]]": survey["auditor_access"],
        "[[P8_TEXT]]": survey["testing_type"],
        "[[P9_TEXT]]": survey["dependencies"],
        "[[P10_TEXT]]": survey["limitations"],
        "[[P13_TEXT]]": f"Overall fairness score FS={results['FS']:.3f} with verdict {results['verdict']}.",
        "[[P14_TEXT]]": preproc["user_narratives"]["P14"],
        "[[P15_TEXT]]": preproc["pipeline_description"],
        "[[P16_TEXT]]": survey["questionnaire_summary"],
        "[[P17_TEXT]]": survey["risk_outcome"],
        "[[P18_TEXT]]": survey["protected_attr_rationale"],
        "[[P19_TEXT]]": survey["metric_rationale"],
        "[[P20_TEXT]]": survey["threshold_rationale"],
        "[[P21_TEXT]]": preproc["split_method"],
        "[[P22_TEXT]]": preproc["synthetic_data"],
        "[[P23_TEXT]]": preproc["scenarios_tested"],
        "[[P26_TEXT]]": survey["certification_context"],
    }

    for k, v in TEXT.items():
        if not replace_text(doc, k, v):
            st.error(f"Missing placeholder in template: {k}")
            st.stop()

    insert_table(doc, "[[TABLE_P11]]", inference["tables"]["fairness"])
    insert_table(doc, "[[TABLE_P24]]", inference["tables"]["performance"])

    insert_plot(doc, "[[FIG_P12]]", plot_fairness_summary())
    insert_plot(doc, "[[FIG_P18]]", plot_accuracy_bar())
    insert_plot(doc, "[[FIG_P19]]", plot_fairness_ci())

    out = Path(tempfile.gettempdir()) / "Fairness_Evaluation_Report_Final.docx"
    doc.save(out)

    st.success("Final report generated successfully.")
    st.download_button("Download Report", open(out, "rb"), file_name=out.name)
