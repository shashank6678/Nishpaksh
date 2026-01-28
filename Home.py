# home.py
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Nishpaksh", layout="centered")

# ---- THEME + FONT SIZE OVERRIDES ----
st.markdown(""" 
    <style>
    [data-testid="stAppViewContainer"] { background-color: #F9FAFB; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; font-size: 17px !important; }
    button[kind="primary"] {
        background-color: #2E86AB !important;
        color: white !important;
        font-size: 18px !important;
        padding: 0.6em 1.2em !important;
        border-radius: 6px !important;
    }
    button[kind="primary"]:hover { background-color: #1B4F72 !important; color: white !important; }
    input[type="radio"], input[type="checkbox"] { accent-color: #2E86AB !important; }
    label[data-testid="stMarkdownContainer"] { font-size: 17px !important; }
    .dataframe { font-size: 16px !important; }
    html, body, [class*="css"]  { font-size: 18px !important; }
    h1, h2, h3, h4 { font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ---- Branding: three logos (TEC - IIITD - MeitY) ----
LOGO_WIDTH = 300
tec_paths = ["tec.png", "tec.jpg", "assets/tec.png", "assets/tec.jpg"]
iiitd_paths = ["iiitd.png", "iiitd.jpg", "assets/iiitd.png", "assets/iiitd.jpg"]
meity_paths = ["meity.png", "meity.jpg", "assets/meity.png", "assets/meity.jpg"]

col_l, col_c, col_r = st.columns([1, 1, 1])

with col_l:  # TEC left
    shown = False
    for p in tec_paths:
        try:
            st.image(p, width=LOGO_WIDTH)
            shown = True
            break
        except Exception:
            continue
    if not shown: st.write("")

with col_c:  # IIITD center
    shown = False
    for p in iiitd_paths:
        try:
            st.image(p, width=LOGO_WIDTH)
            shown = True
            break
        except Exception:
            continue
    if not shown: st.write("")

with col_r:  # MeitY right
    shown = False
    for p in meity_paths:
        try:
            st.image(p, width=LOGO_WIDTH)
            shown = True
            break
        except Exception:
            continue
    if not shown: st.write("")

# ------------------ App Logic ------------------
st.title("Nishpaksh")

data_file = st.file_uploader("Upload tabular dataset (CSV)", type=["csv"])
model_file = st.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])

if data_file:
    try:
        df = pd.read_csv(data_file)
        st.session_state["uploaded_data"] = df
        st.write("Data Preview")
        st.dataframe(df.head())
        columns = df.columns.tolist()

        ground_truth_col = st.selectbox("Select ground truth column", options=columns)
        st.session_state["ground_truth_col"] = ground_truth_col

        sensitive_col = st.selectbox("Select sensitive attribute column", options=columns)
        st.session_state["sensitive_col"] = sensitive_col

        unique_vals = df[sensitive_col].dropna().unique()
        priv_value = st.selectbox("Select privileged group", options=unique_vals)
        st.session_state["privileged_value"] = priv_value
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

if model_file:
    st.session_state["model_file"] = model_file
    st.success("Model file uploaded.")

st.markdown("---")
st.sidebar.title("Navigation")

if "uploaded_data" in st.session_state and "model_file" in st.session_state:
    st.sidebar.success("Files uploaded")
    st.sidebar.page_link("pages/1_Pre_Processing.py", label="Pre-Processing")
    st.sidebar.page_link("pages/2_Inference.py", label="Inference")
    st.sidebar.page_link("pages/3_Output.py", label="Output")
else:
    st.sidebar.info("Upload both data and model to proceed.")
