"""
Pre-processing page — DATA & MODEL PREPARATION ONLY

Responsibilities:
- Light EDA (preview only)
- Leakage / proxy check (preview only)
- Train baseline models and append predictions
- Collect user-authored system descriptions (verbatim)
- Store all outputs in st.session_state
- NO report generation
- NO filesystem writes
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Pre-processing — Data & Model Preparation")

# --------------------------------------------------
# Preconditions
# --------------------------------------------------
if "uploaded_data" not in st.session_state or not isinstance(
    st.session_state["uploaded_data"], pd.DataFrame
):
    st.warning("Upload a dataset on the Home page before proceeding.")
    st.stop()

df = st.session_state["uploaded_data"]

ground_truth_col = st.session_state.get("ground_truth_col")
sensitive_col = st.session_state.get("sensitive_col")

if ground_truth_col is None:
    st.warning("Ground truth column not set on Home page.")
    st.stop()

# --------------------------------------------------
# Initialize session_state container (once)
# --------------------------------------------------
if "preproc" not in st.session_state:
    st.session_state["preproc"] = {
        "ignore_cols": [],
        "eda_done": False,
        "leakage": None,
        "models": None,
        "user_narratives": {},
        "completed": False,
    }

PREPROC = st.session_state["preproc"]

# --------------------------------------------------
# Section selector
# --------------------------------------------------
section = st.sidebar.radio(
    "Pre-processing section",
    [
        "Exploratory Data Analysis",
        "Leakage / Proxy Check",
        "Model Training & Prediction Append",
        "System Description (User Input)",
    ],
)

# ==================================================
# 1. Exploratory Data Analysis (Preview Only)
# ==================================================
if section == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis (Preview)")

    st.dataframe(df.head())

    with st.expander("Summary statistics"):
        st.dataframe(df.describe(include="all"))

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
        ax.set_title("Numeric feature correlations")
        st.pyplot(fig)
        plt.close(fig)

    PREPROC["eda_done"] = True

# ==================================================
# 2. Leakage / Proxy Check (Preview Only)
# ==================================================
elif section == "Leakage / Proxy Check":
    st.subheader("Sensitive Attribute Leakage Check")

    if sensitive_col is None:
        st.info("Sensitive attribute not set on Home page.")
        st.stop()

    ignore_cols = st.multiselect(
        "Columns to exclude",
        options=df.columns.tolist(),
        default=PREPROC.get("ignore_cols", []),
    )
    PREPROC["ignore_cols"] = ignore_cols

    feature_cols = [
        c
        for c in df.columns
        if c not in ignore_cols + [sensitive_col, ground_truth_col]
    ]

    if not feature_cols:
        st.warning("No usable features after exclusions.")
        st.stop()

    X = pd.get_dummies(df[feature_cols].fillna("NA"))
    y = LabelEncoder().fit_transform(df[sensitive_col].astype(str))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)

    importances = (
        pd.Series(clf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(20)
    )

    st.metric("Leakage prediction accuracy", f"{acc:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    importances.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title(f"Top features predicting `{sensitive_col}`")
    st.pyplot(fig)
    plt.close(fig)

    PREPROC["leakage"] = {
        "accuracy": float(acc),
        "top_features": importances.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ),
    }

# ==================================================
# 3. Model Training & Prediction Append
# ==================================================
elif section == "Model Training & Prediction Append":
    st.subheader("Baseline Model Training")

    model_map = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(probability=True, random_state=42),
    }

    models_to_run = st.multiselect(
        "Select models to train",
        list(model_map.keys()),
        default=list(model_map.keys()),
    )

    ignore_cols = PREPROC.get("ignore_cols", [])

    X = df.drop(columns=ignore_cols + [ground_truth_col], errors="ignore")
    y = LabelEncoder().fit_transform(df[ground_truth_col].astype(str))

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    results = []

    for name in models_to_run:
        model = model_map[name]
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(Xtr, ytr)

        preds = pipe.predict(X)
        col_name = f"predicted_{name}"
        df[col_name] = preds

        yhat = pipe.predict(Xte)
        results.append(
            {
                "Model": col_name,
                "Accuracy": accuracy_score(yte, yhat),
                "Precision": precision_score(yte, yhat, zero_division=0),
                "Recall": recall_score(yte, yhat, zero_division=0),
                "F1": f1_score(yte, yhat, zero_division=0),
            }
        )

    res_df = pd.DataFrame(results)
    st.dataframe(res_df.set_index("Model").style.format("{:.3f}"))

    st.session_state["uploaded_data"] = df
    PREPROC["models"] = res_df

# ==================================================
# 4. System Description (User Input — Verbatim)
# ==================================================
elif section == "System Description (User Input)":
    st.subheader("AI System & Pipeline Description")

    st.markdown(
        "This text will appear **verbatim** in the final certification report. "
        "No rewriting or summarization will be applied."
    )

    system_desc = st.text_area(
        "AI system description (P14)",
        value=PREPROC["user_narratives"].get("P14", ""),
        height=180,
    )

    pipeline_desc = st.text_area(
        "Data, model, and pipeline description (P15)",
        value=PREPROC["user_narratives"].get("P15", ""),
        height=180,
    )

    if st.button("Confirm and lock descriptions"):
        PREPROC["user_narratives"]["P14"] = system_desc
        PREPROC["user_narratives"]["P15"] = pipeline_desc
        PREPROC["completed"] = True
        st.success("Descriptions saved and locked for final report.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Pre-processing page stores structured evidence and user-authored text only. "
    "Final Report page will compile all sections deterministically."
)
