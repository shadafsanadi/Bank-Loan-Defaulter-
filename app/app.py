"""
Streamlit dashboard — Home Loan Default Prediction.

Three tabs:
  1. Prediction   — enter customer details, get risk score + SHAP waterfall
  2. Performance  — model metrics, ROC curve, PR curve (pre-generated images)
  3. About        — dataset overview, architecture summary, feature importance chart
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st

from src.predict import predict_default_risk, get_shap_explanation, get_model_info

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    border-left: 4px solid #2196F3;
    margin-bottom: 12px;
}
.risk-high { border-left-color: #f44336 !important; background: #fff5f5 !important; }
.risk-low  { border-left-color: #4CAF50 !important; background: #f5fff5 !important; }
.section-header { font-size: 1.1rem; font-weight: 600; color: #333; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Categorical mappings (get_dummies drop_first=True baseline = first alpha) ─

_INCOME_TYPES = {
    "Commercial associate": {},
    "Pensioner":            {"NAME_INCOME_TYPE_Pensioner": 1},
    "State servant":        {"NAME_INCOME_TYPE_State servant": 1},
    "Unemployed":           {"NAME_INCOME_TYPE_Unemployed": 1},
    "Working":              {"NAME_INCOME_TYPE_Working": 1},
}
_EDUCATION_TYPES = {
    "Academic degree":               {},
    "Higher education":              {"NAME_EDUCATION_TYPE_Higher education": 1},
    "Incomplete higher":             {"NAME_EDUCATION_TYPE_Incomplete higher": 1},
    "Lower secondary":               {"NAME_EDUCATION_TYPE_Lower secondary": 1},
    "Secondary / secondary special": {"NAME_EDUCATION_TYPE_Secondary / secondary special": 1},
}
_FAMILY_STATUSES = {
    "Civil marriage":       {},
    "Married":              {"NAME_FAMILY_STATUS_Married": 1},
    "Separated":            {"NAME_FAMILY_STATUS_Separated": 1},
    "Single / not married": {"NAME_FAMILY_STATUS_Single / not married": 1},
    "Widow":                {"NAME_FAMILY_STATUS_Widow": 1},
}
_HOUSING_TYPES = {
    "Co-op apartment":   {},
    "House / apartment": {"NAME_HOUSING_TYPE_House / apartment": 1},
    "Municipal apartment": {"NAME_HOUSING_TYPE_Municipal apartment": 1},
    "Office apartment":  {"NAME_HOUSING_TYPE_Office apartment": 1},
    "Rented apartment":  {"NAME_HOUSING_TYPE_Rented apartment": 1},
    "With parents":      {"NAME_HOUSING_TYPE_With parents": 1},
}

_RISK_BANDS = [
    (0.0,  0.10, "Very Low",  "#4CAF50", "Applicant shows very strong repayment indicators."),
    (0.10, 0.25, "Low",       "#8BC34A", "Applicant shows good repayment indicators."),
    (0.25, 0.45, "Medium",    "#FFC107", "Applicant shows moderate risk. Manual review recommended."),
    (0.45, 0.65, "High",      "#FF5722", "Applicant shows elevated risk. Senior review required."),
    (0.65, 1.01, "Very High", "#f44336", "Applicant shows very high default risk."),
]


def get_risk_band(prob: float) -> tuple[str, str, str]:
    for lo, hi, label, color, desc in _RISK_BANDS:
        if lo <= prob < hi:
            return label, color, desc
    return "Very High", "#f44336", ""


def build_feature_dict(
    income, credit, annuity, age_years, employed_years,
    ext2, ext3, gender, income_type, education, family_status, housing_type,
) -> dict:
    features = {
        "AMT_INCOME_TOTAL":     income,
        "AMT_CREDIT":           credit,
        "AMT_ANNUITY":          annuity,
        "DAYS_BIRTH":           -int(age_years * 365),
        "DAYS_EMPLOYED":        -int(employed_years * 365) if employed_years > 0 else 365243,
        "EXT_SOURCE_2":         ext2,
        "EXT_SOURCE_3":         ext3,
        "CODE_GENDER_M":        1 if gender == "Male" else 0,
        # Derived features the model was trained on
        "CREDIT_INCOME_RATIO":  credit / (income + 1),
        "ANNUITY_INCOME_RATIO": annuity / (income + 1),
        "CREDIT_TERM_MONTHS":   credit / (annuity + 1),
        "AGE_YEARS":            age_years,
        "EMPLOYED_YEARS":       employed_years if employed_years > 0 else 0,
        "IS_UNEMPLOYED":        1 if employed_years == 0 else 0,
        "EMPLOYMENT_AGE_RATIO": (employed_years / age_years) if age_years > 0 else 0,
        "EXT_SOURCE_PRODUCT":   ext2 * ext3,
        "EXT_SOURCE_MEAN":      (ext2 + ext3) / 2,
    }
    features.update(_INCOME_TYPES.get(income_type, {}))
    features.update(_EDUCATION_TYPES.get(education, {}))
    features.update(_FAMILY_STATUSES.get(family_status, {}))
    features.update(_HOUSING_TYPES.get(housing_type, {}))
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Customer Input
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=60)
    st.title("Customer Details")

    # Sample profiles for demo
    sample = st.selectbox(
        "Load sample profile",
        ["Custom", "Low Risk Profile", "Medium Risk Profile", "High Risk Profile"],
    )

    SAMPLES = {
        "Low Risk Profile":    dict(income=250_000, credit=500_000, annuity=20_000, age=45, emp=10, ext2=0.82, ext3=0.78, gender="Female", itype="Working", edu="Higher education", fstatus="Married", htype="House / apartment"),
        "Medium Risk Profile": dict(income=100_000, credit=400_000, annuity=22_000, age=30, emp=3,  ext2=0.50, ext3=0.45, gender="Male", itype="Working", edu="Secondary / secondary special", fstatus="Single / not married", htype="Rented apartment"),
        "High Risk Profile":   dict(income=60_000,  credit=450_000, annuity=28_000, age=24, emp=0,  ext2=0.22, ext3=0.18, gender="Male", itype="Unemployed", edu="Lower secondary", fstatus="Civil marriage", htype="With parents"),
    }

    s = SAMPLES.get(sample, {})

    st.divider()
    st.subheader("Financials")
    income  = st.number_input("Annual Income (₹)",     value=s.get("income", 150_000),  step=10_000, min_value=0)
    credit  = st.number_input("Loan Amount (₹)",       value=s.get("credit", 500_000),  step=10_000, min_value=0)
    annuity = st.number_input("Annual Instalment (₹)", value=s.get("annuity", 25_000),  step=1_000,  min_value=0)

    st.subheader("Credit Scores")
    ext2 = st.slider("External Score 2", 0.0, 1.0, float(s.get("ext2", 0.5)), step=0.01,
                     help="Normalised credit score from bureau 2 (0=worst, 1=best)")
    ext3 = st.slider("External Score 3", 0.0, 1.0, float(s.get("ext3", 0.5)), step=0.01,
                     help="Normalised credit score from bureau 3 (0=worst, 1=best)")

    st.subheader("Personal")
    age_years      = st.number_input("Age (years)",      value=s.get("age", 35),  min_value=18, max_value=100)
    employed_years = st.number_input("Years Employed",   value=s.get("emp", 5),   min_value=0,  max_value=60,
                                     help="Set to 0 if currently unemployed")
    gender         = st.radio("Gender", ["Female", "Male"],
                              index=0 if s.get("gender", "Female") == "Female" else 1)
    income_type    = st.selectbox("Income Type",    list(_INCOME_TYPES.keys()),
                                  index=list(_INCOME_TYPES.keys()).index(s.get("itype", "Working")))
    education      = st.selectbox("Education",      list(_EDUCATION_TYPES.keys()),
                                  index=list(_EDUCATION_TYPES.keys()).index(s.get("edu", "Higher education")))
    family_status  = st.selectbox("Family Status",  list(_FAMILY_STATUSES.keys()),
                                  index=list(_FAMILY_STATUSES.keys()).index(s.get("fstatus", "Married")))
    housing_type   = st.selectbox("Housing Type",   list(_HOUSING_TYPES.keys()),
                                  index=list(_HOUSING_TYPES.keys()).index(s.get("htype", "House / apartment")))

    predict_btn = st.button("Predict Default Risk", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main Content — Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "About"])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    st.title("Home Loan Default Risk Assessment")
    st.caption("AI-powered credit risk analysis using XGBoost trained on the Home Credit Default Risk dataset.")

    if predict_btn:
        input_dict = build_feature_dict(
            income, credit, annuity, age_years, employed_years,
            ext2, ext3, gender, income_type, education, family_status, housing_type,
        )

        try:
            prediction, probability = predict_default_risk(input_dict)
            risk_label, risk_color, risk_desc = get_risk_band(probability)

            # ── Key metrics row ───────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Default Probability", f"{probability:.1%}")
            with col2:
                st.metric("Risk Band", risk_label)
            with col3:
                debt_to_income = credit / (income + 1)
                st.metric("Debt-to-Income", f"{debt_to_income:.1f}x")
            with col4:
                monthly_burden = annuity / (income / 12 + 1)
                st.metric("Monthly Burden", f"{monthly_burden:.0%}")

            # ── Risk gauge ────────────────────────────────────────────────
            st.progress(min(float(probability), 1.0))

            risk_css = "risk-high" if prediction == 1 else "risk-low"
            verdict = "High Risk — Likely to Default" if prediction == 1 else "Low Risk — Unlikely to Default"

            st.markdown(
                f'<div class="metric-card {risk_css}">'
                f'<div class="section-header">{verdict}</div>'
                f'<p style="color:#555;margin:0">{risk_desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── SHAP explanation ──────────────────────────────────────────
            st.subheader("Why this prediction?")
            shap_factors = get_shap_explanation(input_dict)

            if shap_factors:
                st.caption("Each factor shows its contribution to the default probability. "
                           "Positive = increases risk, Negative = reduces risk.")

                import matplotlib.pyplot as plt
                import numpy as np

                features = [f["feature"] for f in shap_factors[:10]]
                values   = [f["shap_value"] for f in shap_factors[:10]]

                # Reverse so largest bars are at top
                features = features[::-1]
                values   = values[::-1]

                colors = ["#f44336" if v > 0 else "#4CAF50" for v in values]

                fig, ax = plt.subplots(figsize=(9, 5))
                bars = ax.barh(features, values, color=colors, edgecolor="white", linewidth=0.5)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Value (Impact on Default Probability)")
                ax.set_title("Top Features Driving This Prediction")
                ax.grid(True, axis="x", alpha=0.3)

                # Add value labels
                for bar, val in zip(bars, values):
                    xpos = bar.get_width() + 0.001 if val >= 0 else bar.get_width() - 0.001
                    ha = "left" if val >= 0 else "right"
                    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                            f"{val:+.3f}", va="center", ha=ha, fontsize=8)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            else:
                # Fallback: show manually computed risk drivers
                st.info("SHAP explanation not available for this model version. "
                        "Retrain with `make train` to enable per-prediction explanations.")

                st.write("**Key risk signals from your input:**")
                drivers = [
                    ("Debt-to-Income Ratio",  f"{credit/(income+1):.2f}x",  credit/income > 4, "High ratio increases risk"),
                    ("External Score 2",       f"{ext2:.2f}",               ext2 < 0.4,        "Low score increases risk"),
                    ("External Score 3",       f"{ext3:.2f}",               ext3 < 0.4,        "Low score increases risk"),
                    ("Employment Status",      "Unemployed" if employed_years == 0 else f"{employed_years}y", employed_years == 0, "Unemployment increases risk"),
                    ("Monthly Burden",         f"{annuity/(income/12+1):.0%}", annuity/(income/12+1) > 0.5, "High burden increases risk"),
                ]
                for name, value, is_risk, note in drivers:
                    icon = "🔴" if is_risk else "🟢"
                    st.write(f"{icon} **{name}:** {value} — {note}")

            # ── Recommendation ────────────────────────────────────────────
            st.subheader("Recommendation")
            if probability < 0.10:
                st.success("**Approve** — Strong repayment indicators. Standard terms apply.")
            elif probability < 0.25:
                st.success("**Approve with standard review** — Good risk profile.")
            elif probability < 0.45:
                st.warning("**Manual review required** — Moderate risk. Consider requesting additional documentation or co-applicant.")
            elif probability < 0.65:
                st.error("**Senior review required** — High risk. Consider higher down payment or collateral requirement.")
            else:
                st.error("**Decline recommended** — Very high default risk based on available information.")

        except (FileNotFoundError, RuntimeError) as e:
            st.error(str(e))
            st.info("Run `python src/models/train.py` or `make train` to train the model first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.info("Fill in the customer details in the sidebar and click **Predict Default Risk**.")
        st.markdown("""
        **What this tool does:**
        - Predicts the probability that a loan applicant will default
        - Explains which factors drive the prediction (SHAP values)
        - Provides a risk band and actionable recommendation

        **Model:** XGBoost trained on the Home Credit Default Risk dataset
        **Features used:** Application data + bureau history + payment behavior (~300 features)
        **Evaluation:** 5-fold cross-validated ROC-AUC
        """)

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.title("Model Performance")

    try:
        model_info = get_model_info()

        if "test_roc_auc" in model_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test ROC-AUC", f"{model_info['test_roc_auc']:.4f}")
            with col2:
                st.metric("Test PR-AUC", f"{model_info.get('test_pr_auc', 'N/A')}")
            with col3:
                st.metric("CV AUC Mean", f"{model_info.get('cv_roc_auc_mean', 'N/A')}")
            with col4:
                st.metric("Training Rows", f"{model_info.get('n_train_rows', 'N/A'):,}")

            st.caption(f"Model: **{model_info.get('model_name', 'Unknown')}** | "
                       f"Trained: {model_info.get('trained_at', 'N/A')[:10]} | "
                       f"Features: {model_info.get('n_features', 'N/A')}")

        # Model comparison table
        if "model_comparison" in model_info:
            st.subheader("Model Comparison (5-Fold CV)")
            import pandas as pd
            comp_df = pd.DataFrame(model_info["model_comparison"])
            comp_df.columns = ["Model", "CV AUC Mean", "CV AUC Std", "CV AUC Min", "CV AUC Max"]
            st.dataframe(comp_df.style.highlight_max(subset=["CV AUC Mean"], color="#c8e6c9"), use_container_width=True)

        # Performance plots
        plots_dir = _ROOT / "outputs" / "plots"
        plot_files = {
            "ROC Curve":          plots_dir / f"roc_curve_{model_info.get('model_name', 'xgboost')}.png",
            "Precision-Recall":   plots_dir / f"pr_curve_{model_info.get('model_name', 'xgboost')}.png",
            "Threshold Analysis": plots_dir / "threshold_analysis.png",
            "SHAP Importance":    plots_dir / "shap_global_importance.png",
        }

        available = {name: path for name, path in plot_files.items() if path.exists()}
        if available:
            st.subheader("Performance Plots")
            cols = st.columns(min(2, len(available)))
            for i, (name, path) in enumerate(available.items()):
                with cols[i % 2]:
                    st.image(str(path), caption=name, use_container_width=True)
        else:
            st.info("Performance plots will appear here after training completes. Run `make train`.")

    except Exception:
        st.info("Model not yet trained. Run `python src/models/train.py` or `make train`.")

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.title("About This Project")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Business Problem
        Home Credit Group serves customers who have little or no banking history.
        The challenge: predict whether an applicant will repay their loan or default,
        using non-traditional data sources (mobile usage, transaction history, etc.)
        in addition to standard bureau data.

        A missed default costs the business ~5–10x more than a wrongly rejected application.
        The model therefore optimises for recall (catching defaults) while maintaining
        acceptable precision (not rejecting too many good customers).

        ### Dataset
        - **Source:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)
        - **Scale:** 307,511 loan applications across 7 related tables
        - **Class balance:** 91.9% repay, 8.1% default
        - **Features after engineering:** ~300

        ### Architecture
        ```
        Raw Data (7 CSVs)
            → Feature Engineering (120+ engineered features)
            → sklearn ColumnTransformer (imputation + scaling)
            → XGBoost Classifier (5-fold CV, scale_pos_weight=11)
            → SHAP TreeExplainer (per-prediction explanations)
            → FastAPI (serving) + Streamlit (dashboard)
        ```

        ### Tech Stack
        Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · Optuna · FastAPI · Streamlit · Docker
        """)

    with col2:
        st.markdown("### Key Features")
        features_list = [
            "All 7 dataset tables integrated",
            "120+ engineered features",
            "5-fold stratified cross-validation",
            "Multi-model comparison (LR/RF/XGB/LGBM/CB)",
            "SHAP per-prediction explanations",
            "Optuna hyperparameter tuning",
            "FastAPI serving layer",
            "Docker containerisation",
            "29 unit tests",
            "Complete ML documentation",
        ]
        for f in features_list:
            st.markdown(f"✓ {f}")

        st.markdown("### Documents")
        docs = [
            "PROJECT_JOURNAL.md",
            "DATA_DICTIONARY.md",
            "FEATURE_ENGINEERING.md",
            "MODELING_DECISIONS.md",
            "DEPLOYMENT_GUIDE.md",
            "FUTURE_IMPROVEMENTS.md",
            "PROJECT_ARCHITECTURE.md",
        ]
        for d in docs:
            st.markdown(f"📄 {d}")
