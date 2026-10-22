import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
from pathlib import Path

from src.underwriting import (
    evaluate_underwriting_terms,
    generate_credit_memo_html,
    generate_credit_memo_json
)

# ==============================================================================
# Page Configuration & Metadata
# ==============================================================================
st.set_page_config(
    page_title="CreditIQ • Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model artifact
MODEL_PATH = Path(__file__).parent / "models" / "credit_risk_model.pkl"
try:
    loaded = joblib.load(MODEL_PATH)
    model = loaded["model"]
    DEFAULT_THRESHOLD = loaded.get("threshold", 0.3)
    target_mapping = loaded.get("target_mapping", {1: "Good Credit", 2: "Bad Credit"})
except Exception as e:
    st.error(f"❌ Failed to load model artifact at `{MODEL_PATH}`: {e}")
    st.stop()

# ==============================================================================
# Human-Readable Mappings for UCI German Credit Codes
# ==============================================================================
LABELS = {
    # Checking account
    "A11": "Negative balance (< 0 DM)",
    "A12": "Modest balance (0 – 200 DM)",
    "A13": "Substantial balance (≥ 200 DM)",
    "A14": "No checking account",
    
    # Credit history
    "A30": "No credits taken / all paid back duly",
    "A31": "All credits at this bank paid back duly",
    "A32": "Existing credits paid back duly till now",
    "A33": "Delay in paying off in the past",
    "A34": "Critical account / other credits elsewhere",
    
    # Purpose
    "A40": "New automobile",
    "A41": "Used automobile",
    "A42": "Furniture / equipment",
    "A43": "Radio / television",
    "A44": "Domestic appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A47": "Vacation / travel",
    "A48": "Retraining",
    "A49": "Business venture",
    "A410": "Other purpose",
    
    # Savings
    "A61": "Very low (< 100 DM)",
    "A62": "Low (100 – 500 DM)",
    "A63": "Medium (500 – 1,000 DM)",
    "A64": "High (≥ 1,000 DM)",
    "A65": "Unknown / no savings account",
    
    # Employment
    "A71": "Unemployed",
    "A72": "Short tenure (< 1 year)",
    "A73": "Moderate tenure (1 – 4 years)",
    "A74": "Established (4 – 7 years)",
    "A75": "Senior tenure (≥ 7 years)",
    
    # Personal status & gender
    "A91": "Male : divorced / separated",
    "A92": "Female : divorced / separated / married",
    "A93": "Male : single",
    "A94": "Male : married / widowed",
    "A95": "Female : single",
    
    # Other debtors
    "A101": "None",
    "A102": "Co-applicant",
    "A103": "Guarantor",
    
    # Property
    "A121": "Real estate property",
    "A122": "Building society / life insurance",
    "A123": "Automobile or other asset",
    "A124": "Unknown / no property",
    
    # Other installment plans
    "A141": "Bank financed",
    "A142": "Department stores",
    "A143": "None",
    
    # Housing
    "A151": "Rented accommodation",
    "A152": "Homeowner (own property)",
    "A153": "Free accommodation",
    
    # Job
    "A171": "Unemployed / unskilled non-resident",
    "A172": "Unskilled resident",
    "A173": "Skilled employee / official",
    "A174": "Management / self-employed / professional",
    
    # Telephone
    "A191": "None registered",
    "A192": "Registered under customer name",
    
    # Foreign worker
    "A201": "Yes",
    "A202": "No"
}

# ==============================================================================
# Custom CSS for Modern, Premium Styling
# ==============================================================================
st.markdown("""
<style>
    /* Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e1b4b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    /* Section Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }

    /* Result Cards */
    .result-card-good {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 26px;
        margin-top: 10px;
        box-shadow: 0 12px 30px -10px rgba(16, 185, 129, 0.25);
    }
    .result-card-bad {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(153, 27, 27, 0.25) 100%);
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 16px;
        padding: 26px;
        margin-top: 10px;
        box-shadow: 0 12px 30px -10px rgba(239, 68, 68, 0.25);
    }
    .badge-good {
        background-color: #059669;
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-bad {
        background-color: #dc2626;
        color: white;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
    }

    /* Metric Highlights */
    .metric-value-good {
        font-size: 2.4rem;
        font-weight: 800;
        color: #34d399;
        line-height: 1.1;
    }
    .metric-value-bad {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f87171;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Button Enhancements */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.35);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }

    /* Underwriting Section Styles */
    .underwrite-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 22px;
        margin-top: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .covenant-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 15px;
    }
    .covenant-item {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-bottom: 8px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    .memo-download-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px dashed rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Presets Helper Function
# ==============================================================================
if "preset_selected" not in st.session_state:
    st.session_state.preset_selected = None

PRESET_PRIME = {
    "duration": 12,
    "credit_amount": 1500,
    "installment_rate": 2,
    "residence_since": 3,
    "age": 38,
    "existing_credits": 1,
    "dependents": 1,
    "checking_account": "A13",  # >= 200 DM
    "credit_history": "A32",    # existing credits paid duly
    "purpose": "A40",           # new car
    "savings": "A64",           # >= 1000 DM
    "employment": "A75",        # >= 7 years
    "personal_status": "A93",   # male single
    "other_debtors": "A101",    # none
    "property": "A121",         # real estate
    "other_installment_plans": "A143", # none
    "housing": "A152",          # own
    "job": "A173",              # skilled employee
    "telephone": "A192",        # yes
    "foreign_worker": "A201"    # yes
}

PRESET_HIGH_RISK = {
    "duration": 48,
    "credit_amount": 9500,
    "installment_rate": 4,
    "residence_since": 1,
    "age": 22,
    "existing_credits": 2,
    "dependents": 1,
    "checking_account": "A11",  # < 0 DM
    "credit_history": "A34",    # critical account
    "purpose": "A49",           # business
    "savings": "A61",           # < 100 DM
    "employment": "A72",        # < 1 year
    "personal_status": "A92",   # female
    "other_debtors": "A101",    # none
    "property": "A124",         # unknown/no property
    "other_installment_plans": "A141", # bank
    "housing": "A151",          # rent
    "job": "A172",              # unskilled
    "telephone": "A191",        # none
    "foreign_worker": "A201"    # yes
}

def apply_preset(preset_dict):
    for k, v in preset_dict.items():
        st.session_state[f"input_{k}"] = v

# ==============================================================================
# Sidebar: Controls, Presets & Documentation
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Quick Actions & Presets")
    st.write("Load pre-configured credit scenarios to evaluate instantly:")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("🌟 Prime (Low Risk)", use_container_width=True):
            apply_preset(PRESET_PRIME)
            st.rerun()
    with col_pre2:
        if st.button("⚠️ High Risk Profile", use_container_width=True):
            apply_preset(PRESET_HIGH_RISK)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Decision Policy")
    threshold = st.slider(
        "Risk Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(DEFAULT_THRESHOLD),
        step=0.05,
        help="Applicants with an estimated default probability above this threshold are classified as Bad Credit."
    )
    st.caption(
        f"**Active Threshold:** `{threshold:.0%}` (Optimized during training to minimize the 5:1 cost of bad-credit defaults)."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Architecture")
    st.markdown("""
    - **Classifier:** Regularized Logistic Regression
    - **Encoder:** Scikit-Learn `ColumnTransformer` (One-Hot)
    - **Scaler:** `StandardScaler` (Sparse preservation)
    - **Dataset:** German Credit (UCI Machine Learning)
    - **Cross-Validation Accuracy:** `77.5%`
    """)

    st.markdown("---")
    st.caption("🔒 *Educational and demonstration prototype. Not intended for commercial credit underwriting.*")

# ==============================================================================
# Main Hero Section
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💳 CreditIQ Risk Intelligence</div>
    <div class="hero-subtitle">
        Enter applicant parameters below to predict default probabilities and determine lending risk based on statistical learning.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Form Layout: 3 Elegant Logical Sections
# ==============================================================================
with st.container():
    col1, col2, col3 = st.columns(3, gap="medium")

    # --- Column 1: Financial & Loan Request ---
    with col1:
        st.markdown('<div class="card-header">💰 Loan & Financial Status</div>', unsafe_allow_html=True)
        
        duration = st.slider(
            "Loan Duration (months)",
            min_value=4,
            max_value=72,
            value=st.session_state.get("input_duration", 18),
            step=1
        )
        
        credit_amount = st.number_input(
            "Credit Amount (DM)",
            min_value=250,
            max_value=25000,
            value=st.session_state.get("input_credit_amount", 3000),
            step=250
        )
        
        checking_options = ["A11", "A12", "A13", "A14"]
        checking_account = st.selectbox(
            "Checking Account Status",
            options=checking_options,
            index=checking_options.index(st.session_state.get("input_checking_account", "A12")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        savings_options = ["A61", "A62", "A63", "A64", "A65"]
        savings = st.selectbox(
            "Savings Balance",
            options=savings_options,
            index=savings_options.index(st.session_state.get("input_savings", "A61")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        purpose_options = ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]
        purpose = st.selectbox(
            "Loan Purpose",
            options=purpose_options,
            index=purpose_options.index(st.session_state.get("input_purpose", "A40")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )

        installment_rate = st.select_slider(
            "Installment Rate (% of disposable income)",
            options=[1, 2, 3, 4],
            value=st.session_state.get("input_installment_rate", 2)
        )

    # --- Column 2: Personal Profile & Employment ---
    with col2:
        st.markdown('<div class="card-header">👤 Personal & Employment</div>', unsafe_allow_html=True)
        
        age = st.slider(
            "Applicant Age (years)",
            min_value=18,
            max_value=80,
            value=st.session_state.get("input_age", 32),
            step=1
        )
        
        employment_options = ["A71", "A72", "A73", "A74", "A75"]
        employment = st.selectbox(
            "Current Employment Tenure",
            options=employment_options,
            index=employment_options.index(st.session_state.get("input_employment", "A73")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        job_options = ["A171", "A172", "A173", "A174"]
        job = st.selectbox(
            "Job Category",
            options=job_options,
            index=job_options.index(st.session_state.get("input_job", "A173")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        personal_status_options = ["A91", "A92", "A93", "A94"]
        personal_status = st.selectbox(
            "Personal Status & Marital Status",
            options=personal_status_options,
            index=personal_status_options.index(st.session_state.get("input_personal_status", "A93")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        dependents = st.selectbox(
            "Number of Dependents",
            options=[1, 2],
            index=[1, 2].index(st.session_state.get("input_dependents", 1)),
            format_func=lambda x: f"{x} dependent(s)"
        )

        foreign_worker_options = ["A201", "A202"]
        foreign_worker = st.selectbox(
            "Foreign Worker Status",
            options=foreign_worker_options,
            index=foreign_worker_options.index(st.session_state.get("input_foreign_worker", "A201")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )

    # --- Column 3: Credit History & Assets ---
    with col3:
        st.markdown('<div class="card-header">🏠 Assets, History & Collateral</div>', unsafe_allow_html=True)
        
        credit_history_options = ["A30", "A31", "A32", "A33", "A34"]
        credit_history = st.selectbox(
            "Credit Repayment History",
            options=credit_history_options,
            index=credit_history_options.index(st.session_state.get("input_credit_history", "A32")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        housing_options = ["A151", "A152", "A153"]
        housing = st.selectbox(
            "Housing Arrangement",
            options=housing_options,
            index=housing_options.index(st.session_state.get("input_housing", "A152")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        property_options = ["A121", "A122", "A123", "A124"]
        property_val = st.selectbox(
            "Most Valuable Property Asset",
            options=property_options,
            index=property_options.index(st.session_state.get("input_property", "A121")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        debtor_options = ["A101", "A102", "A103"]
        other_debtors = st.selectbox(
            "Other Debtors / Guarantors",
            options=debtor_options,
            index=debtor_options.index(st.session_state.get("input_other_debtors", "A101")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        
        plans_options = ["A141", "A142", "A143"]
        other_installment_plans = st.selectbox(
            "Other Existing Installment Plans",
            options=plans_options,
            index=plans_options.index(st.session_state.get("input_other_installment_plans", "A143")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )

        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            existing_credits = st.selectbox(
                "Existing Bank Credits",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(st.session_state.get("input_existing_credits", 1))
            )
        with col_sub2:
            residence_since = st.selectbox(
                "Years at Current Residence",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(st.session_state.get("input_residence_since", 2))
            )

        telephone_options = ["A191", "A192"]
        telephone = st.selectbox(
            "Telephone Registered",
            options=telephone_options,
            index=telephone_options.index(st.session_state.get("input_telephone", "A192")),
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )

# ==============================================================================
# Prediction Action & Inference
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    predict_clicked = st.button("🚀 Evaluate Credit Risk & Generate Terms", use_container_width=True)
with col_btn2:
    if st.button("🔄 Reset Evaluation", use_container_width=True):
        if "evaluation_result" in st.session_state:
            del st.session_state["evaluation_result"]
            st.rerun()

# Build applicant data dictionary with exact feature names
applicant = {
    "checking_account": checking_account,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "savings": savings,
    "employment": employment,
    "installment_rate": installment_rate,
    "personal_status": personal_status,
    "other_debtors": other_debtors,
    "residence_since": residence_since,
    "property": property_val,
    "age": age,
    "other_installment_plans": other_installment_plans,
    "housing": housing,
    "existing_credits": existing_credits,
    "job": job,
    "dependents": dependents,
    "telephone": telephone,
    "foreign_worker": foreign_worker
}

if predict_clicked:
    applicant_df = pd.DataFrame([applicant])
    
    with st.spinner("Analyzing credit parameters and scoring risk probability..."):
        probabilities = model.predict_proba(applicant_df)
        bad_index = list(model.classes_).index(2)
        bad_probability = float(probabilities[0, bad_index])
        
        # Evaluate Underwriting & Pricing Terms
        terms = evaluate_underwriting_terms(
            applicant=applicant,
            bad_probability=bad_probability,
            threshold=threshold
        )
        
        memo_html = generate_credit_memo_html(
            applicant=applicant,
            labels_dict=LABELS,
            terms=terms,
            bad_probability=bad_probability,
            threshold=threshold
        )
        memo_json = generate_credit_memo_json(
            applicant=applicant,
            terms=terms,
            bad_probability=bad_probability,
            threshold=threshold
        )
        
        st.session_state["evaluation_result"] = {
            "bad_probability": bad_probability,
            "threshold": threshold,
            "terms": terms,
            "memo_html": memo_html,
            "memo_json": memo_json,
            "applicant": applicant
        }

if "evaluation_result" in st.session_state:
    res = st.session_state["evaluation_result"]
    bad_probability = res["bad_probability"]
    threshold = res["threshold"]
    terms = res["terms"]
    memo_html = res["memo_html"]
    memo_json = res["memo_json"]

    st.markdown("---")
    st.markdown("### 📋 Credit Risk & Underwriting Assessment Report")

    # Visual Result Card
    card_class = "result-card-bad" if terms.decision == "Declined" else ("result-card-good" if terms.decision == "Approved" else "underwrite-card")
    badge_class = "badge-bad" if terms.decision == "Declined" else "badge-good"
    badge_text = f"{terms.decision.upper()} • {terms.risk_grade}"
    
    if terms.decision == "Approved":
        headline = "✅ Facility Approved (Prime / Low Risk)"
        recommendation = "Applicant displays strong creditworthiness indicators and falls comfortably within bank risk tolerance. Facility recommended at standard prime rates."
    elif terms.decision == "Conditionally Approved":
        headline = "⚠️ Facility Conditionally Approved (Structured Offer)"
        recommendation = "Applicant meets baseline criteria with elevated risk factors. Approved subject to risk-mitigating covenants: adjusted credit limit, duration curtailment, or guarantor endorsement."
    else:
        headline = "⛔ Credit Facility Flagged / Declined"
        recommendation = "Applicant exceeds the acceptable default probability threshold. Primary unsecured facility declined. Restructuring as a secured micro-credit facility or joint-application required for reconsideration."

    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 1.5rem; font-weight: 800; color: #f8fafc;">{headline}</div>
            <span class="{badge_class}">{badge_text}</span>
        </div>
        <p style="color: #cbd5e1; font-size: 1.05rem; margin-bottom: 0px;">
            {recommendation}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Core Probability & Cutoff Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Estimated Default Probability",
            value=f"{bad_probability:.1%}",
            delta=f"{(bad_probability - threshold):+.1%} vs Threshold",
            delta_color="inverse"
        )
    with m_col2:
        st.metric(
            label="Decision Cutoff Threshold",
            value=f"{threshold:.0%}",
            help="Configured in the sidebar based on cost-matrix optimization."
        )
    with m_col3:
        st.metric(
            label="Credit Rating Grade",
            value=f"{terms.risk_grade} ({terms.risk_tier})"
        )
    with m_col4:
        st.metric(
            label="Requested Facility Amount",
            value=f"{terms.requested_amount:,.0f} DM"
        )

    # Visual Probability Gauge
    st.markdown("#### 🎯 Default Risk Spectrum")
    progress_val = min(max(bad_probability, 0.0), 1.0)
    st.progress(progress_val)
    st.caption(
        f"Probability Gauge: Current Score **{bad_probability:.1%}** | Policy Threshold **{threshold:.0%}** | Status: **{terms.decision}**"
    )

    # ==============================================================================
    # Automated Loan Terms Recommendation Engine
    # ==============================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏛️ Automated Loan Terms & Pricing Recommendation")
    st.write(
        "Dynamic credit terms, risk-adjusted interest rate (APR) pricing, and collateral/capacity limits calculated by the Underwriting Engine:"
    )

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.metric(
            label="Recommended APR",
            value=f"{terms.recommended_apr:.2f}%",
            delta=f"{terms.recommended_apr - 5.50:+.2f}% risk spread",
            delta_color="inverse",
            help="Indexed to prime base benchmark (5.50%) + econometric risk premium."
        )
    with t_col2:
        st.metric(
            label="Max Safe Credit Limit",
            value=f"{terms.max_safe_credit_limit:,.0f} DM",
            delta=f"{(terms.max_safe_credit_limit - terms.requested_amount):+,.0f} DM vs Requested",
            help="Maximum prudent exposure calculated from savings, property, and debt service capacity."
        )
    with t_col3:
        st.metric(
            label="Monthly Payment (Requested)",
            value=f"{terms.monthly_payment_requested:,.2f} DM/mo",
            help="Fixed annuity installment for the requested amount and term."
        )
    with t_col4:
        st.metric(
            label="Monthly Payment (Recommended)",
            value=f"{terms.monthly_payment_recommended:,.2f} DM/mo",
            help="Fixed annuity installment for recommended safe limit."
        )

    # Side-by-Side Facility Term Comparison Table
    st.markdown("#### ⚖️ Financing Structure: Requested vs Recommended Offer")
    
    comp_df = pd.DataFrame([
        {
            "Loan Parameter": "Principal Facility Amount",
            "Requested Application": f"{terms.requested_amount:,.0f} DM",
            "Underwriting Recommendation": f"{terms.max_safe_credit_limit:,.0f} DM (Max Safe Ceiling)"
        },
        {
            "Loan Parameter": "Tenure (Maturity)",
            "Requested Application": f"{terms.requested_duration} months",
            "Underwriting Recommendation": f"{min(terms.requested_duration, 36) if terms.decision != 'Approved' else terms.requested_duration} months"
        },
        {
            "Loan Parameter": "Annual Percentage Rate (APR)",
            "Requested Application": f"{terms.recommended_apr:.2f}% (Risk-Indexed)",
            "Underwriting Recommendation": f"{terms.recommended_apr:.2f}%"
        },
        {
            "Loan Parameter": "Monthly Amortization Payment",
            "Requested Application": f"{terms.monthly_payment_requested:,.2f} DM / month",
            "Underwriting Recommendation": f"{terms.monthly_payment_recommended:,.2f} DM / month"
        },
        {
            "Loan Parameter": "Total Interest Payable",
            "Requested Application": f"{terms.total_interest_requested:,.2f} DM",
            "Underwriting Recommendation": f"{terms.total_interest_recommended:,.2f} DM"
        },
        {
            "Loan Parameter": "Total Facility Cost (Principal + Interest)",
            "Requested Application": f"{terms.total_cost_requested:,.2f} DM",
            "Underwriting Recommendation": f"{terms.total_cost_recommended:,.2f} DM"
        },
        {
            "Loan Parameter": "Debt Service Ratio Assessment",
            "Requested Application": terms.debt_service_burden,
            "Underwriting Recommendation": terms.debt_service_burden
        }
    ])
    st.table(comp_df.set_index("Loan Parameter"))

    # Underwriting Covenants & Pre-conditions
    st.markdown("#### 📜 Underwriting Conditions Precedent & Covenants")
    st.markdown("""
    <div class="covenant-box">
    """, unsafe_allow_html=True)
    for cond in terms.conditions:
        st.markdown(f'<div class="covenant-item"><span>📌</span> <strong>{cond}</strong></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================================================
    # Executive Credit Assessment Memo & Export Hub
    # ==============================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📑 Executive Credit Assessment Memo & Export Hub")
    st.write(
        "Generate official audit-ready documentation for credit committee review, compliance archiving, or core banking integration."
    )

    down_col1, down_col2, down_col3 = st.columns([1.5, 1.5, 2])
    with down_col1:
        st.download_button(
            label="📄 Download Executive Memo (HTML / PDF)",
            data=memo_html,
            file_name=f"Credit_Assessment_Memo_{terms.docket_id}.html",
            mime="text/html",
            use_container_width=True
        )
    with down_col2:
        st.download_button(
            label="💾 Export Audit Docket (JSON)",
            data=memo_json,
            file_name=f"Credit_Audit_Docket_{terms.docket_id}.json",
            mime="application/json",
            use_container_width=True
        )
    with down_col3:
        st.caption(f"Docket ID: `{terms.docket_id}` | Generated: `{terms.timestamp}`")

    # Interactive In-App Memorandum Preview
    with st.expander("👁️ View Live Credit Committee Memorandum Preview", expanded=False):
        st.markdown(
            "Below is the exact memorandum generated for this evaluation. You can also print or save directly to PDF using your browser's Print feature (`Ctrl+P` or the print button inside the preview)."
        )
        components.html(memo_html, height=750, scrolling=True)