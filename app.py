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
# Custom CSS for Modern, Premium Styling & Wizard Navigation
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
        padding: 24px 30px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.02rem;
        margin-bottom: 0px;
    }

    /* Stepper Bar */
    .stepper-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 24px;
        flex-wrap: wrap;
        gap: 10px;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.92rem;
        font-weight: 600;
        color: #64748b;
    }
    .step-item.active {
        color: #38bdf8;
    }
    .step-item.completed {
        color: #10b981;
    }
    .step-badge {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        font-weight: 800;
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
    }
    .step-item.active .step-badge {
        background: #3b82f6;
        color: #ffffff;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.5);
    }
    .step-item.completed .step-badge {
        background: #10b981;
        color: #ffffff;
    }
    .step-divider {
        flex-grow: 1;
        height: 2px;
        background: rgba(148, 163, 184, 0.2);
        margin: 0 10px;
        min-width: 20px;
    }
    .step-divider.completed {
        background: #10b981;
    }

    /* Section Cards */
    .wizard-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.25);
    }
    .card-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .step-tip {
        background: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 18px;
        color: #cbd5e1;
        font-size: 0.9rem;
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
        padding: 5px 16px;
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
        padding: 5px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-block;
    }

    /* Underwriting Section Styles */
    .underwrite-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 24px;
        margin-top: 10px;
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

    /* Primary and Secondary Action Buttons */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 24px;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Wizard State Management & Presets
# ==============================================================================
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1

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

def apply_preset(preset_dict, run_eval=True):
    for k, v in preset_dict.items():
        st.session_state[f"input_{k}"] = v
    if run_eval:
        evaluate_and_store(preset_dict)
        st.session_state.wizard_step = 4

def evaluate_and_store(applicant_dict):
    applicant_df = pd.DataFrame([applicant_dict])
    probabilities = model.predict_proba(applicant_df)
    bad_index = list(model.classes_).index(2)
    bad_probability = float(probabilities[0, bad_index])
    curr_threshold = st.session_state.get("policy_threshold", DEFAULT_THRESHOLD)
    
    terms = evaluate_underwriting_terms(
        applicant=applicant_dict,
        bad_probability=bad_probability,
        threshold=curr_threshold
    )
    memo_html = generate_credit_memo_html(
        applicant=applicant_dict,
        labels_dict=LABELS,
        terms=terms,
        bad_probability=bad_probability,
        threshold=curr_threshold
    )
    memo_json = generate_credit_memo_json(
        applicant=applicant_dict,
        terms=terms,
        bad_probability=bad_probability,
        threshold=curr_threshold
    )
    st.session_state["evaluation_result"] = {
        "bad_probability": bad_probability,
        "threshold": curr_threshold,
        "terms": terms,
        "memo_html": memo_html,
        "memo_json": memo_json,
        "applicant": applicant_dict
    }

# ==============================================================================
# Sidebar: Presets, Policy Slider & Wizard Navigator
# ==============================================================================
with st.sidebar:
    st.markdown("### 🧭 Application Wizard")
    st.caption("Jump directly to any section of the application:")
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("1️⃣ Financing", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
        if st.button("3️⃣ Assets & Hist", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
    with col_nav2:
        if st.button("2️⃣ Profile", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
        if st.button("4️⃣ Decision", use_container_width=True):
            st.session_state.wizard_step = 4
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")
    st.write("Load sample underwriting profiles to evaluate instantly:")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("🌟 Prime (Low Risk)", use_container_width=True):
            apply_preset(PRESET_PRIME)
            st.rerun()
    with col_pre2:
        if st.button("⚠️ High Risk", use_container_width=True):
            apply_preset(PRESET_HIGH_RISK)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Decision Policy")
    threshold = st.slider(
        "Risk Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.get("policy_threshold", DEFAULT_THRESHOLD)),
        step=0.05,
        key="policy_threshold",
        help="Applicants with an estimated default probability above this threshold are classified as Bad Credit."
    )
    st.caption(
        f"**Active Cutoff:** `{threshold:.0%}` (Calibrated for a 5:1 misclassification cost matrix)."
    )

    st.markdown("---")
    st.markdown("### 📊 Model Architecture")
    st.markdown("""
    - **Classifier:** Regularized Logistic Regression
    - **Encoder:** Scikit-Learn `ColumnTransformer`
    - **Scaler:** `StandardScaler` (Sparse preservation)
    - **Dataset:** German Credit (UCI ML Repository)
    """)
    st.caption("🔒 *Educational & demonstration prototype.*")

# ==============================================================================
# Main Hero Section
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💳 CreditIQ Underwriting Studio</div>
    <div class="hero-subtitle">
        Step-by-step credit evaluation workflow powered by machine learning and automated risk-adjusted loan pricing.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Stepper Visual Progress Bar
# ==============================================================================
current_step = st.session_state.wizard_step

def get_step_class(step_num):
    if current_step == step_num:
        return "active"
    elif current_step > step_num:
        return "completed"
    return ""

def get_badge_content(step_num):
    if current_step > step_num:
        return "✓"
    return str(step_num)

st.markdown(f"""
<div class="stepper-container">
    <div class="step-item {get_step_class(1)}">
        <span class="step-badge">{get_badge_content(1)}</span>
        <span>1. Loan Request</span>
    </div>
    <div class="step-divider {('completed' if current_step > 1 else '')}"></div>
    <div class="step-item {get_step_class(2)}">
        <span class="step-badge">{get_badge_content(2)}</span>
        <span>2. Personal Profile</span>
    </div>
    <div class="step-divider {('completed' if current_step > 2 else '')}"></div>
    <div class="step-item {get_step_class(3)}">
        <span class="step-badge">{get_badge_content(3)}</span>
        <span>3. Assets & History</span>
    </div>
    <div class="step-divider {('completed' if current_step > 3 else '')}"></div>
    <div class="step-item {get_step_class(4)}">
        <span class="step-badge">{get_badge_content(4)}</span>
        <span>4. Decision & Terms</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to gather current inputs across session state
def get_current_applicant():
    return {
        "checking_account": st.session_state.get("input_checking_account", "A12"),
        "duration": st.session_state.get("input_duration", 18),
        "credit_history": st.session_state.get("input_credit_history", "A32"),
        "purpose": st.session_state.get("input_purpose", "A40"),
        "credit_amount": st.session_state.get("input_credit_amount", 3000),
        "savings": st.session_state.get("input_savings", "A61"),
        "employment": st.session_state.get("input_employment", "A73"),
        "installment_rate": st.session_state.get("input_installment_rate", 2),
        "personal_status": st.session_state.get("input_personal_status", "A93"),
        "other_debtors": st.session_state.get("input_other_debtors", "A101"),
        "residence_since": st.session_state.get("input_residence_since", 2),
        "property": st.session_state.get("input_property", "A121"),
        "age": st.session_state.get("input_age", 32),
        "other_installment_plans": st.session_state.get("input_other_installment_plans", "A143"),
        "housing": st.session_state.get("input_housing", "A152"),
        "existing_credits": st.session_state.get("input_existing_credits", 1),
        "job": st.session_state.get("input_job", "A173"),
        "dependents": st.session_state.get("input_dependents", 1),
        "telephone": st.session_state.get("input_telephone", "A192"),
        "foreign_worker": st.session_state.get("input_foreign_worker", "A201")
    }

# ==============================================================================
# STEP 1: Loan & Financing Request
# ==============================================================================
if current_step == 1:
    st.markdown("""
    <div class="wizard-card">
        <div class="card-header">💰 Step 1 of 4: Loan & Financing Parameters</div>
        <div class="step-tip">
            Specify the requested facility principal, loan term maturity, loan purpose, and debt-service installment rate.
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        duration = st.slider(
            "Loan Duration (months)",
            min_value=4,
            max_value=72,
            value=st.session_state.get("input_duration", 18),
            step=1,
            help="Total repayment maturity period requested by applicant."
        )
        st.session_state["input_duration"] = duration

        credit_amount = st.number_input(
            "Requested Credit Amount (DM)",
            min_value=250,
            max_value=25000,
            value=st.session_state.get("input_credit_amount", 3000),
            step=250,
            help="Principal amount requested in Deutsche Marks (DM)."
        )
        st.session_state["input_credit_amount"] = credit_amount

    with col2:
        purpose_options = ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]
        default_purp = st.session_state.get("input_purpose", "A40")
        purp_index = purpose_options.index(default_purp) if default_purp in purpose_options else 0
        purpose = st.selectbox(
            "Loan Purpose",
            options=purpose_options,
            index=purp_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_purpose"] = purpose

        installment_rate = st.select_slider(
            "Installment Rate (% of disposable income)",
            options=[1, 2, 3, 4],
            value=st.session_state.get("input_installment_rate", 2),
            help="Estimated percentage of applicant's disposable net income dedicated to loan repayment."
        )
        st.session_state["input_installment_rate"] = installment_rate

    # Quick Repayment Projection
    est_monthly_rough = credit_amount / duration
    st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(148, 163, 184, 0.2); border-radius: 10px; padding: 14px; margin-top: 15px; display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #94a3b8; font-size: 0.95rem;">Principal Burden: <strong>{credit_amount:,.0f} DM</strong> over <strong>{duration} months</strong></span>
            <span style="color: #38bdf8; font-weight: 700; font-size: 1.05rem;">Estimated Base Principal: ~{est_monthly_rough:,.2f} DM/mo</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step Navigation
    col_nav_l, col_nav_r = st.columns([3, 1])
    with col_nav_r:
        if st.button("Next: Personal Profile ➔", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()

# ==============================================================================
# STEP 2: Personal & Employment Profile
# ==============================================================================
elif current_step == 2:
    st.markdown("""
    <div class="wizard-card">
        <div class="card-header">👤 Step 2 of 4: Personal Profile & Employment Standing</div>
        <div class="step-tip">
            Evaluate demographic indicators, professional stability, employment tenure, and family dependents.
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        age = st.slider(
            "Applicant Age (years)",
            min_value=18,
            max_value=80,
            value=st.session_state.get("input_age", 32),
            step=1
        )
        st.session_state["input_age"] = age

        employment_options = ["A71", "A72", "A73", "A74", "A75"]
        def_emp = st.session_state.get("input_employment", "A73")
        emp_index = employment_options.index(def_emp) if def_emp in employment_options else 2
        employment = st.selectbox(
            "Current Employment Tenure",
            options=employment_options,
            index=emp_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_employment"] = employment

        job_options = ["A171", "A172", "A173", "A174"]
        def_job = st.session_state.get("input_job", "A173")
        job_index = job_options.index(def_job) if def_job in job_options else 2
        job = st.selectbox(
            "Job Category & Skill Level",
            options=job_options,
            index=job_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_job"] = job

    with col2:
        personal_status_options = ["A91", "A92", "A93", "A94", "A95"]
        def_status = st.session_state.get("input_personal_status", "A93")
        status_index = personal_status_options.index(def_status) if def_status in personal_status_options else 2
        personal_status = st.selectbox(
            "Personal & Marital Status",
            options=personal_status_options,
            index=status_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_personal_status"] = personal_status

        dependents = st.selectbox(
            "Number of Financial Dependents",
            options=[1, 2],
            index=[1, 2].index(st.session_state.get("input_dependents", 1)),
            format_func=lambda x: f"{x} dependent(s) liable"
        )
        st.session_state["input_dependents"] = dependents

        foreign_worker_options = ["A201", "A202"]
        def_foreign = st.session_state.get("input_foreign_worker", "A201")
        foreign_index = foreign_worker_options.index(def_foreign) if def_foreign in foreign_worker_options else 0
        foreign_worker = st.selectbox(
            "Foreign Worker Status",
            options=foreign_worker_options,
            index=foreign_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_foreign_worker"] = foreign_worker

    st.markdown("</div>", unsafe_allow_html=True)

    # Step Navigation
    col_nav_b, col_nav_space, col_nav_n = st.columns([1, 2, 1])
    with col_nav_b:
        if st.button("⬅ Back to Loan Request", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col_nav_n:
        if st.button("Next: Assets & History ➔", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()

# ==============================================================================
# STEP 3: Assets, History & Collateral
# ==============================================================================
elif current_step == 3:
    st.markdown("""
    <div class="wizard-card">
        <div class="card-header">🏠 Step 3 of 4: Financial Assets, Collateral & Credit History</div>
        <div class="step-tip">
            Review existing banking reserves, asset backing, credit repayment track records, and co-applicant guarantees.
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        checking_options = ["A11", "A12", "A13", "A14"]
        def_chk = st.session_state.get("input_checking_account", "A12")
        chk_index = checking_options.index(def_chk) if def_chk in checking_options else 1
        checking_account = st.selectbox(
            "Checking Account Liquidity Status",
            options=checking_options,
            index=chk_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_checking_account"] = checking_account

        savings_options = ["A61", "A62", "A63", "A64", "A65"]
        def_sav = st.session_state.get("input_savings", "A61")
        sav_index = savings_options.index(def_sav) if def_sav in savings_options else 0
        savings = st.selectbox(
            "Savings Reserves Balance",
            options=savings_options,
            index=sav_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_savings"] = savings

        credit_history_options = ["A30", "A31", "A32", "A33", "A34"]
        def_chist = st.session_state.get("input_credit_history", "A32")
        chist_index = credit_history_options.index(def_chist) if def_chist in credit_history_options else 2
        credit_history = st.selectbox(
            "Historical Repayment Performance",
            options=credit_history_options,
            index=chist_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_credit_history"] = credit_history

        property_options = ["A121", "A122", "A123", "A124"]
        def_prop = st.session_state.get("input_property", "A121")
        prop_index = property_options.index(def_prop) if def_prop in property_options else 0
        property_val = st.selectbox(
            "Primary Property / Collateral Asset",
            options=property_options,
            index=prop_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_property"] = property_val

        housing_options = ["A151", "A152", "A153"]
        def_hse = st.session_state.get("input_housing", "A152")
        hse_index = housing_options.index(def_hse) if def_hse in housing_options else 1
        housing = st.selectbox(
            "Residential Housing Arrangement",
            options=housing_options,
            index=hse_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_housing"] = housing

    with col2:
        debtor_options = ["A101", "A102", "A103"]
        def_deb = st.session_state.get("input_other_debtors", "A101")
        deb_index = debtor_options.index(def_deb) if def_deb in debtor_options else 0
        other_debtors = st.selectbox(
            "Other Debtors / Guarantors",
            options=debtor_options,
            index=deb_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_other_debtors"] = other_debtors

        plans_options = ["A141", "A142", "A143"]
        def_plan = st.session_state.get("input_other_installment_plans", "A143")
        plan_index = plans_options.index(def_plan) if def_plan in plans_options else 2
        other_installment_plans = st.selectbox(
            "Other Installment Commitments",
            options=plans_options,
            index=plan_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_other_installment_plans"] = other_installment_plans

        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            existing_credits = st.selectbox(
                "Existing Credits at Bank",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(st.session_state.get("input_existing_credits", 1))
            )
            st.session_state["input_existing_credits"] = existing_credits
        with col_sub2:
            residence_since = st.selectbox(
                "Years at Residence",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(st.session_state.get("input_residence_since", 2))
            )
            st.session_state["input_residence_since"] = residence_since

        telephone_options = ["A191", "A192"]
        def_tel = st.session_state.get("input_telephone", "A192")
        tel_index = telephone_options.index(def_tel) if def_tel in telephone_options else 1
        telephone = st.selectbox(
            "Telephone Registered",
            options=telephone_options,
            index=tel_index,
            format_func=lambda x: f"{x}: {LABELS[x]}"
        )
        st.session_state["input_telephone"] = telephone

    st.markdown("</div>", unsafe_allow_html=True)

    # Step Navigation
    col_nav_b, col_nav_space, col_nav_e = st.columns([1, 1, 2])
    with col_nav_b:
        if st.button("⬅ Back to Profile", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with col_nav_e:
        if st.button("🚀 Submit for Underwriting Evaluation ➔", use_container_width=True):
            applicant_data = get_current_applicant()
            with st.spinner("Executing statistical scoring and underwriting evaluation..."):
                evaluate_and_store(applicant_data)
            st.session_state.wizard_step = 4
            st.rerun()

# ==============================================================================
# STEP 4: Executive Underwriting & Decision Report
# ==============================================================================
elif current_step == 4:
    # Ensure evaluation has been run
    if "evaluation_result" not in st.session_state:
        applicant_data = get_current_applicant()
        with st.spinner("Scoring applicant parameters and generating decision..."):
            evaluate_and_store(applicant_data)

    res = st.session_state["evaluation_result"]
    bad_probability = res["bad_probability"]
    threshold = res["threshold"]
    terms = res["terms"]
    memo_html = res["memo_html"]
    memo_json = res["memo_json"]

    # Header controls
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown("### 📋 Executive Credit Risk & Underwriting Assessment")
    with col_head2:
        if st.button("✏️ Revise Parameters", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()

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

    # Underwriting Terms & Pricing Recommendation
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
    st.markdown('<div class="covenant-box">', unsafe_allow_html=True)
    for cond in terms.conditions:
        st.markdown(f'<div class="covenant-item"><span>📌</span> <strong>{cond}</strong></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Executive Credit Assessment Memo & Export Hub
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

    # Bottom action buttons
    st.markdown("---")
    col_bot1, col_bot2 = st.columns([1, 1])
    with col_bot1:
        if st.button("⬅ Return to Review Application (Step 1)", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col_bot2:
        if st.button("🔄 Reset and Start Fresh Application", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("input_") or k in ["evaluation_result", "preset_selected"]:
                    del st.session_state[k]
            st.session_state.wizard_step = 1
            st.rerun()