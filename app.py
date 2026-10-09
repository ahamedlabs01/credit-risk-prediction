import streamlit as st
import pandas as pd
import joblib

# Load the saved model
loaded = joblib.load("models/credit_risk_model.pkl")

model = loaded["model"]
threshold = loaded["threshold"]
target_mapping = loaded["target_mapping"]

# Page configuration
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Prediction")
st.write("Predict credit risk using Machine Learning.")
st.warning("Educational project only. Not financial advice.")

st.header("Applicant Details")

# Numeric inputs
duration = st.number_input("Credit Duration (months)", 1, 72, 12)
credit_amount = st.number_input("Credit Amount", 1, 200000, 2000)
installment_rate = st.selectbox("Installment Rate", [1, 2, 3, 4])
residence_since = st.selectbox("Years at Residence", [1, 2, 3, 4])
age = st.number_input("Age", 18, 100, 30)
existing_credits = st.selectbox("Existing Credits", [1, 2, 3, 4])
dependents = st.selectbox("Dependents", [1, 2])

# Categorical inputs
options = {
    "checking_account": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45",
                "A46", "A47", "A48", "A49", "A410"],
    "savings": ["A61", "A62", "A63", "A64", "A65"],
    "employment": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"]
}

applicant = {
    "checking_account": st.selectbox("Checking Account", options["checking_account"]),
    "duration": duration,
    "credit_history": st.selectbox("Credit History", options["credit_history"]),
    "purpose": st.selectbox("Purpose", options["purpose"]),
    "credit_amount": credit_amount,
    "savings": st.selectbox("Savings", options["savings"]),
    "employment": st.selectbox("Employment", options["employment"]),
    "installment_rate": installment_rate,
    "personal_status": st.selectbox("Personal Status", options["personal_status"]),
    "other_debtors": st.selectbox("Other Debtors", options["other_debtors"]),
    "residence_since": residence_since,
    "property": st.selectbox("Property", options["property"]),
    "age": age,
    "other_installment_plans": st.selectbox("Other Installment Plans", options["other_installment_plans"]),
    "housing": st.selectbox("Housing", options["housing"]),
    "existing_credits": existing_credits,
    "job": st.selectbox("Job", options["job"]),
    "dependents": dependents,
    "telephone": st.selectbox("Telephone", options["telephone"]),
    "foreign_worker": st.selectbox("Foreign Worker", options["foreign_worker"])
}

if st.button("Predict Credit Risk"):
    applicant_df = pd.DataFrame([applicant])

    probabilities = model.predict_proba(applicant_df)
    bad_index = list(model.classes_).index(2)
    bad_probability = probabilities[0, bad_index]

    prediction = 2 if bad_probability >= threshold else 1

    st.subheader("Prediction Result")
    st.write("**Result:**", target_mapping[prediction])
    st.metric("Estimated Bad Credit Probability", f"{bad_probability:.1%}")

    st.caption(
        "This is a model estimate, not a guarantee of creditworthiness."
    )