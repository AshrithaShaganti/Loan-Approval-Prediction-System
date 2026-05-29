import streamlit as st
import joblib
import numpy as np

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("loan_model.pkl")
scaler = joblib.load("loan_scaler.pkl")

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#1f77b4;
}

.high-risk{
    background:#ffebee;
    padding:20px;
    border-radius:12px;
    color:#b71c1c;
    font-size:24px;
    font-weight:bold;
    text-align:center;
}

.low-risk{
    background:#e8f5e9;
    padding:20px;
    border-radius:12px;
    color:#1b5e20;
    font-size:24px;
    font-weight:bold;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
    "<div class='title'>🏦 AI Loan Approval Prediction System</div>",
    unsafe_allow_html=True
)

st.write(
    "Predict whether a customer is likely to default on a loan."
)

st.divider()

# =====================================
# INPUTS
# =====================================

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    income = st.number_input(
        "Annual Income ($)",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=1000
    )

    loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000,
        max_value=300000,
        value=50000,
        step=1000
    )

    credit_score = st.slider(
        "Credit Score",
        300,
        850,
        650
    )

with col2:

    months_employed = st.slider(
        "Months Employed",
        0,
        120,
        24
    )

    interest_rate = st.slider(
        "Interest Rate (%)",
        1.0,
        30.0,
        10.0
    )

    dti_ratio = st.slider(
        "DTI Ratio",
        0.0,
        1.0,
        0.30
    )

# =====================================
# BUTTON
# =====================================

predict = st.button(
    "🔍 Predict Loan Status",
    use_container_width=True
)

# =====================================
# PREDICTION
# =====================================

if predict:

    data = np.array([[
        income,
        interest_rate,
        loan_amount,
        credit_score,
        age,
        months_employed,
        dti_ratio
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(
        data_scaled
    )[0]

    probability = model.predict_proba(
        data_scaled
    )[0]

    default_prob = probability[1] * 100

    st.divider()

    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Income",
        f"${income:,.0f}"
    )

    c2.metric(
        "Loan Amount",
        f"${loan_amount:,.0f}"
    )

    c3.metric(
        "Credit Score",
        credit_score
    )

    st.write("")

    if prediction == 1:

        st.markdown(
            f"""
            <div class='high-risk'>
            ❌ LOAN REJECTED
            <br><br>
            Default Risk: {default_prob:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class='low-risk'>
            ✅ LOAN APPROVED
            <br><br>
            Default Risk: {default_prob:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.progress(
        min(
            int(default_prob),
            100
        )
    )

    st.info(
        f"Estimated Default Probability: {default_prob:.2f}%"
    )

    st.subheader("Recommendation")

    if default_prob > 70:

        st.error("""
        High Risk Customer

        • Avoid approval
        • Request collateral
        • Reduce loan amount
        """)

    elif default_prob > 40:

        st.warning("""
        Medium Risk Customer

        • Additional verification needed
        • Review credit history
        """)

    else:

        st.success("""
        Low Risk Customer

        • Eligible for approval
        • Standard loan processing
        """)

st.divider()

st.caption(
    "Loan Approval Prediction System | Random Forest Model"
)