# ============================================================
# 📉 app.py — Customer Churn Prediction Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
import pickle

# ============================================================
# 🌈 PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="📉 Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 🎨 PLOTLY THEME
# ============================================================

pio.templates.default = "plotly_white"

# ============================================================
# 🔄 LOAD MODEL & METRICS (ONCE)
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_pipeline.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("telco_churn_10000.csv")

@st.cache_data
def load_metrics():
    with open("model_metrics.pkl", "rb") as f:
        return pickle.load(f)
    
@st.cache_resource
def load_columns():
    with open("training_columns.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_label_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)
    
model = load_model()
df = load_data()
metrics = load_metrics()

training_columns = load_columns()
label_encoders = load_label_encoders()


# ============================================================
# 🧭 SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Overview",
        "📊 Dataset Overview",
        "🔍 EDA",
        "📈 Model Metrics",
        "📉 Prediction"
    ]
)

# ============================================================
# 🏠 OVERVIEW
# ============================================================

if page == "🏠 Overview":
    st.title("📉 Customer Churn Prediction Dashboard")

    st.markdown("""
    ### 📘 Project Overview  
    This application predicts whether a **customer is likely to churn**
    using a trained **machine learning pipeline**.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", df.shape[0])
    c2.metric("Churned Customers", df[df["Churn"] == "Yes"].shape[0])
    c3.metric(
        "Churn Rate (%)",
        round(df["Churn"].value_counts(normalize=True)["Yes"] * 100, 2)
    )

# ============================================================
# 📊 DATASET OVERVIEW
# ============================================================

elif page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📘 Summary Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ============================================================
# 🔍 EDA
# ============================================================

elif page == "🔍 EDA":
    st.header("🔍 Exploratory Data Analysis")

    fig = px.histogram(df, x="Churn", color="Churn")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="Contract", color="Churn", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x="Churn", y="MonthlyCharges")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 📈 MODEL METRICS
# ============================================================

elif page == "📈 Model Metrics":
    st.header("📈 Model Performance")

    results = pd.read_csv("model_results.csv")
    st.dataframe(results, use_container_width=True)

    fig = px.bar(
        results,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Best Model")
    st.json(metrics)

# ============================================================
# 📉 PREDICTION
# ============================================================

elif page == "📉 Prediction":
    st.header("📉 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input(
            "Monthly Charges", 0.0, 200.0, 70.0
        )
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # ================= CREATE INPUT DATAFRAME (MATCH TRAINING) =================
    # Manual mappings for categorical variables (must match training)
    contract_mapping = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    internet_mapping = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    }

    payment_mapping = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }


    # Create empty DataFrame with training columns
    input_df = pd.DataFrame(columns=training_columns)
    input_df.loc[0] = 0  # initialize all values to 0

    # Fill numerical values
    input_df.at[0, "tenure"] = tenure
    input_df.at[0, "MonthlyCharges"] = monthly_charges

    # Encode categorical inputs
    input_df.at[0, "Contract"] = contract_mapping[contract]
    input_df.at[0, "InternetService"] = internet_mapping[internet]
    input_df.at[0, "PaymentMethod"] = payment_mapping[payment]
    

    if st.button("🔮 Predict Churn"):
        prediction = model.predict(input_df)[0]

        if prediction in [1, "Yes"]:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")
