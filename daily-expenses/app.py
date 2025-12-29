import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Daily Expense Truth",
    layout="wide"
)

st.title("💰 Daily Expense Truth – Behavioral Spending Analysis")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("daily-expenses/daily_expense_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# --------------------------------------------------
# Load ML Model & Encoders
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("daily-expenses/risk_model.pkl")
    encoders = joblib.load("daily-expenses/label_encoders.pkl")
    return model, encoders

model, label_encoders = load_model()

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[df["Date"].min(), df["Date"].max()]
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

mood = st.sidebar.multiselect(
    "Select Mood",
    options=df["Mood"].unique(),
    default=df["Mood"].unique()
)

# --------------------------------------------------
# Apply Filters
# --------------------------------------------------
filtered_df = df[
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1])) &
    (df["Category"].isin(category)) &
    (df["Mood"].isin(mood))
]

# --------------------------------------------------
# KPI Metrics
# --------------------------------------------------
total_spent = filtered_df["Amount"].sum()
avg_spent = filtered_df["Amount"].mean()

col1, col2 = st.columns(2)
col1.metric("💵 Total Expense", f"₹ {total_spent:,.0f}")
col2.metric("📊 Average Expense", f"₹ {avg_spent:,.0f}")

st.markdown("---")

# --------------------------------------------------
# Visualizations
# --------------------------------------------------
col3, col4 = st.columns(2)

# Category-wise spending
fig_category = px.bar(
    filtered_df.groupby("Category", as_index=False)["Amount"].sum(),
    x="Category",
    y="Amount",
    title="Spending by Category"
)
col3.plotly_chart(fig_category, use_container_width=True)

# Planned vs Unplanned
fig_planned = px.pie(
    filtered_df,
    names="Planned",
    values="Amount",
    title="Planned vs Unplanned Spending"
)
col4.plotly_chart(fig_planned, use_container_width=True)

# Mood-wise average spending
fig_mood = px.bar(
    filtered_df.groupby("Mood", as_index=False)["Amount"].mean(),
    x="Mood",
    y="Amount",
    title="Average Spending by Mood"
)
st.plotly_chart(fig_mood, use_container_width=True)

# --------------------------------------------------
# Data Preview
# --------------------------------------------------
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_df)

st.markdown("---")

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.header("🤖 High-Risk Spending Prediction")
st.write("Enter expense details to predict spending risk")

col1, col2 = st.columns(2)

with col1:
    amount_input = st.number_input("Amount", min_value=0)
    category_input = st.selectbox("Category", df["Category"].unique())
    mood_input = st.selectbox("Mood", df["Mood"].unique())

with col2:
    planned_input = st.selectbox("Planned", df["Planned"].unique())
    time_input = st.selectbox("Time of Day", df["Time_of_Day"].unique())

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "Amount": amount_input,
        "Category": category_input,
        "Mood": mood_input,
        "Planned": planned_input,
        "Time_of_Day": time_input
    }])

    try:
        for col in ["Category", "Mood", "Planned", "Time_of_Day"]:
            input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ High Risk Spending Detected!")
        else:
            st.success("✅ Normal Spending Pattern")

    except Exception as e:
        st.error("⚠️ Unable to predict. Input contains unseen values.")
