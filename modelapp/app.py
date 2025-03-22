import streamlit as st
import pandas as pd
import joblib

# Load the trained churn model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

model = joblib.load(MODEL_PATH)

# Streamlit UI Layout
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter customer details to predict if they will churn.")

# Sidebar inputs
st.sidebar.header("Customer Input Features")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Income ($)", min_value=1000, value=50000)
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=0, value=20000)

# Prepare input data
user_data = pd.DataFrame([[age, income, account_balance]], columns=['age', 'income', 'account_balance'])

# Prediction button
if st.sidebar.button("Predict"):
    prediction = model.predict(user_data)[0]  # Get prediction (0 = Stay, 1 = Churn)
    result_text = "⚠️ This customer is likely to churn!" if prediction == 1 else "✅ This customer is likely to stay."
    
    # Display prediction
    st.header("Prediction Result")
    st.success(result_text)
