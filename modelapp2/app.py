import streamlit as st
import pandas as pd
import joblib
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)



FUNCTIONS = {
    "Analyze KPIs": None,
    "Customer Churn Prediction": {
        "Customer general data": "churn_model.pkl",
        "Customer credit card data": "churn_cc_model.pkl"
    }
}

# Streamlit UI Layout
st.title("📊 Machine Learning Dashboard")

# Step 1: Select Function
st.header("Select a Function")
function_choice = st.selectbox("Choose a function:", list(FUNCTIONS.keys()))

# Step 2: Show models based on selected function
if function_choice:
    if FUNCTIONS[function_choice] is None:
        st.write("You selected 'Analyze KPIs'. Here we will provide insights into important KPIs.")
        # KPI analysis logic (You can define the KPI analysis here)
        
    else:
        # Show models related to the selected function
        model_choice = st.selectbox(f"Choose a model for {function_choice}:", list(FUNCTIONS[function_choice].keys()))

        # Load the selected model
        MODEL_PATH = os.path.join(SCRIPT_DIR, FUNCTIONS[function_choice][model_choice])

        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file `{FUNCTIONS[function_choice][model_choice]}` not found!")
            st.stop()

        # Load model
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ {model_choice} model loaded successfully!")

        # Sidebar inputs
        st.sidebar.header("Customer Input Features")
        age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.sidebar.number_input("Income ($)", min_value=1000, value=50000)
        account_balance = st.sidebar.number_input("Account Balance ($)", min_value=0, value=20000)

        # Prepare input data
        user_data = pd.DataFrame([[age, income, account_balance]], columns=['age', 'income', 'account_balance'])

        # Prediction button
        if st.sidebar.button("Predict"):
            prediction = model.predict(user_data)[0]
            result_text = "⚠️ High Risk!" if prediction == 1 else "✅ Low Risk!"
            
            # Display prediction
            st.header("Prediction Result")
            st.success(result_text)
