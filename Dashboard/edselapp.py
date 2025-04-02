import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load the trained model
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("xgboost_model.pkl")  # Ensure the model is saved as .pkl

# Streamlit app title
st.title("ROI Prediction App")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Define input fields based on dataset columns
def user_inputs():
    Click_Through_Rate = st.sidebar.number_input("Feature 1 (e.g., Click-Through Rate)", min_value=0, max_value=999999, step=0.1)
    Cost_Per_Click = st.sidebar.number_input("Feature 2 (e.g., Cost Per Click)", min_value=1, max_value=999999, step=1)
    Campaign_Type = st.sidebar.selectbox("Feature 3 (e.g., Campaign Type)", ["Email", "Influencer", "Social Media", "Display", "Search"])
    Conversion_Rate = st.sidebar.number_input("Feature 4 (e.g., Conversion Rate)", min_value=0, max_value=999999, step=500)
    Engagement_Score = st.sidebar.number_input("Feature 5 (e.g., Engagement Score)", min_value=0, max_value=10, step=1)
    Channel_Used = st.sidebar.selectbox("Feature 6 (e.g., Channel Used)", ["Email", "Facebook", "Website", "Youtube", "Instagram", "Google Ads"])

    # Convert categorical feature
    data = pd.DataFrame({
        "Click_Through_Rate": [Click_Through_Rate],
        "Cost_Per_Click": [Cost_Per_Click],
        "Conversion_Rate": [Conversion_Rate],
        "Engagement_Score": [Engagement_Score],
        "Campaign_Type": [Campaign_Type],
        "Channel_Used": [Channel_Used]
    })

    encoder = LabelEncoder()
    data["Campaign_Type"] = encoder.fit_transform(data["Campaign_Type"])
    data["Channel_Used"] = encoder.fit_transform(data["Channel_Used"])

    return data

# Get user inputs
input_data = user_inputs()

# Predict button
if st.sidebar.button("Predict ROI"):
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction result
    st.subheader(f"Predicted ROI: {prediction[0]:.2f}")
    st.write(f"RSME of the model: {np.sqrt(model.score(input_data, prediction)):.2f}")