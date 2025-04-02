import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib  # To load the trained model
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("roi_xgboost.pkl")  # Ensure the model is saved as .pkl

# Streamlit app title
st.title("💵ROI Prediction App")
st.markdown("🚀 Enter your campaign details and get real-time predictions!")




# Sidebar for user inputs
st.sidebar.header("Input Features")



def user_inputs():
    Click_Through_Rate = st.sidebar.number_input("Click-Through Rate", min_value=0.0, max_value=1.0, step=0.001)
    Cost_Per_Click = st.sidebar.number_input("Cost Per Click", min_value=1, max_value=999999, step=1)
    Acquisition_Cost = st.sidebar.number_input("Acquisition Cost", min_value=1, max_value=999999, step=1)
    Campaign_Type = st.sidebar.selectbox("Campaign Type", ["Email", "Influencer", "Social Media", "Display", "Search"])
    Conversion_Rate = st.sidebar.number_input("Conversion Rate", min_value=0, max_value=1, step=500)
    Engagement_Score = st.sidebar.number_input("Engagement Score", min_value=0, max_value=10, step=1)
    Channel_Used = st.sidebar.selectbox("Channel Used", ["Email", "Facebook", "Website", "Youtube", "Instagram", "Google Ads"])
    Target_Audience = st.sidebar.selectbox("Target Audience", ["Men 18-24", "Men 25-34", "All Ages", "Women 25-34", "Women 35-44"])
    Day_Type = st.sidebar.selectbox("Day Type", ["Weekday", "Weekend"])
    Is_Holiday = st.sidebar.selectbox("Is Holiday", ["1", "0"])
    Duration = st.sidebar.selectbox("Duration", ["15", "30", "45", "60"])

    # Convert categorical feature
    data = pd.DataFrame({
        "Campaign_Type": [Campaign_Type],
        "Target_Audience": [Target_Audience],
        "Duration": [Duration],
        "Channel_Used": [Channel_Used],
        "Conversion_Rate": [Conversion_Rate],
        "Acquisition_Cost": [Acquisition_Cost],
        "Engagement_Score": [Engagement_Score],
        "Day_Type": [Day_Type],
        "Click-Through_Rate": [Click_Through_Rate],
        "Cost_Per_Click": [Cost_Per_Click],
        "Is_Holiday": [Is_Holiday]
        
    })

    # Categorical Mapping (Ensure it's consistent with training)
    category_mappings = {
        "Campaign_Type": {"Email": 0, "Influencer": 1, "Social Media": 2, "Display": 3, "Search": 4},
        "Channel_Used": {"Email": 0, "Facebook": 1, "Website": 2, "Youtube": 3, "Instagram": 4, "Google Ads": 5},
        "Target_Audience": {"Men 18-24": 0, "Men 25-34": 1, "All Ages": 2, "Women 25-34": 3, "Women 35-44": 4},
        "Day_Type": {"Weekday": 0, "Weekend": 1},
        "Is_Holiday": {"1": 1, "0": 0},
        "Duration": {"15": 15, "30": 30, "45": 45, "60": 60}
    }

    # Apply categorical mapping
    for col, mapping in category_mappings.items():
        data[col] = data[col].map(mapping)

    return data


# Get user inputs
input_data = user_inputs()
dinput = xgb.DMatrix(input_data)

# Predict button
if st.sidebar.button("Predict ROI"):
    
    # Make prediction
    prediction = model.predict(dinput)
    
    # Display prediction result
    st.subheader(f"Predicted ROI: {prediction[0]:.2f}")
    #st.write(f"RSME of the model: {np.sqrt(model.score(input_data, prediction)):.2f}")
    #st.write("The model's ROI predictions are, on average, {:.2f} away from the actual ROI.".format(np.sqrt(model.score(input_data, prediction))))