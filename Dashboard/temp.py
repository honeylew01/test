'''
# Define input fields based on dataset columns
def user_inputs():
    Click_Through_Rate = st.sidebar.number_input("Feature 1 (e.g., Click-Through Rate)", min_value=0.0, max_value=999999.0, step=0.1)
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

    campaign_mapping = {"Email": 0, "Influencer": 1, "Social Media": 2, "Display": 3, "Search": 4}
    channel_mapping = {"Email": 0, "Facebook": 1, "Website": 2, "Youtube": 3, "Instagram": 4, "Google Ads": 5}

    data["Campaign_Type"] = data["Campaign_Type"].map(campaign_mapping)
    data["Channel_Used"] = data["Channel_Used"].map(channel_mapping)
    
    return data
'''