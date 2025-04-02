import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# API Base URL
API_BASE_URL = "http://127.0.0.1:5000"

# Set page configuration
st.set_page_config(
    page_title="Customer Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define cluster features
CLUSTER_FEATURES = [
    "age", "gender", "income/month", "account balance", "loyalty score", 
    "education level", "total_withdrawals", "total_deposits", 
    "transaction_count", "Facebook", "Twitter", "Email", "Instagram", "has_loan"
]

# Function to get cluster colors
def get_cluster_color(cluster_id):
    colors = {
        0: "#3498db",  # Blue
        1: "#2ecc71",  # Green
        2: "#e74c3c",  # Red
        3: "#f39c12",  # Orange
    }
    return colors.get(cluster_id, "#95a5a6")  # Default gray

# Function to get all customers
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_all_customers():
    try:
        response = requests.get(f"{API_BASE_URL}/api/customers")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['customers'])
            return df
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return pd.DataFrame()

# Function to get a specific customer
def get_customer(customer_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/customers/{customer_id}")
        if response.status_code == 200:
            return response.json()['customer']
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to assign cluster to new customer
def assign_cluster(customer_data):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/assign_cluster", 
            json=customer_data
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to update customer data
def update_customer(customer_id, updated_data):
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/customers/{customer_id}", 
            json=updated_data
        )
        if response.status_code == 200:
            return True
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return False

# Function to add a new customer
def add_new_customer(customer_data):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/customers", 
            json=customer_data
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# App title
st.title("Customer Clustering Dashboard")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🔍 Fetch Customer", 
    "🔮 Predict Cluster", 
    "✏️ Update Customer", 
    "➕ Add Customer"
])

# Tab 1: Dashboard Information
with tab1:
    st.header("Cluster Dashboard")
    
    # Get all customers data
    df = get_all_customers()
    
    if not df.empty:
        # Convert cluster to numeric if it exists
        if 'cluster' in df.columns:
            df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce')
            
            st.subheader("Customer Distribution by Cluster")
            
            # Count customers by cluster
            cluster_counts = df['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            # Create a bar chart
            fig = px.bar(
                cluster_counts, 
                x='Cluster', 
                y='Count',
                color='Cluster',
                color_discrete_map={
                    0: "#3498db",
                    1: "#2ecc71",
                    2: "#e74c3c",
                    3: "#f39c12"
                },
                labels={'Cluster': 'Cluster ID', 'Count': 'Number of Customers'},
                title='Customer Distribution by Cluster'
            )
            st.plotly_chart(fig)
            
            # Display cluster statistics
            st.subheader("Cluster Statistics")
            
            # Create a DataFrame with customer counts and percentages
            total_customers = len(df)
            cluster_stats = cluster_counts.copy()
            cluster_stats['Percentage'] = (cluster_stats['Count'] / total_customers * 100).round(2)
            
            # Add cluster descriptions
            descriptions = {
                0: "Budget-conscious individuals with moderate engagement",
                1: "High-value customers with strong digital presence",
                2: "Conservative savers with limited digital footprint",
                3: "Young professionals with high transaction frequency"
            }
            
            cluster_stats['Description'] = cluster_stats['Cluster'].map(descriptions)
            
            # Sort by Cluster ID (ascending order)
            cluster_stats = cluster_stats.sort_values('Cluster')
            
            # Display statistics
            st.dataframe(
                cluster_stats[['Cluster', 'Count', 'Percentage', 'Description']],
                column_config={
                    "Cluster": st.column_config.NumberColumn("Cluster ID"),
                    "Count": st.column_config.NumberColumn("Number of Customers"),
                    "Percentage": st.column_config.NumberColumn("% of Total", format="%.2f%%"),
                    "Description": "Cluster Description"
                },
                hide_index=True
            )
        else:
            st.warning("Cluster information not available in the dataset")
    else:
        st.error("Unable to fetch customer data. Please check if the API is running.")

# Tab 2: Fetch Customer Information
with tab2:
    st.header("Fetch Customer Information")
    
    customer_id = st.text_input("Enter Customer ID", "")
    
    if st.button("Fetch Customer"):
        if not customer_id:
            st.warning("Please enter a customer ID")
        else:
            try:
                # Check if ID can be converted to integer
                int(customer_id)
                
                # Modified get_customer function to suppress API error messages
                try:
                    response = requests.get(f"{API_BASE_URL}/api/customers/{customer_id}")
                    if response.status_code == 200:
                        customer = response.json()['customer']
                    else:
                        customer = None
                except Exception:
                    customer = None
                
                if customer:
                    # Create columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Customer Details")
                        
                        # Only include specific fields: customer id, age group, gender, loyalty score, education
                        # Create a cleaner display format with better column names
                        customer_info = {
                            "Attribute": [],
                            "Value": []
                        }
                        
                        # Customer ID
                        customer_info["Attribute"].append("Customer ID")
                        customer_info["Value"].append(customer.get("customer id", "N/A"))
                        
                        # Age Group
                        customer_info["Attribute"].append("Age Group")
                        customer_info["Value"].append(customer.get("age group", "N/A"))
                        
                        # Gender (convert to text for better readability)
                        customer_info["Attribute"].append("Gender")
                        gender_val = customer.get("gender", "N/A")
                        customer_info["Value"].append("Female" if gender_val == 0 else "Male" if gender_val == 1 else gender_val)
                        
                        # Loyalty Score
                        customer_info["Attribute"].append("Loyalty Score")
                        customer_info["Value"].append(customer.get("loyalty score", "N/A"))
                        
                        # Education Level (convert to text if numeric)
                        customer_info["Attribute"].append("Education Level")
                        education_val = customer.get("education level", "N/A")
                        if education_val in [0, 0.0]:
                            education_text = "High School"
                        elif education_val in [0.33, 1/3]:
                            education_text = "Bachelor's"
                        elif education_val in [0.67, 2/3]:
                            education_text = "Master's"
                        elif education_val in [1, 1.0]:
                            education_text = "PhD"
                        else:
                            education_text = education_val
                        customer_info["Value"].append(education_text)
                        
                        # Display as a DataFrame with clear headers
                        df_info = pd.DataFrame(customer_info)
                        st.dataframe(
                            df_info,
                            hide_index=True,
                            column_config={
                                "Attribute": st.column_config.Column("Customer Information", width="medium"),
                                "Value": st.column_config.Column("Details", width="medium")
                            }
                        )
                    
                    with col2:
                        if 'cluster' in customer and 'cluster_description' in customer:
                            st.subheader("Cluster Information")
                            
                            # Display cluster info in a visually appealing way
                            cluster_id = int(customer['cluster'])
                            cluster_desc = customer['cluster_description']
                            
                            # Show cluster card
                            st.markdown(
                                f"""
                                <div style="padding: 20px; border-radius: 10px; background-color: {get_cluster_color(cluster_id)}; color: white;">
                                    <h3 style="margin-top: 0;">Cluster {cluster_id}</h3>
                                    <p>{cluster_desc}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            
                        else:
                            st.warning("Cluster information not available for this customer")
                else:
                    st.error(f"No customer found with ID: {customer_id}. Please check that you've entered a valid customer ID and try again. Valid IDs are integers that exist in the database.")
            except ValueError:
                st.error("Invalid customer ID format. Customer IDs should be integers. Please enter a numeric customer ID (e.g., 1, 2, 3, etc.)")

# Tab 3: Predict Cluster for a New Customer
with tab3:
    st.header("Predict Cluster for a New Customer")
    
    st.subheader("Enter Customer Information")
    st.caption("Fill in the fields below to predict the customer's cluster. Not all fields are required.")
    
    # Create a multi-column layout for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=1.0, value=0.5, step=0.01, 
                             help="Age value (normalized between 0-1). 0 is youngest, 1 is oldest.")
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        income = st.number_input("Monthly Income", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                                help="Income value (normalized between 0-1)")
        balance = st.number_input("Account Balance", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                                 help="Account balance (normalized between 0-1)")
        loyalty = st.number_input("Loyalty Score", min_value=0.0, max_value=1000.0, value=500.0)
        education = st.selectbox("Education Level", options=[0, 0.33, 0.67, 1.0], 
                               format_func=lambda x: ["High School", "Bachelor's", "Master's", "PhD"][int(x*3) if x > 0 else 0])
        
    with col2:
        withdrawals = st.number_input("Total Withdrawals", min_value=0.0, value=500.0)
        deposits = st.number_input("Total Deposits", min_value=0.0, value=500.0)
        transactions = st.number_input("Transaction Count", min_value=0, value=20)
        
        # Social media presence
        st.subheader("Social Media Presence")
        col_fb, col_tw = st.columns(2)
        col_em, col_ig = st.columns(2)
        
        with col_fb:
            facebook = st.checkbox("Facebook", value=True)
        with col_tw:
            twitter = st.checkbox("Twitter", value=False)
        with col_em:
            email = st.checkbox("Email", value=True)
        with col_ig:
            instagram = st.checkbox("Instagram", value=True)
            
        has_loan = st.checkbox("Has Loan", value=False)
    
    # Create the customer data for prediction
    customer_data = {
        "age": float(age),
        "gender": int(gender),
        "income/month": float(income),
        "account balance": float(balance),
        "loyalty score": float(loyalty),
        "education level": float(education),
        "total_withdrawals": float(withdrawals),
        "total_deposits": float(deposits),
        "transaction_count": int(transactions),
        "Facebook": int(facebook),
        "Twitter": int(twitter),
        "Email": int(email),
        "Instagram": int(instagram),
        "has_loan": int(has_loan)
    }
    
    if st.button("Predict Cluster"):
        result = assign_cluster(customer_data)
        
        if result:
            st.success(f"Prediction Successful!")
            
            # Display the result
            cluster_id = result['cluster']
            cluster_desc = result['cluster_description']
            
            # Create a card with cluster information
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {get_cluster_color(cluster_id)}; color: white;">
                    <h2 style="margin-top: 0;">Assigned to Cluster {cluster_id}</h2>
                    <p style="font-size: 18px;">{cluster_desc}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Show any missing features that were filled with defaults
            if 'missing_features' in result and result['missing_features']:
                st.info(f"Some features were not provided and filled with average values: {', '.join(result['missing_features'])}")
            
            # Show the provided data
            st.subheader("Customer Data Used for Prediction")
            st.json(customer_data)
            
            # Option to save this customer (would require additional API endpoint)
            if st.button("Save This Customer"):
                response = add_new_customer(customer_data)
                if response:
                    st.success(f"Customer saved with ID: {response['customer']['customer id']}")
                    st.balloons()

# Tab 4: Update Customer Data
with tab4:
    st.header("Update Customer Data")
    
    # First, search for a customer to update
    customer_id_to_update = st.text_input("Enter Customer ID to Update", "")
    
    if st.button("Search Customer") and customer_id_to_update:
        customer = get_customer(customer_id_to_update)
        
        if customer:
            st.success(f"Customer found! Update the information below.")
            
            # Create a form for updating customer data
            with st.form("update_customer_form"):
                st.subheader("Update Customer Information")
                
                # Create columns for the form
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pre-fill with existing data if available
                    age = st.number_input(
                        "Age", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=float(customer.get("age", 0.5)),
                        step=0.01,
                        help="Age value (normalized between 0-1). 0 is youngest, 1 is oldest."
                    )
                    
                    gender = st.selectbox(
                        "Gender", 
                        options=[0, 1], 
                        format_func=lambda x: "Female" if x == 0 else "Male",
                        index=int(customer.get("gender", 0))
                    )
                    
                    income = st.number_input(
                        "Monthly Income", 
                        min_value=0.0, 
                        max_value=1.0,
                        value=float(customer.get("income/month", 0.5)),
                        step=0.01,
                        help="Income value (normalized between 0-1)"
                    )
                    
                    balance = st.number_input(
                        "Account Balance", 
                        min_value=0.0, 
                        max_value=1.0,
                        value=float(customer.get("account balance", 0.5)),
                        step=0.01,
                        help="Account balance (normalized between 0-1)"
                    )
                    
                    loyalty = st.number_input(
                        "Loyalty Score", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        value=float(customer.get("loyalty score", 500.0))
                    )
                    
                    education_options = [0.0, 0.33, 0.67, 1.0]
                    education_labels = ["High School", "Bachelor's", "Master's", "PhD"]
                    
                    try:
                        education_val = float(customer.get("education level", 0.0))
                        education_default = 0
                        for i, val in enumerate(education_options):
                            if abs(education_val - val) < 0.2:  # Find closest match
                                education_default = i
                                break
                    except (ValueError, TypeError):
                        education_default = 0
                        
                    education = st.selectbox(
                        "Education Level", 
                        options=education_options,
                        format_func=lambda x: education_labels[int(x*3) if x > 0 else 0],
                        index=education_default
                    )
                    
                with col2:
                    withdrawals = st.number_input(
                        "Total Withdrawals", 
                        min_value=0.0, 
                        value=float(customer.get("total_withdrawals", 500.0))
                    )
                    
                    deposits = st.number_input(
                        "Total Deposits", 
                        min_value=0.0, 
                        value=float(customer.get("total_deposits", 500.0))
                    )
                    
                    transactions = st.number_input(
                        "Transaction Count", 
                        min_value=0, 
                        value=int(float(customer.get("transaction_count", 20)))
                    )
                    
                    # Social media presence
                    st.subheader("Social Media Presence")
                    col_fb, col_tw = st.columns(2)
                    col_em, col_ig = st.columns(2)
                    
                    with col_fb:
                        facebook = st.checkbox(
                            "Facebook", 
                            value=bool(int(float(customer.get("Facebook", 1))))
                        )
                    with col_tw:
                        twitter = st.checkbox(
                            "Twitter", 
                            value=bool(int(float(customer.get("Twitter", 0))))
                        )
                    with col_em:
                        email = st.checkbox(
                            "Email", 
                            value=bool(int(float(customer.get("Email", 1))))
                        )
                    with col_ig:
                        instagram = st.checkbox(
                            "Instagram", 
                            value=bool(int(float(customer.get("Instagram", 1))))
                        )
                        
                    has_loan = st.checkbox(
                        "Has Loan", 
                        value=bool(int(float(customer.get("has_loan", 0))))
                    )
                
                # Submit button
                submit_button = st.form_submit_button("Update Customer")
                
                if submit_button:
                    # Create the updated customer data
                    updated_data = {
                        "age": float(age),
                        "gender": int(gender),
                        "income/month": float(income),
                        "account balance": float(balance),
                        "loyalty score": float(loyalty),
                        "education level": float(education),
                        "total_withdrawals": float(withdrawals),
                        "total_deposits": float(deposits),
                        "transaction_count": int(transactions),
                        "Facebook": int(facebook),
                        "Twitter": int(twitter),
                        "Email": int(email),
                        "Instagram": int(instagram),
                        "has_loan": int(has_loan)
                    }
                    
                    # Call the update function
                    success = update_customer(customer_id_to_update, updated_data)
                    
                    if success:
                        st.success("Customer data updated successfully!")
                        st.balloons()
                        
                        # Refresh the customer data to show updated info
                        updated_customer = get_customer(customer_id_to_update)
                        if updated_customer and 'cluster' in updated_customer:
                            st.info(f"Customer is now assigned to Cluster {updated_customer['cluster']}: {updated_customer['cluster_description']}")
                    else:
                        st.error("Failed to update customer data.")
                        
                        # Show what the update would look like
                        st.subheader("Data That Would Be Updated")
                        
                        # Display before and after
                        before_after = pd.DataFrame({
                            'Before': [customer.get(k, '') for k in CLUSTER_FEATURES],
                            'After': [updated_data.get(k, '') for k in CLUSTER_FEATURES]
                        }, index=CLUSTER_FEATURES)
                        
                        st.dataframe(before_after)
        else:
            st.error(f"No customer found with ID: {customer_id_to_update}")

# Tab 5: Add New Customer
with tab5:
    st.header("Add New Customer")
    
    with st.form("add_customer_form"):
        st.subheader("Enter New Customer Information")
        
        # Create columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer ID field (optional)
            new_customer_id = st.text_input("Customer ID (Optional)", key="new_customer_id_input")
            
            age = st.number_input("Age", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="new_age",
                                help="Age value (normalized between 0-1). 0 is youngest, 1 is oldest.")
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="new_gender")
            income = st.number_input("Monthly Income", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="new_income",
                                    help="Income value (normalized between 0-1)")
            balance = st.number_input("Account Balance", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="new_balance",
                                     help="Account balance (normalized between 0-1)")
            loyalty = st.number_input("Loyalty Score", min_value=0.0, max_value=1000.0, value=500.0, key="new_loyalty")
            education = st.selectbox("Education Level", options=[0.0, 0.33, 0.67, 1.0], key="new_education",
                                 format_func=lambda x: ["High School", "Bachelor's", "Master's", "PhD"][int(x*3) if x > 0 else 0])
            
        with col2:
            withdrawals = st.number_input("Total Withdrawals", min_value=0.0, value=500.0, key="new_withdrawals")
            deposits = st.number_input("Total Deposits", min_value=0.0, value=500.0, key="new_deposits")
            transactions = st.number_input("Transaction Count", min_value=0, value=20, key="new_transactions")
            
            # Social media presence
            st.subheader("Social Media Presence")
            col_fb, col_tw = st.columns(2)
            col_em, col_ig = st.columns(2)
            
            with col_fb:
                facebook = st.checkbox("Facebook", value=True, key="new_fb")
            with col_tw:
                twitter = st.checkbox("Twitter", value=False, key="new_tw")
            with col_em:
                email = st.checkbox("Email", value=True, key="new_em")
            with col_ig:
                instagram = st.checkbox("Instagram", value=True, key="new_ig")
                
            has_loan = st.checkbox("Has Loan", value=False, key="new_loan")
        
        # Submit button
        submit_button = st.form_submit_button("Add Customer")
        
        if submit_button:
            # Create the new customer data
            new_customer_data = {
                "customer id": int(new_customer_id) if new_customer_id else None,
                "age": float(age),
                "gender": int(gender),
                "income/month": float(income),
                "account balance": float(balance),
                "loyalty score": float(loyalty),
                "education level": float(education),
                "total_withdrawals": float(withdrawals),
                "total_deposits": float(deposits),
                "transaction_count": int(transactions),
                "Facebook": int(facebook),
                "Twitter": int(twitter),
                "Email": int(email),
                "Instagram": int(instagram),
                "has_loan": int(has_loan)
            }
            
            # Call the add function
            response = add_new_customer(new_customer_data)
            
            if response:
                st.success(f"New customer added successfully with ID: {response['customer']['customer id']}!")
                st.balloons()
                
                # Show the new customer's cluster
                if 'cluster' in response['customer']:
                    cluster_id = int(response['customer']['cluster'])
                    st.info(f"Customer was assigned to Cluster {cluster_id}")
                    
                    # Show cluster description if available
                    result = assign_cluster(new_customer_data)
                    if result:
                        cluster_desc = result['cluster_description']
                        st.markdown(
                            f"""
                            <div style="padding: 20px; border-radius: 10px; background-color: {get_cluster_color(cluster_id)}; color: white;">
                                <h3 style="margin-top: 0;">Cluster {cluster_id}: {cluster_desc}</h3>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.error("Failed to add new customer. Check the API connection.")
                
                # Show what the prediction would be
                result = assign_cluster({k: v for k, v in new_customer_data.items() if k in CLUSTER_FEATURES})
                
                if result:
                    st.subheader("Cluster Prediction for New Customer")
                    cluster_id = result['cluster']
                    cluster_desc = result['cluster_description']
                    
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {get_cluster_color(cluster_id)}; color: white;">
                            <h3 style="margin-top: 0;">Would be assigned to Cluster {cluster_id}</h3>
                            <p>{cluster_desc}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Show the POST data that would be sent
                st.subheader("Customer Data")
                st.json(new_customer_data)

# Footer
st.markdown("---")
st.caption("Customer Clustering Dashboard | Powered by K-means Clustering")

# Display API status
try:
    response = requests.get(f"{API_BASE_URL}")
    if response.status_code == 200:
        st.sidebar.success("✅ API is connected and running")
    else:
        st.sidebar.warning("⚠️ API is running but returned an unexpected response")
except Exception as e:
    st.sidebar.error(f"❌ API is not connected. Make sure the Flask API is running at {API_BASE_URL}")

# Add data information in the sidebar
st.sidebar.title("Data Information")
df = get_all_customers()
if not df.empty:
    st.sidebar.metric("Total Customers", len(df))
    
    if 'cluster' in df.columns:
        clusters = df['cluster'].dropna().astype(int).unique()
        st.sidebar.metric("Number of Clusters", len(clusters))
        
    # Show dataset features
    available_features = [col for col in df.columns if col in CLUSTER_FEATURES]
    st.sidebar.write("Available Features:")
    st.sidebar.write(", ".join(available_features))

# Add help and information
with st.sidebar.expander("Help & Information"):
    st.write("""
    **Customer Clustering Dashboard**
    
    This app allows you to explore customer segments created using K-means clustering.
    
    **Features:**
    - Dashboard: View cluster distribution and insights
    - Fetch Customer: Look up customers and their assigned clusters
    - Predict Cluster: Categorize new customers
    - Update Customer: Modify existing customer data
    - Add Customer: Add new customers to the database
    
    For all input fields that have values between 0-1, these are normalized values where 0 represents the minimum and 1 represents the maximum in the dataset.
    """)


# Version information
st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0.0 | Made with Streamlit")