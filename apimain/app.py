import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings


#run the following command on a new terminal to start it
#streamlit run app.py

#suppress warnings
warnings.filterwarnings("ignore")

#page config
st.set_page_config(
    page_title="Bank Customer Segmentation",
    page_icon="💰",
    layout="wide"
)

#title of the page
st.title("Bank Customer Segmentation App")

#this ensures model is only loaded once
@st.cache_resource
def load_model_and_data():
    
    import k_means_code

    #access respective variables from k_means_code
    return {
        'df': k_means_code.df,
        'df_cluster': k_means_code.df_cluster,
        'df_cluster2': k_means_code.df_cluster2,
        'cluster_names': k_means_code.cluster_names,
        'cluster_strategy': k_means_code.cluster_strategy,
        'scaler': k_means_code.scaler,
        'kmeans': k_means_code.kmeans,
        'kmeans2': k_means_code.kmeans2,
        'cluster_features': k_means_code.cluster_features,
        'less_cluster_features': k_means_code.less_cluster_features
    }

#function to save dataframe to csv
def save_dataframe(df, filename="cleaned main dataset.csv"):
    df.to_csv(filename, index=True)
    st.success(f"Data saved successfully to {filename}")

#load model and data
try:
    model_data = load_model_and_data()
    
    #create tabs for different functionalities
    dashboard_tab, get_customer_data_tab, predict_customer_tab, update_customer_tab, add_customer_tab = st.tabs(["Dashboard", "Fetch Customer Information", "Predict New Customer", "Update Customer", "Add New Customer"])
    
    #tab 1: dashboard
    with dashboard_tab:
        #maybe change this header to smth else
        st.header("Dashboard") #header for the tab
        
        st.subheader("Cluster Information") #subheader
        
        #dataframe for displaying cluster info
        cluster_info = pd.DataFrame({
            "Cluster": list(model_data['cluster_names'].values()),
            "Business Strategy": [model_data['cluster_strategy'][name] for name in model_data['cluster_names'].values()]
        })
        
        st.table(cluster_info)
        
        #show distribution of clusters
        st.subheader("Cluster Distribution")
        cluster_counts = model_data['df_cluster']['cluster_num'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster Number', 'Count']
        cluster_counts['Cluster Name'] = cluster_counts['Cluster Number'].map(model_data['cluster_names'])

        ###############################################################################
        #do this later. bar chart looks ugly right now

        #set up the plot with a balanced size
        fig, ax = plt.subplots(figsize=(3, 1.5))
        bars = ax.barh(cluster_counts['Cluster Name'], cluster_counts['Count'], color='skyblue')

        #add title and labels
        ax.set_xlabel('Count', fontsize=3)  
        ax.set_ylabel('Cluster Name', fontsize=3)
        ax.set_title('Number of Counts per Cluster', fontsize=4)

        #x and y axxis ticks
        plt.yticks(rotation=0, fontsize=2)
        plt.xticks(fontsize=4)  

        #annotate the bars with the actual count values
        for bar in bars:
            width = bar.get_width()  #get width of the bar (the count)
            ax.text(width / 2, bar.get_y() + bar.get_height() / 2,  #position the text in the center of the bar
                    f'{int(width)}', ha='center', va='center', fontsize=3)  #entere text inside the bar

        #padding for better spacing
        plt.tight_layout()  #adjusts layout to prevent label overlap

        #show plot
        st.pyplot(fig)
        ###############################################################################    

    #tab 2: fetch Customer Cluster (from FastAPI endpoint 1)
    with get_customer_data_tab:
        st.header("Fetch Existing Customer Information")
        
        #valid customer IDs
        valid_ids = model_data['df_cluster']["customer id"].tolist()
        
        if valid_ids:
            st.write(f"There are {len(valid_ids)} valid customer IDs in the dataset.")
            st.write(f"Please enter an integer from {min(valid_ids)} to {max(valid_ids)}.")
            
            #user input for customer ID
            customer_id = st.number_input("Enter Customer ID", value=valid_ids[0], key="fetch_customer_id")
            
            if st.button("Fetch Information"):
                try:
                    if customer_id not in valid_ids:
                        st.error(f"❌ Invalid Customer ID! Please enter an integer from {min(valid_ids)} to {max(valid_ids)}.")
                        
                    else:
                        #this replicates the API logic
                        cluster_num = model_data['df_cluster'].at[customer_id, "cluster_num"]
                        cluster_name = model_data['cluster_names'][cluster_num]
                        business_strategy = model_data['cluster_strategy'][cluster_name]
                        
                        #display results in a nice format
                        st.success(f"Customer {customer_id} belongs to Cluster {cluster_num}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Cluster Name:** {cluster_name}")
                        with col2:
                            st.info(f"**Business Strategy:** {business_strategy}")
                        
                        #show customer details
                        st.subheader("Customer Details")
                        customer_data = model_data["df"].loc[customer_id]
                        st.dataframe(pd.DataFrame(customer_data).transpose())
                    
                except KeyError:
                    st.error(f"Customer ID {customer_id} not found in the dataset.")
        else:
            st.warning("No valid customer IDs found in the dataset.")
    
    #tab 3: predict cluster for new input (from FastAPI endpoint 2)
    with predict_customer_tab:
        st.header("Predict Cluster for New Customer") #header for tab
        
        #create form for all input features
        with st.form("new_customer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
                account_balance = st.number_input("Account Balance", min_value=0, value=10000)
                loyalty_score = st.slider("Loyalty Score", min_value=0, max_value=10, value=5)
                education_level = st.slider("Education Level", min_value=0, max_value=4, value=2)
                has_loan = st.selectbox("Has Loan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col2:
                facebook_interaction = st.slider("Facebook Interaction", min_value=0, max_value=10, value=5)
                twitter_interaction = st.slider("Twitter Interaction", min_value=0, max_value=10, value=3)
                email_interaction = st.slider("Email Interaction", min_value=0, max_value=10, value=7)
                instagram_interaction = st.slider("Instagram Interaction", min_value=0, max_value=10, value=4)
                total_withdrawal_amount = st.number_input("Total Withdrawal Amount", min_value=0, value=3000)
                total_deposit_amount = st.number_input("Total Deposit Amount", min_value=0, value=5000)
                transaction_count = st.number_input("Transaction Count", min_value=0, value=20)
            
            submitted = st.form_submit_button("Predict Cluster")
            
            if submitted:
                #this replicates the predict-cluster API logic
                features = [
                    age, gender, monthly_income, account_balance, loyalty_score, 
                    education_level, facebook_interaction, twitter_interaction,
                    email_interaction, instagram_interaction, total_withdrawal_amount,
                    total_deposit_amount, transaction_count, has_loan
                ]
                
                scaled_input = model_data['scaler'].transform([features])
                cluster_num = model_data['kmeans2'].predict(scaled_input)[0]
                cluster_name = model_data['cluster_names'][cluster_num]
                business_strategy = model_data['cluster_strategy'][cluster_name]
                
                #display prediction results
                st.success(f"Predicted Cluster: {cluster_num}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Cluster Name:** {cluster_name}")
                with col2:
                    st.info(f"**Business Strategy:** {business_strategy}")
    
    #tab 4: update customer data
    with update_customer_tab:
        st.header("Update Customer Information") #header for tab
        
        #get valid customer IDs
        valid_ids = model_data['df_cluster']["customer id"].tolist()
        
        if valid_ids:
            #create input for customer ID we want to update
            customer_id = st.number_input("Enter Customer ID to Update", value=valid_ids[0], key="update_customer_id")
            
            if st.button("Load Customer Data"):
                try:
                    if customer_id not in valid_ids:
                        st.error(f"❌ Invalid Customer ID! Please enter an integer from {min(valid_ids)} to {max(valid_ids)}.")

                    #get the customer data
                    customer_data = model_data['df'].loc[customer_id].copy()
                    
                    #display current data
                    st.subheader("Current Customer Data")
                    st.dataframe(pd.DataFrame(customer_data).transpose())
                    
                    #create form for updating
                    with st.form("update_customer_form"):
                        st.subheader("Update Customer Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            updated_age = st.number_input("Age", min_value=18, max_value=100, value=int(customer_data['age']) if not pd.isna(customer_data['age']) else 30)
                            updated_gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=int(customer_data['gender']) if not pd.isna(customer_data['gender']) else 0)
                            updated_income = st.number_input("Monthly Income", min_value=0, value=int(customer_data['income/month']) if not pd.isna(customer_data['income/month']) else 5000)
                            updated_balance = st.number_input("Account Balance", min_value=0, value=int(customer_data['account balance']) if not pd.isna(customer_data['account balance']) else 10000)
                            updated_loyalty = st.slider("Loyalty Score", min_value=0, max_value=10, value=int(customer_data['loyalty score']) if not pd.isna(customer_data['loyalty score']) else 5)
                            updated_education = st.slider("Education Level", min_value=0, max_value=4, value=int(customer_data['education level']) if not pd.isna(customer_data['education level']) else 2)
                            updated_loan = st.selectbox("Has Loan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=int(customer_data['loan']) if not pd.isna(customer_data['loan']) else 0)
                        
                        with col2:
                            updated_facebook = st.slider("Facebook Interaction", min_value=0, max_value=10, value=int(customer_data['Facebook']) if not pd.isna(customer_data['Facebook']) else 5)
                            updated_twitter = st.slider("Twitter Interaction", min_value=0, max_value=10, value=int(customer_data['Twitter']) if not pd.isna(customer_data['Twitter']) else 3)
                            updated_email = st.slider("Email Interaction", min_value=0, max_value=10, value=int(customer_data['Email']) if not pd.isna(customer_data['Email']) else 7)
                            updated_instagram = st.slider("Instagram Interaction", min_value=0, max_value=10, value=int(customer_data['Instagram']) if not pd.isna(customer_data['Instagram']) else 4)
                            updated_withdrawal = st.number_input("Total Withdrawal Amount", min_value=0, value=int(customer_data['total_withdrawals']) if not pd.isna(customer_data['total_withdrawals']) else 3000)
                            updated_deposit = st.number_input("Total Deposit Amount", min_value=0, value=int(customer_data['total_deposits']) if not pd.isna(customer_data['total_deposits']) else 5000)
                            updated_transactions = st.number_input("Transaction Count", min_value=0, value=int(customer_data['transaction_count']) if not pd.isna(customer_data['transaction_count']) else 20)
                        
                        update_submitted = st.form_submit_button("Update Customer Information")
                        
                        if update_submitted:
                            #update customer data in the dataframe
                            model_data['df'].at[customer_id, 'age'] = updated_age
                            model_data['df'].at[customer_id, 'gender'] = updated_gender
                            model_data['df'].at[customer_id, 'income/month'] = updated_income
                            model_data['df'].at[customer_id, 'account balance'] = updated_balance
                            model_data['df'].at[customer_id, 'loyalty score'] = updated_loyalty
                            model_data['df'].at[customer_id, 'education level'] = updated_education
                            model_data['df'].at[customer_id, 'loan'] = updated_loan
                            model_data['df'].at[customer_id, 'Facebook'] = updated_facebook
                            model_data['df'].at[customer_id, 'Twitter'] = updated_twitter
                            model_data['df'].at[customer_id, 'Email'] = updated_email
                            model_data['df'].at[customer_id, 'Instagram'] = updated_instagram
                            model_data['df'].at[customer_id, 'total_withdrawals'] = updated_withdrawal
                            model_data['df'].at[customer_id, 'total_deposits'] = updated_deposit
                            model_data['df'].at[customer_id, 'transaction_count'] = updated_transactions
                            
                            #re-calculate net transaction
                            model_data['df'].at[customer_id, 'net_transaction'] = updated_deposit - updated_withdrawal
                            
                            #update customer's cluster assignment
                            features = [
                                updated_age, updated_gender, updated_income, updated_balance, updated_loyalty,
                                updated_education, updated_facebook, updated_twitter, updated_email,
                                updated_instagram, updated_withdrawal, updated_deposit, updated_transactions, updated_loan
                            ]
                            
                            #prepare feature vector for reduced feature set
                            features_array = np.array(features).reshape(1, -1)
                            
                            #standardise and predict
                            scaled_input = model_data['scaler'].transform(features_array)
                            new_cluster_num = model_data['kmeans2'].predict(scaled_input)[0]
                            
                            #update cluster number if the customer is in df_cluster2
                            if customer_id in model_data['df_cluster2'].index:
                                model_data['df_cluster2'].at[customer_id, 'cluster_num'] = new_cluster_num
                            
                            #get new cluster information
                            cluster_name = model_data['cluster_names'][new_cluster_num]
                            business_strategy = model_data['cluster_strategy'][cluster_name]
                            
                            #save changes
                            save_dataframe(model_data['df'])
                            
                            #display success message
                            st.success(f"Customer {customer_id} information updated successfully!")
                            st.info(f"New cluster assignment: Cluster {new_cluster_num} - {cluster_name}")
                            st.info(f"Business Strategy: {business_strategy}")
                            
                            #display updated data
                            st.subheader("Updated Customer Data")
                            st.dataframe(pd.DataFrame(model_data['df'].loc[customer_id]).transpose())
                
                except KeyError:
                    st.error(f"Customer ID {customer_id} not found in the dataset.")
        else:
            st.warning("No valid customer IDs found in the dataset.")
    
    #tab 5: add new customer
    with add_customer_tab:
        st.header("Add New Customer")
        
        with st.form("add_new_customer_form"):
            st.subheader("Enter New Customer Information")
            
            #get the next available customer ID
            next_id = model_data['df']["customer id"].max() + 1 if not model_data['df'].empty else 1
            
            #show the user their customer id
            st.write(f"New Customer ID will be: {next_id}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_age = st.number_input("Age", min_value=18, max_value=100, value=30, key="new_age")
                new_gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="new_gender")
                new_income = st.number_input("Monthly Income", min_value=0, value=5000, key="new_income")
                new_balance = st.number_input("Account Balance", min_value=0, value=10000, key="new_balance")
                new_loyalty = st.slider("Loyalty Score", min_value=0, max_value=10, value=5, key="new_loyalty")
                new_education = st.slider("Education Level", min_value=0, max_value=4, value=2, key="new_education")
                new_loan = st.selectbox("Has Loan", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="new_loan")
                new_housing = st.selectbox("Has Housing", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="new_housing")
                
                #job options based on dataset
                job_options = ["retired", "self employed/entrepreneur", "student", "unemployed", "unknown", "white collar"]
                new_job = st.selectbox("Job", options=job_options, key="new_job")
                
                #marital status options
                marital_options = ["single", "married", "other"]
                new_marital = st.selectbox("Marital Status", options=marital_options, key="new_marital")
            
            with col2:
                new_facebook = st.slider("Facebook Interaction", min_value=0, max_value=10, value=5, key="new_facebook")
                new_twitter = st.slider("Twitter Interaction", min_value=0, max_value=10, value=3, key="new_twitter")
                new_email = st.slider("Email Interaction", min_value=0, max_value=10, value=7, key="new_email")
                new_instagram = st.slider("Instagram Interaction", min_value=0, max_value=10, value=4, key="new_instagram")
                new_withdrawal = st.number_input("Total Withdrawal Amount", min_value=0, value=3000, key="new_withdrawal")
                new_deposit = st.number_input("Total Deposit Amount", min_value=0, value=5000, key="new_deposit")
                new_transactions = st.number_input("Transaction Count", min_value=0, value=20, key="new_transactions")
                new_campaign = st.selectbox("Previous Campaign Success", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="new_campaign")
            
            add_submitted = st.form_submit_button("Add New Customer")
            
            if add_submitted:
                #create a new customer record
                new_customer = {
                    'age': new_age,
                    'age group': 'young' if new_age < 30 else 'middle' if new_age < 60 else 'senior',
                    'gender': new_gender,
                    'income/month': new_income,
                    'account balance': new_balance,
                    'loyalty score': new_loyalty,
                    'education level': new_education,
                    'Facebook': new_facebook,
                    'Twitter': new_twitter,
                    'Email': new_email,
                    'Instagram': new_instagram,
                    'prev campaign success': new_campaign,
                    'total_withdrawals': new_withdrawal,
                    'total_deposits': new_deposit,
                    'net_transaction': new_deposit - new_withdrawal,
                    'transaction_count': new_transactions,
                    'housing': new_housing,
                    'loan': new_loan
                }
                
                #add job columns (one-hot encoded)
                for job_type in job_options:
                    column_name = f"job grouped_{job_type}"
                    new_customer[column_name] = 1 if new_job == job_type else 0
                
                #add marital columns (one-hot encoded)
                for marital_status in ["single", "married"]:
                    column_name = f"marital_{marital_status}"
                    new_customer[column_name] = 1 if new_marital == marital_status else 0
                
                #add the new customer to the dataframe
                model_data['df'].loc[next_id] = pd.Series(new_customer)
                
                #predict cluster for new customer
                features = [
                    new_age, new_gender, new_income, new_balance, new_loyalty,
                    new_education, new_facebook, new_twitter, new_email,
                    new_instagram, new_withdrawal, new_deposit, new_transactions, new_loan
                ]
                
                #prepare feature vector for reduced feature set
                features_array = np.array(features).reshape(1, -1)
                
                #standardise and predict
                scaled_input = model_data['scaler'].transform(features_array)
                new_cluster_num = model_data['kmeans2'].predict(scaled_input)[0]
                
                #get new cluster information
                cluster_name = model_data['cluster_names'][new_cluster_num]
                business_strategy = model_data['cluster_strategy'][cluster_name]
                
                #save changes
                save_dataframe(model_data['df'])
                
                #display success message
                st.success(f"New customer added successfully with ID: {next_id}")
                st.info(f"Cluster assignment: Cluster {new_cluster_num} - {cluster_name}")
                st.info(f"Business Strategy: {business_strategy}")
                
                #display new customer data
                st.subheader("New Customer Data")
                st.dataframe(pd.DataFrame(model_data['df'].loc[next_id]).transpose())

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please make sure your k_means_code.py file is in the same directory and contains the expected variables.")
    st.write("The application expects 'cleaned main dataset.csv' to be available in the current directory.")