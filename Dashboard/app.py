import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split




# setting the current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# business questions options
FUNCTIONS = {
    "Analyze KPIs": None, #Question ?
    
    
    
    "Customer Churn Prediction": {      #Question 10
        "Customer general data": "churn_model.pkl",
        "Customer credit card data": "churn_cc_model.pkl"
    }
}






################## Functions for Q10 ##################
def create_interaction_features(df):
    df['Age_Balance'] = df['Age'] * df['Balance']
    df['Age_NumOfProducts'] = df['Age'] * df['NumOfProducts']
    df['Age_IsActiveMember'] = df['Age'] * df['IsActiveMember']
    df['Balance_NumOfProducts'] = df['Balance'] * df['NumOfProducts']
    df['Balance_IsActiveMember'] = df['Balance'] * df['IsActiveMember']
    df['NumOfProducts_IsActiveMember'] = df['NumOfProducts'] * df['IsActiveMember']
    return df

def preprocess_credit_card_data():
    df = pd.read_excel('default of credit card clients.xls', header=1)
    
    # Replace -2 with 0 in pay columns
    df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].replace(-2, 0)
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], drop_first=True)
    
    # Define churn based on payment and usage patterns
    df['Sudden_Large_Payment'] = ((df['PAY_AMT1'] > df['BILL_AMT1'] * 0.9) | (df['PAY_AMT2'] > df['BILL_AMT2'] * 0.9)).astype(int)
    df['Decreasing_Usage'] = ((df['BILL_AMT2'] > df['BILL_AMT1']) & (df['BILL_AMT3'] > df['BILL_AMT2'])).astype(int)
    df['Churn'] = ((df[['BILL_AMT1', 'BILL_AMT2']].sum(axis=1) == 0) | (df['Decreasing_Usage'] == 1) | (df['Sudden_Large_Payment'] == 1)).astype(int)
    
    return df
###############################################################







# Title
st.title("📊 Welcome to Group 13 AI banking system!")





# Step 1: Select Function
st.header("What would you like to do today?")
function_choice = st.selectbox("Choose a function:", list(FUNCTIONS.keys()))





# Step 2: Show models based on selected function
if function_choice:
    if FUNCTIONS[function_choice] is None: 
        st.write("You selected 'Analyze KPIs'. Here we will provide insights into important KPIs.")




    ############### Question 10 ###################
    elif function_choice == "Customer Churn Prediction":  
        model_choice = st.selectbox(f"Choose a model for {function_choice}:", list(FUNCTIONS[function_choice].keys()))

        # Load the selected model
        MODEL_PATH = os.path.join(SCRIPT_DIR, FUNCTIONS[function_choice][model_choice])
        
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file `{FUNCTIONS[function_choice][model_choice]}` not found!")
            st.stop()

        # Load model
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ {model_choice} model loaded successfully!")

        if model_choice == "Customer general data":
            # sidebar inputs
            st.sidebar.header("Customer General Data Inputs")
            # inputs
            credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
            age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
            tenure = st.sidebar.number_input("Time with bank (in years)", min_value=0, value=3)
            balance = st.sidebar.number_input("Balance ($)", min_value=0, value=20000)
            num_of_products = st.sidebar.number_input("Number of Products", min_value=1, value=1)
            has_credit_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
            has_credit_card = 1 if has_credit_card == "Yes" else 0
            is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
            is_active_member = 1 if is_active_member == "Yes" else 0
            estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0, value=50000)
            geography = st.sidebar.selectbox("Where is customer located at", ["France", "Germany", "Spain"])
            geography_germany = 1 if geography == "Germany" else 0
            geography_spain = 1 if geography == "Spain" else 0
            gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
            gender_male = 1 if gender == "Male" else 0

            #input data
            user_data = pd.DataFrame([[credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member,
                                       estimated_salary, geography_germany, geography_spain, gender_male]],
                                     columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                                              'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',
                                              'Geography_Spain', 'Gender_Male'])
            user_data = create_interaction_features(user_data)

            # prediction button
            if st.sidebar.button("Predict"):
                prediction = model.predict(user_data)[0]
                result_text = "⚠️ This customer is likely to churn!" if prediction == 1 else "✅ This customer is likely to stay."
    
                st.header("Prediction Result")
    
                # change background color based on prediction
                if prediction == 1:
                    #  red for churn
                    st.markdown(
                        f"<style>body {{background-color: red;}}</style>", 
                        unsafe_allow_html=True
                    )
                    st.error(result_text)  
                else:
                    # green for stay
                    st.markdown(
                        f"<style>body {{background-color: green;}}</style>", 
                        unsafe_allow_html=True
                    )
                    st.success(result_text) 

        elif model_choice == "Customer credit card data":
            df = preprocess_credit_card_data()

            # sidebar inputs
            st.sidebar.header("Customer Credit Card Data Inputs")
            
            # inputs
            limit_balance = st.sidebar.number_input("Credit Limit", min_value=1, value=10000)
            sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
            sex_value = 1 if sex == "Male" else 2
            education = st.sidebar.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
            education_value = 1 if education == "Graduate School" else (2 if education == "University" else (3 if education == "High School" else 4))
            marriage = st.sidebar.selectbox("Marriage status", ["Married", "Single", "Other"])
            marriage_value = 1 if marriage == "Married" else (2 if marriage == "Single" else 3)
            age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
            
            pay_options = [
                "Paid duly", 
                "1 month delay", 
                "2 months delay", 
                "3 months delay", 
                "4 months delay", 
                "5 months delay", 
                "6 months delay", 
                "7 months delay", 
                "8 months delay", 
                "9 months and above"]
            
            pay_3_text = st.sidebar.selectbox("Repayment status this month", pay_options)
            pay_4_text = st.sidebar.selectbox("Repayment status 1 month ago", pay_options)
            pay_5_text = st.sidebar.selectbox("Repayment status 2 months ago", pay_options)
            pay_6_text = st.sidebar.selectbox("Repayment status 3 months ago", pay_options)

            pay_status_map = {
                "Paid duly": -1,
                "1 month delay": 1,
                "2 months delay": 2,
                "3 months delay": 3,
                "4 months delay": 4,
                "5 months delay": 5,
                "6 months delay": 6,
                "7 months delay": 7,
                "8 months delay": 8,
                "9 months and above": 9}
            
            pay_3 = pay_status_map.get(pay_3_text, -1)
            pay_4 = pay_status_map.get(pay_4_text, -1)
            pay_5 = pay_status_map.get(pay_5_text, -1)
            pay_6 = pay_status_map.get(pay_6_text, -1)
            
            bill_amt3 = st.sidebar.number_input("Bill statement this month", min_value=0, value=1000)
            bill_amt4 = st.sidebar.number_input("Bill statement 1 month ago", min_value=0, value=1000)
            bill_amt5 = st.sidebar.number_input("Bill statement 2 months ago", min_value=0, value=1000)
            bill_amt6 = st.sidebar.number_input("Bill statement 3 months ago", min_value=0, value=1000)
            
            pay_amt3 = st.sidebar.number_input("Amount paid this month", min_value=0, value=200)
            pay_amt4 = st.sidebar.number_input("Amount paid 1 month ago", min_value=0, value=200)
            pay_amt5 = st.sidebar.number_input("Amount paid 2 months ago", min_value=0, value=200)
            pay_amt6 = st.sidebar.number_input("Amount paid 3 months ago", min_value=0, value=200)
            
            column_names = [
                'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                'PAY_3_0', 'PAY_3_1', 'PAY_3_2', 'PAY_3_3', 'PAY_3_4', 'PAY_3_5', 'PAY_3_6',
                'PAY_3_7', 'PAY_3_8', 'PAY_4_0', 'PAY_4_1', 'PAY_4_2', 'PAY_4_3',
                'PAY_4_4', 'PAY_4_5', 'PAY_4_6', 'PAY_4_7', 'PAY_4_8', 'PAY_5_0',
                'PAY_5_2', 'PAY_5_3', 'PAY_5_4', 'PAY_5_5', 'PAY_5_6', 'PAY_5_7',
                'PAY_5_8', 'PAY_6_0', 'PAY_6_2', 'PAY_6_3', 'PAY_6_4', 'PAY_6_5',
                'PAY_6_6', 'PAY_6_7', 'PAY_6_8', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                'BILL_AMT6', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            pay_3_encoded = [1 if pay_3 == i else 0 for i in range(9)]
            pay_4_encoded = [1 if pay_4 == i else 0 for i in range(9)]
            pay_5_encoded = [1 if pay_5 == i else 0 for i in range(9)]
            pay_6_encoded = [1 if pay_6 == i else 0 for i in range(9)]

            pay_5_encoded = pay_5_encoded[:1] + pay_5_encoded[2:] 
            pay_6_encoded = pay_6_encoded[:1] + pay_6_encoded[2:]  

            user_data = pd.DataFrame([[limit_balance, sex_value, education_value, marriage_value, age] + 
                                      pay_3_encoded + pay_4_encoded + pay_5_encoded + pay_6_encoded + 
                                      [bill_amt3, bill_amt4, bill_amt5, bill_amt6] + 
                                      [pay_amt3, pay_amt4, pay_amt5, pay_amt6]],
                                     columns=column_names)

            # prediction button
            if st.sidebar.button("Predict"):
                prediction = model.predict(user_data)[0]
                result_text = "⚠️ This customer is likely to churn!" if prediction == 1 else "✅ This customer is likely to stay."
    
                st.header("Prediction Result")
    
                # change background color
                if prediction == 1:
                    #  red for churn
                    st.markdown(
                        f"<style>body {{background-color: red;}}</style>", 
                        unsafe_allow_html=True
                    )
                    st.error(result_text) 
                else:
                    #  green for stay
                    st.markdown(
                        f"<style>body {{background-color: green;}}</style>", 
                        unsafe_allow_html=True
                    )
                    st.success(result_text)
                    
        ####################################################
