import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE




############# Model 1 ##################
df = pd.read_csv("Churn_Modelling.csv")


df = df.drop(columns=['RowNumber','CustomerId','Surname'])

#one-hot encoding
df= pd.get_dummies(df,columns=["Geography","Gender"],drop_first=True)


x = df.drop(columns='Exited')
y = df['Exited']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=3101)



def create_interaction_features(df):
    df['Age_Balance'] = df['Age'] * df['Balance']
    df['Age_NumOfProducts'] = df['Age'] * df['NumOfProducts']
    df['Age_IsActiveMember'] = df['Age'] * df['IsActiveMember']
    df['Balance_NumOfProducts'] = df['Balance'] * df['NumOfProducts']
    df['Balance_IsActiveMember'] = df['Balance'] * df['IsActiveMember']
    df['NumOfProducts_IsActiveMember'] = df['NumOfProducts'] * df['IsActiveMember']
    return df

# interaction features for both the train and test
x_train = create_interaction_features(x_train)
x_test = create_interaction_features(x_test)

smote = SMOTE(sampling_strategy=0.5, random_state=3101)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)



# modelling
rf = RandomForestClassifier(class_weight={0:1, 1:5}, random_state=3101)

param_grid = {
    'n_estimators': [1000],  #number of trees
    'max_depth': [ 10], #tree depth
    'min_samples_split': [ 15], #min samples to split
    'min_samples_leaf': [ 5], #min samples per leaf
    'bootstrap': [False]
}

grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='recall', n_jobs=-1)
grid_search.fit(x_resampled, y_resampled)



best_rf = grid_search.best_estimator_

joblib.dump(best_rf, 'churn_model.pkl')
print("Churn model 1 trained & saved!")







############# Model 2 ##################

df = pd.read_excel('default of credit card clients.xls', header=1)



# replace -2 with 0 in pay
df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].replace(-2, 0)

#one-hot encoding
df = pd.get_dummies(df, columns=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], drop_first=True)


#def churn
df['Sudden_Large_Payment'] = ((df['PAY_AMT1'] > df['BILL_AMT1'] * 0.9) | (df['PAY_AMT2'] > df['BILL_AMT2'] * 0.9)).astype(int)
df['Decreasing_Usage'] = ((df['BILL_AMT2'] > df['BILL_AMT1']) &
                          (df['BILL_AMT3'] > df['BILL_AMT2'])).astype(int) # decreasing card usage for last 3 mths


df['Churn'] = ((df[['BILL_AMT1', 'BILL_AMT2']].sum(axis=1) == 0) |  #last 2 mths 0 payments
               (df['Decreasing_Usage'] == 1) |
               (df['Sudden_Large_Payment'] == 1)).astype(int)  #suddenly pay off balance and stop




features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_3_0', 'PAY_3_1', 'PAY_3_2', 'PAY_3_3', 'PAY_3_4', 'PAY_3_5', 
    'PAY_3_6', 'PAY_3_7', 'PAY_3_8', 'PAY_4_0', 'PAY_4_1', 'PAY_4_2', 
    'PAY_4_3', 'PAY_4_4', 'PAY_4_5', 'PAY_4_6', 'PAY_4_7', 'PAY_4_8', 
    'PAY_5_0', 'PAY_5_2', 'PAY_5_3', 'PAY_5_4', 'PAY_5_5', 'PAY_5_6', 
    'PAY_5_7', 'PAY_5_8', 'PAY_6_0', 'PAY_6_2', 'PAY_6_3', 'PAY_6_4', 
    'PAY_6_5', 'PAY_6_6', 'PAY_6_7', 'PAY_6_8', 'BILL_AMT3', 'BILL_AMT4', 
    'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    ] # remove id, default, payamt1, payamt2, pay0, pay2, bill1, bill2, sudden large payment, low credit
target = "Churn"

x = df[features]
y = df[target]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=3101)


#modelling
rf = RandomForestClassifier(class_weight='balanced', random_state=3101)

param_grid = {
    'n_estimators': [ 100],  
    'max_depth': [None],  
    'min_samples_split': [5],  
    'min_samples_leaf': [1],  
    'bootstrap': [ False]
}

grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='recall', n_jobs=-1)
grid_search.fit(x_train, y_train)

best_rf = grid_search.best_estimator_



joblib.dump(best_rf, 'churn_cc_model.pkl')
print("Churn model 2 trained & saved!")
