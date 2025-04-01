import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("cleaned main dataset.csv",delimiter=",")

# features used for clustering
cluster_features = [
    'customer id', 'age', 'age group', 'gender', 'income/month', 'account balance',
    'loyalty score', 'education level', 'Facebook', 'Twitter', 'Email',
    'Instagram', 'prev campaign success', 'total_withdrawals',
    'total_deposits', 'net_transaction', 'transaction_count',
    'housing', 'loan', 'job grouped_retired', 'job grouped_self employed/entrepreneur',
    'job grouped_student', 'job grouped_unemployed', 'job grouped_unknown',
    'job grouped_white collar', 'marital_married', 'marital_single'
]
# same as features above, but only a selected few
less_cluster_features = [
    'age', 'gender', 'income/month', 'account balance',
    'loyalty score', 'education level', 'Facebook', 'Twitter', 'Email',
    'Instagram', 'total_withdrawals',
    'total_deposits', 'transaction_count',
    'loan'
]



# the corresponding label of each cluster number
cluster_names = {
    0: "High-Value Power Users",
    1: "Value-Driven Frequent Users",
    2: "Affluent Inactives",
    3: "Budget-Conscious Occasionals"
}
#the corresponding business strategy for each cluster/segment to improve ROI
cluster_strategy ={
"High-Value Power Users": "Upsell bank products to increase profits",
"Value-Driven Frequent Users": "Build loyalty to reduce churn",
"Affluent Inactives": "Increase engagement to reduce churn",
"Budget-Conscious Occasionals": "Provide incentives like discounts to increase profits"

}

# Drop rows with missing values
df_cluster = df[cluster_features].dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_cluster["cluster_num"]=clusters


#kmeans but with less features
df_cluster2=df[less_cluster_features].dropna()

x2_scaled = scaler.fit_transform(df_cluster2)
kmeans2 = KMeans(n_clusters=4, random_state=42)
clusters2 = kmeans2.fit_predict(x2_scaled)
df_cluster2["cluster_num"] = clusters2




