from fastapi import FastAPI
from pydantic import BaseModel
from k_means_code import df_cluster, cluster_names, kmeans2, scaler, cluster_strategy
import numpy as np

app = FastAPI()

class fetchcluster_input(BaseModel):
    customer_id: int
class predictcluster_input(BaseModel):
    age:int
    gender:int
    monthly_income:int
    account_balance:int
    loyalty_score:int
    education_level:int
    facebook_interaction:int
    twitter_interaction:int
    email_interaction:int
    instagram_interaction:int
    total_withdrawal_amount:int
    total_deposit_amount:int
    transaction_count:int
    has_loan:int




# --- 1. Precomputed cluster lookup from DataFrame ---
@app.post("/fetch-cluster")
def fetch_cluster(data: fetchcluster_input):
    try:

        cluster_num = df_cluster.at[data.customer_id, "cluster_num"]
        cluster_name=cluster_names[cluster_num]
        business_strategy=cluster_strategy[cluster_name]
        return f'Cluster {cluster_num}: {cluster_name} , Business strategy: {business_strategy}'
    except KeyError:
        return {"error": f"Customer id {data.customer_id} is not a valid customer id"}


# --- 2. Predict cluster for new input (Real-time segmentation) ---
@app.post("/predict-cluster")
def predict_cluster(data:predictcluster_input):
    features = [
        data.age,
        data.gender,
        data.monthly_income,
        data.account_balance,
        data.loyalty_score,
        data.education_level,
        data.facebook_interaction,
        data.twitter_interaction,
        data.email_interaction,
        data.instagram_interaction,
        data.total_withdrawal_amount,
        data.total_deposit_amount,
        data.transaction_count,
        data.has_loan
    ]
    scaled_input = scaler.transform([features])
    cluster_num = kmeans2.predict(scaled_input)[0]
    cluster_name = cluster_names[cluster_num]
    business_strategy = cluster_strategy[cluster_name]
    return f'Cluster {cluster_num}: {cluster_name} , Business strategy: {business_strategy}'