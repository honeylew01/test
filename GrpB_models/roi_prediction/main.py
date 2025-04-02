from data_cleaning import DataCleaning
from feature_engineering import FeatureEngineering
from roi_regression_xgboost import XGBoostModel
from model_optimization import XGBoostOptimizer
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np

file_path = 'marketing_campaign_dataset.csv'


model = XGBoostModel(file_path)
mse, rmse, trained_model = model.xgboost_modelling()
print(f"Results - RMSE: {rmse:.4f}\nThe model's ROI predictions are, on average, {rmse:.4f} away from the actual ROI.")

xgb.plot_importance(trained_model, importance_type="weight", xlabel="Feature Importance", title="XGBoost Feature Importance")
plt.show() #Plots the importance of features in the dataset.

# Save the model
joblib.dump(trained_model, '../../Dashboard/roi_xgboost.pkl')
 #Load the model
#loaded_model = joblib.load('../../Dashboard/roi_xgboost.pkl')