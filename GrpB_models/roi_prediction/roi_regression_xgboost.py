
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from feature_engineering import FeatureEngineering
from xgboost import cv
import matplotlib.pyplot as plt


class XGBoostModel:

        def __init__(self, file_path):
            self.file_path = file_path

        def encode_categorical_features(self):
            df = FeatureEngineering(self.file_path).add_features()
            le = LabelEncoder()
            df['Day_Type'] = le.fit_transform(df['Day_Type'])
            df['Campaign_Type'] = le.fit_transform(df['Campaign_Type'])
            df['Target_Audience'] = le.fit_transform(df['Target_Audience'])
            df['Channel_Used'] = le.fit_transform(df['Channel_Used'])
            df['Is_Holiday'] = le.fit_transform(df['Is_Holiday'])
            categorical_cols = ['Day_Type', 'Campaign_Type', 'Target_Audience', 'Channel_Used', 'Is_Holiday']
            df[categorical_cols] = df[categorical_cols].apply(lambda x: x.astype('category'))
            return df

        def traintestsplit(self):
            df = self.encode_categorical_features()
            X = df.drop(['ROI'], axis=1)
            y = df['ROI']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        def xgboost_modelling(self):
            X_train, X_test, y_train, y_test = self.traintestsplit()

            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

            params = {
                'booster': 'gbtree',
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'alpha': 4.201332861544761, 
                'colsample_bytree': 0.6463286021553563, 
                'gamma': 5.356686700691181, 
                'lambda': 4.405233368619567, 
                'learning_rate': 0.08647595866291882, 
                'max_depth': 3, 
                'min_child_weight': 9.635893105010524, 
                'n_estimators': 270.0, 
                'subsample': 0.5199596730744189,
                'tree_method': 'approx'
            }

            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                nfold=10,  # 10-fold cross-validation
                early_stopping_rounds=20,
                metrics='rmse',
                as_pandas=True,
                seed=42
                )
            
            print("Best CV score: ", cv_results['test-rmse-mean'].min())
            best_num_boost_round = cv_results['test-rmse-mean'].idxmin()
            print(f"Best number of boosting rounds: {best_num_boost_round}")

            model = xgb.train(
                params, dtrain, num_boost_round = best_num_boost_round,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=20, verbose_eval=50
            )
        
            y_pred = model.predict(dtest)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            return mse, rmse, model

    
