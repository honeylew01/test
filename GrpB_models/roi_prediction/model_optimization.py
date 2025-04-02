from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import cv
from roi_regression_xgboost import XGBoostModel


class XGBoostOptimizer:
        
    def __init__(self, file_path, max_evals=50):
        self.file_path = file_path
        self.max_evals = max_evals
        self.model = XGBoostModel(file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.traintestsplit()

    def objective(self, params):
        params['max_depth'] = int(params['max_depth'])  # Convert max_depth to int
        params['n_estimators'] = int(params['n_estimators'])
        
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train, enable_categorical=True)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test, enable_categorical=True)
        
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

        best_num_boost_round = cv_results['test-rmse-mean'].idxmin()

        model = xgb.train(
            params, dtrain, num_boost_round=best_num_boost_round,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20, verbose_eval=50 #maybe change to 50
        )
        
        y_pred = model.predict(dtest)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {'loss': rmse, 'status': STATUS_OK}
    
    def optimize(self):
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'n_estimators': hp.quniform('n_estimators', 100, 2000, 10),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'subsample': hp.uniform('subsample', 0, 1),
            'alpha': hp.uniform('alpha', 0, 10),
            'lambda': hp.uniform('lambda', 0, 10),
            'gamma': hp.uniform('gamma', 0, 10),
            'min_child_weight': hp.uniform('min_child_weight', 1, 15)
        }
        
        trials = Trials()
        best_params = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
        
        print("Best Hyperparameters:", best_params)
        return best_params

# Run optimization
if __name__ == "__main__":
    optimizer = XGBoostOptimizer('GrpB_models/B3_Edsel/marketing_campaign_dataset.csv', max_evals=50)
    best_params = optimizer.optimize()

#Best Hyperparameters: 
# {'alpha': 4.201332861544761, 
# 'colsample_bytree': 0.6463286021553563, 
# 'gamma': 5.356686700691181, 
# 'lambda': 4.405233368619567, 
# 'learning_rate': 0.08647595866291882, 
# 'max_depth': 3.0, 
# 'min_child_weight': 9.635893105010524, 
# 'n_estimators': 270.0, 
# 'subsample': 0.5199596730744189}
        