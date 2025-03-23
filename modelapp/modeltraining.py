import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("customer_data.csv")  # Ensure this dataset exists

# Feature selection
X = data[['age', 'income', 'account_balance']]
y = data['churn']  # Target variable (1 = churn, 0 = stay)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Churn Prediction Model
churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
churn_model.fit(X_train, y_train)

# Save the model
joblib.dump(churn_model, "model.pkl")
print("✅ Model trained & saved!")
