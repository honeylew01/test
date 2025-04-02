from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
import json

app = Flask(__name__)

# HTML template for the home page
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Clustering API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 20px; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
        code { font-family: monospace; }
        .endpoint { background-color: #e9f7fe; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { font-weight: bold; color: #0066cc; }
    </style>
</head>
<body>
    <h1>Customer Clustering API Documentation</h1>
    <p>This API provides endpoints to retrieve customer information and assign clusters based on K-means clustering.</p>
    
    <h2>Available Endpoints:</h2>
    
    <div class="endpoint">
        <p><span class="method">GET</span> <code>/api/customers</code></p>
        <p>Retrieve all customers with their cluster assignments.</p>
        <p>Example: <code>GET http://127.0.0.1:5000/api/customers</code></p>
    </div>
    
    <div class="endpoint">
        <p><span class="method">GET</span> <code>/api/customers/&lt;customer_id&gt;</code></p>
        <p>Retrieve a specific customer by ID with their cluster information.</p>
        <p>Example: <code>GET http://127.0.0.1:5000/api/customers/123</code></p>
    </div>
    
    <div class="endpoint">
        <p><span class="method">POST</span> <code>/api/customers</code></p>
        <p>Add a new customer to the database.</p>
        <p>Example: <code>POST http://127.0.0.1:5000/api/customers</code></p>
    </div>
    
    <div class="endpoint">
        <p><span class="method">PUT</span> <code>/api/customers/&lt;customer_id&gt;</code></p>
        <p>Update an existing customer's information.</p>
        <p>Example: <code>PUT http://127.0.0.1:5000/api/customers/123</code></p>
    </div>
    
    <div class="endpoint">
        <p><span class="method">POST</span> <code>/api/assign_cluster</code></p>
        <p>Assign a cluster to a customer based on provided features.</p>
        <p>Example request:</p>
        <pre>
POST http://127.0.0.1:5000/api/assign_cluster
Content-Type: application/json

{
    "age": 30,
    "gender": 1,
    "income/month": 5000,
    "account balance": 12000,
    "loyalty score": 500,
    "education level": 2,
    "total_withdrawals": 3000,
    "total_deposits": 5000,
    "transaction_count": 25,
    "Facebook": 1,
    "Twitter": 0,
    "Email": 1,
    "Instagram": 1,
    "has_loan": 0
}
        </pre>
    </div>
    
    <div class="endpoint">
        <p><span class="method">POST</span> <code>/api/retrain</code></p>
        <p>Retrain the clustering model with updated data.</p>
        <p>Example: <code>POST http://127.0.0.1:5000/api/retrain</code></p>
    </div>
</body>
</html>
"""

# Path constants
DATA_PATH = "generated_bank_data.csv"
MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"

# Features used for clustering
CLUSTER_FEATURES = [
    "age", "gender", "income/month", "account balance", "loyalty score", 
    "education level", "total_withdrawals", "total_deposits", 
    "transaction_count", "Facebook", "Twitter", "Email", "Instagram", "has_loan"
]

# Load or train the model
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # Load existing model and scaler
        with open(MODEL_PATH, 'rb') as f:
            kmeans = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    else:
        # Train new model
        df = pd.read_csv(DATA_PATH)
        
        # Ensure all features are present
        for feature in CLUSTER_FEATURES:
            if feature not in df.columns:
                print(f"Warning: Feature '{feature}' not found in dataset")
        
        # Select only available features
        available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
        df_cluster = df[available_features].dropna()
        
        # Convert features to numeric
        for col in df_cluster.columns:
            df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)
        
        # Run KMeans with k=4
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(X_scaled)
        
        # Save model and scaler
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(kmeans, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    
    return kmeans, scaler

# Initialize model and scaler
kmeans, scaler = load_or_train_model()

# Helper function to get cluster description
def get_cluster_description(cluster_id):
    descriptions = {
        0: "Budget-conscious individuals with moderate engagement",
        1: "High-value customers with strong digital presence",
        2: "Conservative savers with limited digital footprint",
        3: "Young professionals with high transaction frequency"
    }
    return descriptions.get(cluster_id, "Unknown cluster")

@app.route('/')
def home():
    """Home page with API documentation"""
    return HOME_HTML

@app.route('/api/customers', methods=['GET'])
def get_customers():
    """Retrieve all customers with their cluster information"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Add cluster information if not already present
        if 'cluster' not in df.columns:
            # Select only available features
            available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
            df_cluster = df[available_features].dropna()
            
            # Convert features to numeric
            for col in df_cluster.columns:
                df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
            
            # Scale and predict
            X_scaled = scaler.transform(df_cluster)
            clusters = kmeans.predict(X_scaled)
            df.loc[df_cluster.index, 'cluster'] = clusters
        
        # Convert to dictionary for JSON response
        customers = df.fillna('').to_dict(orient='records')
        
        # Add cluster descriptions
        for customer in customers:
            if 'cluster' in customer and customer['cluster'] != '':
                cluster_id = int(customer['cluster'])
                customer['cluster_description'] = get_cluster_description(cluster_id)
        
        return jsonify({
            'status': 'success',
            'count': len(customers),
            'customers': customers
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/customers/<customer_id>', methods=['GET'])
def get_customer(customer_id):
    """Retrieve a specific customer by ID"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        if 'customer id' not in df.columns:
            return jsonify({
                'status': 'error',
                'message': 'Customer ID column not found in data'
            }), 404
        
        # Find the customer
        customer = df[df['customer id'] == int(customer_id)]
        
        if customer.empty:
            return jsonify({
                'status': 'error',
                'message': f'Customer with ID {customer_id} not found'
            }), 404
        
        # Add cluster info if possible
        available_features = [f for f in CLUSTER_FEATURES if f in customer.columns]
        if len(available_features) > 0:
            customer_features = customer[available_features].dropna()
            
            # Convert features to numeric
            for col in customer_features.columns:
                customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce')
            
            if not customer_features.empty:
                X_scaled = scaler.transform(customer_features)
                cluster = kmeans.predict(X_scaled)[0]
                customer_data = customer.iloc[0].to_dict()
                customer_data['cluster'] = int(cluster)
                customer_data['cluster_description'] = get_cluster_description(cluster)
                return jsonify({
                    'status': 'success',
                    'customer': customer_data
                })
        
        # Return without cluster info if we couldn't calculate it
        return jsonify({
            'status': 'success',
            'customer': customer.iloc[0].to_dict()
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/customers/<customer_id>', methods=['PUT'])
def update_customer(customer_id):
    """Update a customer's information"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        if 'customer id' not in df.columns:
            return jsonify({
                'status': 'error',
                'message': 'Customer ID column not found in data'
            }), 404
        
        # Find the customer
        customer_idx = df[df['customer id'] == int(customer_id)].index
        
        if len(customer_idx) == 0:
            return jsonify({
                'status': 'error',
                'message': f'Customer with ID {customer_id} not found'
            }), 404
        
        # Get the update data
        update_data = request.json
        
        # Update the customer data
        for key, value in update_data.items():
            if key in df.columns:
                df.loc[customer_idx, key] = value
        
        # Save the updated DataFrame
        df.to_csv(DATA_PATH, index=False)
        
        # Recalculate cluster for this customer if possible
        available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
        if len(available_features) > 0:
            customer_features = df.loc[customer_idx, available_features].dropna()
            
            # Convert features to numeric
            for col in customer_features.columns:
                customer_features[col] = pd.to_numeric(customer_features[col], errors='coerce')
            
            if not customer_features.empty:
                X_scaled = scaler.transform(customer_features)
                cluster = kmeans.predict(X_scaled)[0]
                df.loc[customer_idx, 'cluster'] = cluster
                df.to_csv(DATA_PATH, index=False)
        
        return jsonify({
            'status': 'success',
            'message': f'Customer {customer_id} updated successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/customers', methods=['POST'])
def add_customer():
    """Add a new customer"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Get the new customer data
        new_customer = request.json
        
        # Generate a new customer ID if not provided
        if 'customer id' not in new_customer or not new_customer['customer id']:
            max_id = df['customer id'].max() if 'customer id' in df.columns else 0
            new_customer['customer id'] = int(max_id) + 1
        
        # Add the new customer to the DataFrame
        df = pd.concat([df, pd.DataFrame([new_customer])], ignore_index=True)
        
        # Calculate cluster for the new customer if possible
        available_features = [f for f in CLUSTER_FEATURES if f in new_customer]
        if len(available_features) == len(CLUSTER_FEATURES):
            customer_features = np.array([[float(new_customer[feature]) for feature in CLUSTER_FEATURES]])
            X_scaled = scaler.transform(customer_features)
            cluster = kmeans.predict(X_scaled)[0]
            new_customer['cluster'] = int(cluster)
            
            # Update the last row with the cluster
            df.loc[df.index[-1], 'cluster'] = cluster
        
        # Save the updated DataFrame
        df.to_csv(DATA_PATH, index=False)
        
        return jsonify({
            'status': 'success',
            'message': f'Customer added successfully with ID {new_customer["customer id"]}',
            'customer': new_customer
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/assign_cluster', methods=['POST'])
def assign_cluster():
    """Assign a cluster to a customer based on provided features"""
    try:
        data = request.json
        
        # Check which required features are present
        available_features = [feature for feature in CLUSTER_FEATURES if feature in data]
        missing_features = [feature for feature in CLUSTER_FEATURES if feature not in data]
        
        if len(available_features) < len(CLUSTER_FEATURES) / 2:  # Require at least half of the features
            return jsonify({
                'status': 'error',
                'message': f'Too many missing features: {", ".join(missing_features)}'
            }), 400
        
        # Get average values for missing features from training data
        df = pd.read_csv(DATA_PATH)
        avg_values = {}
        for feature in missing_features:
            if feature in df.columns:
                avg_values[feature] = float(df[feature].mean())
            else:
                avg_values[feature] = 0.0  # Default value if feature not in dataset
        
        # Combine provided data with average values for missing features
        customer_data = {}
        for feature in CLUSTER_FEATURES:
            if feature in data:
                customer_data[feature] = float(data[feature])
            else:
                customer_data[feature] = avg_values[feature]
        
        # Extract features in the correct order and convert to correct types
        feature_values = [float(customer_data[feature]) for feature in CLUSTER_FEATURES]
        customer_array = np.array([feature_values])
        
        # Scale and predict
        X_scaled = scaler.transform(customer_array)
        cluster = int(kmeans.predict(X_scaled)[0])
        
        return jsonify({
            'status': 'success',
            'customer_data': customer_data,
            'cluster': cluster,
            'cluster_description': get_cluster_description(cluster),
            'missing_features': missing_features if missing_features else []
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)