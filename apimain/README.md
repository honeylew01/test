# Running the Project with Docker

To run this project using Docker, follow the steps below:

## Prerequisites

- Ensure Docker is installed on your system.
- The project uses Python 3.10 as specified in the Dockerfile.
- Ensure that your terminal's directory is in the API folder

## Build and Run Instructions

1. Build the Docker image :

   ```bash
   docker build -t customer-segmentation-api .
   ```
2. Run the Docker image :

   ```bash
   docker run -p 8000:8000 customer-segmentation-api
   ```
3. The application will be accessible at `http://localhost:8000/docs`

## Configuration

- The application exposes port `8000` as defined in the Dockerfile
- No additional environment variables are required for this setup.

## Notes

- The `cleaned main dataset.csv` file is included in the project directory for data processing.

For further details, refer to the project documentation or contact the development team.
## Summary of API
- This API is used to deploy a real-time customer segmentation model using the K-means algorithm.
- This API supports immediate segmentation for both real-time new customer additions/queries , as well as 
for real-time updates to customer data of customers who already exist in the database
- Additional helper functions are included to allow for the ability to 
add/delete/update customer data in the database
- Periodic/ Dynamic re-training of the K-means segmentation model using recently updated data is also
supported

## Explanation of features


1. age : int
2. gender : int (Male=0, Female=1)
3. monthly_income : int
4. account_balance : int
5. loyalty_score : int (1-1000) (The higher the loyalty, the higher the loyalty_score)
6. education_level : int (Primary=0 , Secondary=1 , Tertiary=2, Postgrad=3)
7. facebook_interaction : int (1= exposed to facebook ads at least once, 0 otherwise)
8. twitter_interaction : int (1= exposed to twitter ads at least once, 0 otherwise)
9. email_interaction : int (1= clicked email ads at least once, 0 otherwise)
10. instagram_interaction : int (1= exposed to instagram ads at least once, 0 otherwise)
11. total_withdrawal_amount : int (sum across all transactions)
12. total_deposit_amount : int (sum across all transactions)
13. transaction_count : int
14. has_loan : int (False=0, True=1)


## API functions

1. #### get_customer_info_from_database
Takes in a customer id and returns the customer attributes, the cluster that the customer belongs to as well as the corresponding business strategy

#### Features
1. customer_id : int, range from 1 to 20000

#### Example output
```json
{
  "Customer info": {
    "age": 29,
    "gender": 0,
    "income/month": 8074,
    "account balance": 2488,
    "loyalty score": 725,
    "education level": 0,
    "Facebook": 0,
    "Twitter": 1,
    "Email": 1,
    "Instagram": 0,
    "total_withdrawals": 31789,
    "total_deposits": 13472.29,
    "transaction_count": 3,
    "loan": 1,
    "cluster_num": 0,
    "customer id": 56
  },
  "Segmentation info": "Cluster 0: High-Value Power Users",
  "Business strategy": "Upsell bank products to increase profits"
}
```
2. #### update_customer_details_to_database
Updates customer details of **existing** customers in database whenever the field (e.g. monthly_income) is provided. If a particular field is not provided in the request body,
no change is made to that attribute. This function also automatically re-segments customers after each update.


#### Features

1. customer_id: int
2. age: Optional[int] = None
3. gender: Optional[int] = None
4. monthly_income: Optional[int] = None
5. account_balance: Optional[int] = None
6. loyalty_score: Optional[int] = None
7. education_level: Optional[int] = None
8. facebook_interaction: Optional[int] = None
9. twitter_interaction: Optional[int] = None
10. email_interaction: Optional[int] = None
11. instagram_interaction: Optional[int] = None
12. total_withdrawal_amount: Optional[int] = None
13. total_deposit_amount: Optional[int] = None
14. transaction_count: Optional[int] = None
15. has_loan: Optional[int] = None

#### Example input
```json
{
  "customer_id": 7287,
  "total_deposit_amount": 9876,
  "transaction_count": 3
}
```
(For customer with ID 7287, only total deposit amount and transaction count updated,
the rest of the attributes did not change)
#### Example output
```json
{
  "message": "Customer id 7287 updated successfully",
  "updated_customer": {
    "age": 53,
    "gender": 1,
    "income/month": 12770,
    "account balance": 112,
    "loyalty score": 489,
    "education level": 1,
    "Facebook": 1,
    "Twitter": 1,
    "Email": 1,
    "Instagram": 1,
    "total_withdrawals": 1068.94,
    "total_deposits": 9876,
    "transaction_count": 3,
    "loan": 1,
    "cluster_num": 3,
    "customer id": 7287
  }
}
```
3. #### delete_customer_from_database
Takes in a customer ID and deletes the customer from the database. It can no longer be retrieved
or used for periodic retraining of the K-means algorithm, unless added back in.

#### Features
1. customer_id : int, range from 1 to 20000

#### Example output
```json
{
  "message": "Customer with ID 787 deleted successfully"
}
```
4. #### add_customer_to_database

Adds customer to database. The customer added must have a unique ID from all
other customers existing in the database. Also automatically segments the newly added customer,
segmentation details can be accessed using get_customer_info_from_database. All
fields are required.


#### Features

1. customer_id: int (ID must be unique)
2. age:int
3. gender:int
4. monthly_income:int
5. account_balance:int
6. loyalty_score:int
7. education_level:int
8. facebook_interaction:int
9. twitter_interaction:int
10. email_interaction:int
11. instagram_interaction:int
12. total_withdrawal_amount:int
13. total_deposit_amount:int
14. transaction_count:int
15. has_loan:int

#### Example output
```json
{
  "message": "Customer successfully added to databank",
  "updated_customer": {
    "age": 45,
    "gender": 0,
    "income/month": 8283,
    "account balance": 9723,
    "loyalty score": 342,
    "education level": 3,
    "Facebook": 1,
    "Twitter": 0,
    "Email": 1,
    "Instagram": 0,
    "total_withdrawals": 2839,
    "total_deposits": 778,
    "transaction_count": 7,
    "loan": 0,
    "cluster_num": 0,
    "customer id": 20001
  }
}
```


5. #### predict_customer_segment_in_real_time

Takes in customer attributes (without customer id) and predicts the cluster the customer belongs to, as well as the 
corresponding business strategy. Similar to add_customer_to_database, but this function
does not add the requested customer to the database.

#### Features


1. age:int
2. gender:int
3. monthly_income:int
4. account_balance:int
5. loyalty_score:int
6. education_level:int
7. facebook_interaction:int
8. twitter_interaction:int
9. email_interaction:int
10. instagram_interaction:int
11. total_withdrawal_amount:int
12. total_deposit_amount:int
13. transaction_count:int
14. has_loan:int

#### Example input 

```json
{
  "age": 34,
  "gender": 1,
  "monthly_income": 9763,
  "account_balance": 26514,
  "loyalty_score": 45,
  "education_level": 2,
  "facebook_interaction": 1,
  "twitter_interaction": 0,
  "email_interaction": 0,
  "instagram_interaction": 0,
  "total_withdrawal_amount": 926,
  "total_deposit_amount": 0,
  "transaction_count": 1,
  "has_loan": 1
}
```
#### Example output

```json
"Cluster 1: Value-Driven Frequent Users , Business strategy: Build loyalty to reduce churn"
```

#### 6.  retrain_model_with_latest_data

Retrains the k-means algorithm using the current database. It is used
to periodically retrain and re-segment customers after changes are made to the
database, such as adding/deleting/updating customers in the database.

#### Features
No features, just click execute


#### Example output
```json
"Model successfully updated"
```

This is temporary for now, while working out integration to streamlit
```
streamlit run app.py
```

