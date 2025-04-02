## Running the model training python script using Docker
Please ensure you have Docker installed and the repository cloned to your local device.
1. Navigate to the file directory
  ```bash
  cd DSA3101-Group-Project/GrpB_models/b5_churn_prediction
  ```

2. Build the image
  ```bash
  docker build -t churn-model .
  ```
3. Run the container (Note: the following code assumes that the repository is in your local C drive)
  ```bash
  docker run --rm -v ${PWD}/C:/DSA3101-Group-Project/GrpB_models/b5_churn_prediction churn-model
  ```
4. If successful, you will see
  ```bash
  "Model training file executed successfully!"
  ```