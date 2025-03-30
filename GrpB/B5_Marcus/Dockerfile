FROM python:3.10

WORKDIR /GrpB/B5_Marcus

COPY . /GrpB/B5_Marcus

RUN pip install --no-cache-dir pandas joblib scikit-learn imbalanced-learn openpyxl xlrd


CMD ["python", "churn_mod_train.py"]
