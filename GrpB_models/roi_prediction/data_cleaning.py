
import pandas as pd
import numpy as np

class DataCleaning:

    def __init__(self, file_path):
        self.file_path = file_path

    def clean_data(self):
        df = pd.read_csv(self.file_path)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.drop('Company', axis = 1)
        df = df.drop('Location', axis = 1)
        df = df.drop('Language', axis = 1)
        df = df.drop('Customer_Segment', axis = 1)
        df = df.drop('Campaign_ID', axis = 1)
        df['Duration'] = df['Duration'].str.replace(' days', '').astype(int)
        df['Acquisition_Cost'] = df['Acquisition_Cost'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
        df['Date'] = pd.to_datetime(df['Date'])

        return df
