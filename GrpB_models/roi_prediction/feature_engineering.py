
import pandas as pd
from data_cleaning import DataCleaning
import holidays

class FeatureEngineering:
    
        def __init__(self, file_path):
            self.file_path = file_path
    
        def add_weekdayweekend_to_date(self):
            df = DataCleaning(self.file_path).clean_data()
            df['Date'] = pd.to_datetime(df['Date'])
            df["Day_Type"] = df["Date"].apply(lambda x: "Weekend" if x.weekday() >= 5 else "Weekday")
            return df
        
        def add_features(self):
            us_holidays_2021 = set(holidays.US(years=2021).keys())
            df = self.add_weekdayweekend_to_date()
            df['Click-Through_Rate'] = df['Clicks'] / df['Impressions']
            df['Cost_Per_Click'] = df['Acquisition_Cost'] / df['Clicks']
            df['Is_Holiday'] = df['Date'].isin(us_holidays_2021).astype(int)
            df = df.drop('Date', axis = 1)
            df = df.drop('Impressions', axis = 1)
            df = df.drop('Clicks', axis = 1)
            return df
