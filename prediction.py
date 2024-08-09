from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            X[col + '_year'] = X[col].dt.year.fillna(0).astype(int)
            X[col + '_month'] = X[col].dt.month.fillna(0).astype(int)
            X[col + '_day'] = X[col].dt.day.fillna(0).astype(int)
            X[col + '_hour'] = X[col].dt.hour.fillna(0).astype(int)
            X.drop(columns=[col], inplace=True)
        return X
    
datetime_columns = ['EDate', 'FollowUp_Date', 'Next_FollowUp1']
preprocessor = Pipeline(steps=[
    ('datetime', DateTimeTransformer(datetime_columns)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


model_pipeline = joblib.load('booking_model.pkl')

# input data
input_data = pd.DataFrame({
    'EDate': [pd.to_datetime('18-07-2024  15:48:24')],
    'Project': ['Sage Golden Spring'],
    'Remark1' : 'lead given by existing customer. .visit done. interested in D block',
    'HandledByEmployee' : 'Ritu Mehta',
    'FollowUp_Date': [pd.to_datetime('18-07-2024  15:50:18')],
    'FollowUp_Mode': ['Telephonic'],
    'Next_FollowUp1': [pd.to_datetime('19-07-2024  20:50:18')],       
    'CustomerGrade': ['Hot'],
    'Marital_Status': ['Married'],
    'Occupation': ['Unknown'],
    'Location': ['Ayodhya Bypass'],
    'Budget': [1250000.0],
    'Source': ['Reference'],
    'EnquiryStage': ['Open'],
    

    
    
})

preprocessed_data = model_pipeline.named_steps['preprocessor'].transform(input_data)
probabilities = model_pipeline.named_steps['classifier'].predict_proba(preprocessed_data)[:, 1]

print('Probability of Booking : ',probabilities)