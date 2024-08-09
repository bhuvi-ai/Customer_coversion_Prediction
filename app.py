from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from transformers import DateTimeTransformer 

app = FastAPI()

model_pipeline = joblib.load('booking_model.pkl')

class InputData(BaseModel):
    EDate: list[str]
    Project: list[str]
    Remark1: list[str]
    HandledByEmployee: list[str]
    FollowUp_Date: list[str]
    FollowUp_Mode: list[str]
    Next_FollowUp1: list[str]
    CustomerGrade: list[str]
    Marital_Status: list[str]
    Occupation: list[str]
    Location: list[str]
    Budget: list[float]
    Source: list[str]
    EnquiryStage: list[str]

@app.post('/predict')
def predict(input_data: InputData):
    input_df = pd.DataFrame({
        "EDate": input_data.EDate,
        "Project": input_data.Project,
        "Remark1": input_data.Remark1,
        "HandledByEmployee": input_data.HandledByEmployee,
        "FollowUp_Date": input_data.FollowUp_Date,
        "FollowUp_Mode": input_data.FollowUp_Mode,
        "Next_FollowUp1": input_data.Next_FollowUp1,
        "CustomerGrade": input_data.CustomerGrade,
        "Marital_Status": input_data.Marital_Status,
        "Occupation": input_data.Occupation,
        "Location": input_data.Location,
        "Budget": input_data.Budget,
        "Source": input_data.Source,
        "EnquiryStage": input_data.EnquiryStage,
    })

    preprocessed_data = model_pipeline.named_steps['preprocessor'].transform(input_df)

    probabilities = model_pipeline.named_steps['classifier'].predict_proba(preprocessed_data)[:, 1]

    return {"probability_of_booking": probabilities.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='192.168.0.194', port=8000)

