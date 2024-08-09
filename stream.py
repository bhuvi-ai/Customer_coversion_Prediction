import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_URL = "http://192.168.0.194:8000/predict"

def main():
    st.title("Booking Prediction")

    # Input fields
    EDate = st.text_input("EDate (dd-mm-yyyy HH:MM:SS)", value=datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    Project = st.text_input("Project")
    Remark1 = st.text_area("Remark1")
    HandledByEmployee = st.text_input("HandledByEmployee")
    FollowUp_Date = st.text_input("FollowUp_Date (dd-mm-yyyy HH:MM:SS)", value=datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    FollowUp_Mode = st.selectbox("FollowUp_Mode", ["Telephonic", "Email", "In-Person","Whatsapp"])
    Next_FollowUp1 = st.text_input("Next_FollowUp1 (dd-mm-yyyy HH:MM:SS)", value=(datetime.now() + pd.DateOffset(days=1)).strftime("%d-%m-%Y %H:%M:%S"))
    CustomerGrade = st.selectbox("CustomerGrade", ["Unknown","Hot","Proposal", "Opportunity", "Cold"])
    Marital_Status = st.selectbox("Marital_Status", ["Married", "Single"])
    Occupation = st.text_input("Occupation")
    Location = st.text_input("Location")
    Budget = st.number_input("Budget", min_value=0.0, format="%.2f")
    Source = st.text_input("Source")
    EnquiryStage = st.selectbox("EnquiryStage", ["Open", "Pending","Site Visit - Proposed",
"Lead Confirmation",
"Home Visit - Done",
"Office Visit - Done",
"Finalization",
"Negotiation",
"Finalization",
"Site Visit - Done",
"Negotiation - Proposed"])

    data = {
        "EDate": [EDate],
        "Project": [Project],
        "Remark1": [Remark1],
        "HandledByEmployee": [HandledByEmployee],
        "FollowUp_Date": [FollowUp_Date],
        "FollowUp_Mode": [FollowUp_Mode],
        "Next_FollowUp1": [Next_FollowUp1],
        "CustomerGrade": [CustomerGrade],
        "Marital_Status": [Marital_Status],
        "Occupation": [Occupation],
        "Location": [Location],
        "Budget": [Budget],
        "Source": [Source],
        "EnquiryStage": [EnquiryStage],
    }

    input_df = pd.DataFrame(data)

    if st.button("Predict"):
        response = requests.post(API_URL, json=input_df.to_dict(orient="list"))
        if response.status_code == 200:
            result = response.json()
            st.success(f"Probability of Booking: {result['probability_of_booking'][0]:.4f}")
        else:
            st.error("Error: Unable to get prediction.")

if __name__ == "__main__":
    main()
