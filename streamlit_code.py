import math
import pickle
from datetime import date

import pandas as pd
import streamlit as st

st.title("Prediction orders Rotterdam")

@st.cache_data
def forecast_api(df):
    with open("VAR_Prophet_Model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    predictions = loaded_model.predict(df)
    return predictions

def create_test(date, rainfall):
    df = pd.DataFrame({"ds": [date], "add1": [rainfall * 0.1]})
    return df

# create a Streamlit app
def main():
    min_date = date(2023, 1, 1)
    max_date = date(2023, 12, 31)
    future_date = st.date_input(
        "Select a date in 2023", min_value=min_date, max_value=max_date, value=min_date
    )
    rainfall = st.number_input(
        "Expected rain in mm: (if predicted rain is lower than 0.5: fill in -1)", value=0
    )
    if rainfall < -1:
        st.warning("Negative numbers are not allowed.")
    else:
        output = create_test(future_date, rainfall)
    prediction = forecast_api(output)
