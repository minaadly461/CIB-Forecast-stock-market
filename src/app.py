import streamlit as st
import requests

st.set_page_config(
    page_title="CIB Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 CIB Stock Prediction Dashboard")

st.write(
    "Predict the next trading day's CIB closing price."
)

st.subheader("Stock Input")

open_price = st.number_input(
    "Open",
    min_value=0.0
)

high_price = st.number_input(
    "High",
    min_value=0.0
)

low_price = st.number_input(
    "Low",
    min_value=0.0
)

close_price = st.number_input(
    "Close",
    min_value=0.0
)

if st.button("Predict"):

    data = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:
            result = response.json()

            prediction = result["predicted_next_day_close"]

            st.success(
                f"Predicted Next-Day Close: {prediction:.2f}"
            )

        else:
            st.error(
                f"Prediction failed. Status code: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the CIB Prediction API."
        )