import streamlit as st
import pandas as pd
import pickle

# Page settings

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load cleaned dataset

car = pd.read_csv("cleaned_car_data.csv")

# Load trained model

model = pickle.load(
    open("LinearRegressionModel.pkl", "rb")
)

# Main title

st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>🚗 Car Price Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>Find the estimated resale price of your car instantly</h4>",
    unsafe_allow_html=True
)

st.write("---")

# Car name dropdown

name = st.selectbox(
    "🚘 Select Car Name",
    sorted(car["name"].unique())
)

# Company dropdown

company = st.selectbox(
    "🏢 Select Company",
    sorted(car["company"].unique())
)

# Car age input

car_age = st.slider(
    "📅 Car Age (Years)",
    min_value=0,
    max_value=30,
    value=5
)

# Kilometers driven

kms_driven = st.slider(
    "🛣️ Kilometers Driven",
    min_value=0,
    max_value=200000,
    value=30000,
    step=1000
)

# Fuel type dropdown

fuel_type = st.selectbox(
    "⛽ Fuel Type",
    sorted(car["fuel_type"].unique())
)

st.write("---")

# Predict button

if st.button("💰 Predict Price"):

    # Create dataframe

    input_data = pd.DataFrame({
        "name": [name],
        "company": [company],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type],
        "car_age": [car_age]
    })

    # Predict price

    prediction = model.predict(input_data)[0]

    # Prevent negative values

    prediction = max(0, prediction)

    st.balloons()

    # Show result

    st.success(
        f"Estimated Car Price: ₹ {prediction:,.0f}"
    )

    st.info(
        "Prediction is based on machine learning model trained on used car dataset."
    )