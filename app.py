import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("model/car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("Car Price Prediction App")
st.write("Enter car details to get an estimated selling price.")

# ---- User Inputs (UI replaces input()) ----
car_name = st.text_input("Car Name")

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    step=1
)

km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner Type",
    ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
)

# ---- Encoding (must match training) ----
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
transmission_map = {"Manual": 0, "Automatic": 1}
owner_map = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4
}

# ---- Prediction ----
if st.button("Predict Selling Price"):
    input_data = np.array([[
        year,
        km_driven,
        fuel_map[fuel_type],
        transmission_map[transmission],
        owner_map[owner]
    ]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Selling Price: ₹ {int(prediction[0]):,}")
