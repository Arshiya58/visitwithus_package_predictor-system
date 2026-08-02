import os
import streamlit as st
import pandas as pd
import joblib


# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model_v1.joblib")
model = joblib.load(model_path)

model = joblib.load(model_path)

st.set_page_config(page_title="Visit With Us", page_icon="✈️")

st.title(" Tourism Package Predictor")

st.write("""
Predict whether a customer is likely to purchase the Wellness Tourism Package.
Enter the customer details below and click **Predict**.
""")

# -------------------------
# Customer Details
# -------------------------

age = st.number_input("Age", 18, 100, 30)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration Of Pitch (Minutes)",
    0,
    1000,
    15
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number Of Persons Visiting",
    1,
    10,
    2
)

number_of_followups = st.number_input(
    "Number Of Followups",
    0,
    10,
    2
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

number_of_trips = st.number_input(
    "Number Of Trips",
    0,
    20,
    2
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

children = st.number_input(
    "Children Visiting",
    0,
    5,
    0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    1000,
    500000,
    30000
)

# -------------------------
# Prediction
# -------------------------

input_df = pd.DataFrame([{

    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": monthly_income

}])

if st.button("Predict Purchase"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(" Customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.error(" Customer is unlikely to purchase the Wellness Tourism Package.")
