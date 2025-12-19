import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =====================================
# LOAD MODEL & PREPROCESSORS
# =====================================
model = joblib.load("xgb_model.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")
label_encoders = joblib.load("label_encoders.pkl")
numerical_cols = joblib.load("numerical_cols.pkl")
categorical_cols = joblib.load("categorical_cols.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="CTC Prediction", page_icon="💼")
st.title("Expected CTC Prediction App")

st.write("Enter candidate details to predict Expected CTC")

# =====================================
# USER INPUTS
# =====================================
user_input = {}

st.subheader("Experience Details")
user_input["Total_Experience"] = st.number_input(
    "Total Experience (years)", 0.0, 40.0
)

user_input["Total_Experience_in_field_applied"] = st.number_input(
    "Experience in Applied Field (years)", 0.0, 40.0
)

st.subheader("Current Profile")
user_input["Current_CTC"] = st.number_input("Current CTC", 0.0)
user_input["No_Of_Companies_worked"] = st.number_input(
    "No. of Companies Worked", 0, 20
)

user_input["Number_of_Publications"] = st.number_input(
    "Number of Publications", 0, 50
)

user_input["Certifications"] = st.number_input(
    "Number of Certifications", 0, 50
)

user_input["International_degree_any"] = st.selectbox(
    "International Degree", ["No", "Yes"]
)

st.subheader("Education & Job Details")

user_input["Education"] = st.selectbox(
    "Education", label_encoders["Education"].classes_
)

user_input["Department"] = st.selectbox(
    "Department", label_encoders["Department"].classes_
)

user_input["Role"] = st.selectbox(
    "Role", label_encoders["Role"].classes_
)

user_input["Industry"] = st.selectbox(
    "Industry", label_encoders["Industry"].classes_
)

# =====================================
# PREDICTION
# =====================================
if st.button("Predict Expected CTC"):

    input_df = pd.DataFrame(columns=feature_columns)

    for key, value in user_input.items():
        input_df.at[0, key] = value

    # Type casting
    for col in numerical_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    # Imputation
    input_df[numerical_cols] = num_imputer.transform(input_df[numerical_cols])
    input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])

    # Safe encoding
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = input_df[col].apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        input_df[col] = le.transform(input_df[col])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Expected CTC: ₹ {prediction:,.2f}")
