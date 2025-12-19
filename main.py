# Salary Prediction Model
# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# 1. LOAD DATA

df = pd.read_csv("expected_ctc.csv")
print("Dataset shape:", df.shape)

# 2. DEFINE TARGET

target_col = "Expected_CTC"

# 3. DROP USELESS COLUMNS (CRITICAL FIX)

drop_cols = [
    "IDX",
    "Applicant_ID",
    "Organization",
    "Designation",
    "University_Grad",
    "University_PG",
    "University_PHD",
    "Curent_Location",
    "Preferred_location",
    "Passing_Year_Of_Graduation",
    "Passing_Year_Of_PG",
    "Passing_Year_Of_PHD"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 4. SPLIT FEATURES / TARGET

X = df.drop(columns=[target_col])
y = df[target_col]

# 5. IDENTIFY COLUMN TYPES

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)


# 6. HANDLE MISSING VALUES

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# 7. ENCODE CATEGORICAL VARIABLES

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# =====================================
# 8. TRAIN–TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 9. TRAIN XGBOOST MODEL
# =====================================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# 10. EVALUATION
# =====================================
y_pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2  :", r2_score(y_test, y_pred))

# =====================================
# 11. SAFE PREDICTION FUNCTION
# =====================================
def predict_expected_ctc(input_data: dict):

    input_df = pd.DataFrame(columns=X.columns)

    for key, value in input_data.items():
        if key in input_df.columns:
            input_df.at[0, key] = value

    # Ensure correct dtypes
    for col in numerical_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    # Impute
    input_df[numerical_cols] = num_imputer.transform(input_df[numerical_cols])
    input_df[categorical_cols] = cat_imputer.transform(input_df[categorical_cols])

    # Encode safely
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = input_df[col].apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        input_df[col] = le.transform(input_df[col])

    return model.predict(input_df)[0]

# =====================================
# 12. SAMPLE PREDICTION
# =====================================
sample_candidate = {
    "Total_Experience": 5,
    "Total_Experience_in_field_applied": 3,
    "Current_CTC": 6,
    "No_Of_Companies_worked": 2,
    "Number_of_Publications": 0,
    "Certifications": 1,
    "International_degree_any": "No",
    "Education": "B.Tech",
    "Department": "Engineering",
    "Role": "Software Engineer",
    "Industry": "IT"
}

predicted_salary = predict_expected_ctc(sample_candidate)

print("\n===== PREDICTED EXPECTED CTC =====")
print("Predicted Expected CTC:", predicted_salary)

# =====================================
# 13. SAVE EVERYTHING
# =====================================
joblib.dump(model, "xgb_model.pkl")
joblib.dump(num_imputer, "num_imputer.pkl")
joblib.dump(cat_imputer, "cat_imputer.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(numerical_cols, "numerical_cols.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nModel & preprocessors saved successfully")
